"""SmolTalkBench as an Inspect task: AlpacaEval-style win rates against reference replies.

    inspect eval experiments/run4/eval/smoltalk_bench.py@smoltalk_bench_gold --model <model>

The model answers each instruction of the gold config once, and the judge (the `grader` model
role) compares its reply with the reference as AlpacaEval 2.0 does, see judge.py. The score
of a sample is the probability that the model's reply is better; NaN when the judge named
neither output. The judge needs top logprobs, and DeepSeek's thinking is turned off so
that its first token is the answer. The gamed baselines of the gold_gamed config, judged by
the same judge, calibrate the LC win rate.
"""

import functools
import math
import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, FieldSpec, hf_dataset
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import SampleScore, Score, Scorer, Target, metric, scorer
from inspect_ai.solver import TaskState, generate

sys.path.insert(0, str(Path(__file__).parent))
import judge

BENCH = "HuggingFaceSwarms/SmolTalkBench"
BENCH_REVISION = "dbda1eaf1e9839f51e7ef073783dacd226d7ebfb"
GAMED = "gold_gamed"
JUDGE = "hf-inference-providers/deepseek-ai/DeepSeek-V4.1-Flash:fireworks-ai"
# AlpacaEval's settings (one token, temperature 1, top-5 logprobs), without DeepSeek's thinking.
JUDGE_CONFIG = GenerateConfig(max_tokens=1, temperature=1.0, logprobs=True, top_logprobs=5,
                              extra_body={"chat_template_kwargs": {"thinking": False}})


@task
def smoltalk_bench_gold() -> Task:
    """Win rate against the gold references (Llama-3.1-70B-Instruct's replies in smoltalk)."""
    return smoltalk_bench(hf_dataset(
        BENCH, name="gold", split="test", revision=BENCH_REVISION,
        sample_fields=FieldSpec(input="instruction", target="output", id="id",
                                metadata=["topic", "category", "group", "generator"])),
        gamed=GAMED)


def smoltalk_bench(dataset: Dataset, gamed: str | None = None) -> Task:
    """`gamed` names the SmolTalkBench config whose gamed baselines calibrate the LC win rate."""
    return Task(dataset=dataset, solver=generate(), scorer=alpaca_eval_judge(),
                metrics=[alpaca_eval_win_rates(gamed)])


@functools.cache
def gamed_baselines(config: str) -> tuple[str, list[tuple[list[float], list[int], list[int]]]]:
    """The judge of a config of gamed baselines, and each baseline's preferences, output
    lengths and reference lengths."""
    from datasets import load_dataset

    rows = load_dataset(BENCH, config, split="test", revision=BENCH_REVISION)
    judges = set(rows["judge"])
    assert len(judges) == 1, f"{config} was judged by {judges}"
    baselines = defaultdict(lambda: ([], [], []))
    for row in rows:
        prefs, output_lengths, reference_lengths = baselines[row["variant"]]
        prefs.append(row["preference"])
        output_lengths.append(row["output_length"])
        reference_lengths.append(row["reference_length"])
    return judges.pop(), list(baselines.values())


@metric
def alpaca_eval_win_rates(gamed: str | None = None):
    """AlpacaEval's win rates (percent) over the judged samples; Inspect leaves out NaN scores.
    With `gamed`, the LC win rate is calibrated by that config's gamed baselines, and NaN if
    another judge scored the samples."""
    def compute(scores: list[SampleScore]) -> dict[str, float]:
        meta = [s.score.metadata for s in scores]
        prefs = [m["preference"] for m in meta]
        lengths = [m["output_length"] for m in meta], [m["reference_length"] for m in meta]
        if gamed is None:
            lc = judge.length_controlled(prefs, *lengths)
        else:
            gamed_judge, baselines = gamed_baselines(gamed)
            lc = (judge.length_controlled(prefs, *lengths, gamed=baselines)
                  if {m["judge"] for m in meta} == {gamed_judge} else
                  {"length_controlled_win_rate": math.nan, "lc_standard_error": math.nan})
        return {k: float(v) for k, v in (judge.win_rates(prefs) | lc).items()}
    return compute


@scorer(metrics=[alpaca_eval_win_rates()])
def alpaca_eval_judge() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        instruction, output, reference = state.input_text, state.output.completion, target.text
        switched = judge.is_switched(instruction)
        system, user = judge.judge_messages(
            instruction, *((output, reference) if switched else (reference, output)))
        grader = get_model(role="grader", default=JUDGE)
        result = await grader.generate([ChatMessageSystem(content=system["content"]),
                                        ChatMessageUser(content=user["content"])],
                                       config=JUDGE_CONFIG)
        logprobs = result.choices[0].logprobs if result.choices else None
        top = {t.token: t.logprob for t in logprobs.content[0].top_logprobs} \
            if logprobs and logprobs.content else {}
        preference = judge.preference(top)
        if switched:
            preference = 3 - preference
        return Score(
            value=preference - 1, answer=output, explanation=result.completion,
            metadata={"preference": preference, "switched": switched, "judge": str(grader),
                      "output_length": len(output), "reference_length": len(reference),
                      "judge_top_logprobs": top})
    return score

