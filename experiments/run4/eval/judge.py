"""AlpacaEval's pairwise judge and win rates, ported from tatsu-lab/alpaca_eval (Apache-2.0):
https://github.com/tatsu-lab/alpaca_eval

This follows AlpacaEval 2.0's default annotator, weighted_alpaca_eval_gpt4_turbo:

- The judge sees an instruction and two outputs named "m" and "M", and answers with one token.
  The preference is read from that token's top-5 logprobs: P(m) / (P(m) + P(M)).
- output_1 is the reference and output_2 the model, swapped for about half of the instructions.
  The swap is seeded by the instruction alone, so every model sees the same order on a prompt.
- Preferences are on AlpacaEval's scale: 1 when the reference wins, 2 when the model wins.
- The win rate is the mean of preference - 1, in percent, with its standard error. The discrete
  win rate rounds each preference first.
- The length-penalised win rate applies the length penalty of WildBench (Lin et al., 2024,
  https://github.com/allenai/WildBench): a reply more than 500 characters longer than its
  reference counts as a tie at most. WildBench only turns its judge's "slightly better" into a
  tie; this judge's preference is a probability, so every win of such a reply is capped.
  WildBench's other half, a much longer reference's win becoming a tie, is left out, so a short
  reply is never excused.
- The length-controlled win rate is the `length_controlled_minimal` GLM: a logistic regression
  of the preference on tanh(standardised length difference) and an intercept, predicted at
  equal length. AlpacaEval's default GLM also needs instruction difficulties, which are only
  fitted for its own 805 instructions.
"""

import math
import random
import statistics

SYSTEM = (
    "You are a highly efficient assistant, who evaluates and selects the best large language "
    "model (LLMs) based on the quality of their responses to a given instruction. This process "
    "will be used to create a leaderboard reflecting the most accurate and human-preferred answers."
)
USER = '''I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{
    "instruction": """{instruction}""",
}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{
    {
        "model_identifier": "m",
        "output": """{output_1}"""
    },
    {
        "model_identifier": "M",
        "output": """{output_2}"""
    }
}

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): m or M.

## Best Model Identifier'''
SEED = 0
# WildBench's K: how many characters longer than its reference a reply can be and still win
LENGTH_PENALTY_CHARS = 500
# How close to 0 or 1 a win probability gets in the GLM fit
PROB_EPS = 1e-12


def is_switched(instruction: str) -> bool:
    """Whether the model's output goes first, as AlpacaEval's RandomSwitchTwoColumnsProcessor."""
    return random.Random(f"is_switched_outputs{instruction}{SEED}").choices([False, True])[0]


def judge_messages(instruction: str, output_1: str, output_2: str) -> list[dict]:
    """The judge's system and user messages, placeholders filled in template order."""
    user = USER
    for key, value in (("instruction", instruction), ("output_1", output_1),
                       ("output_2", output_2)):
        user = user.replace("{" + key + "}", value, 1)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def preference(top_logprobs: dict[str, float]) -> float:
    """AlpacaEval's weighted preference from the first token's top logprobs: 1 when output_1
    ("m") is certainly better, 2 when output_2 ("M") is, NaN when neither token is listed."""
    if "m" not in top_logprobs and "M" not in top_logprobs:
        return math.nan
    p_m = math.exp(top_logprobs.get("m", -math.inf))
    p_big_m = math.exp(top_logprobs.get("M", -math.inf))
    return 2 - p_m / (p_m + p_big_m)


def win_rates(preferences: list[float]) -> dict[str, float]:
    """AlpacaEval's head-to-head summary of model-vs-reference preferences in [1, 2]."""
    wins = [p - 1 for p in preferences if not math.isnan(p)]
    discrete = [0.5 if w == 0.5 else float(round(w)) for w in wins]
    n = len(wins)
    return {
        "win_rate": 100 * statistics.fmean(wins) if n else math.nan,
        "standard_error": 100 * statistics.stdev(wins) / math.sqrt(n) if n > 1 else math.nan,
        "discrete_win_rate": 100 * statistics.fmean(discrete) if n else math.nan,
        "n_wins": sum(w > 0.5 for w in wins),
        "n_wins_base": sum(w < 0.5 for w in wins),
        "n_draws": sum(w == 0.5 for w in wins),
        "n_total": n,
    }


def length_penalised_win_rate(preferences: list[float], output_lengths: list[int],
                               reference_lengths: list[int]) -> dict[str, float]:
    """The win rate (percent) with WildBench's length penalty, one-sided: the preference of a
    reply more than LENGTH_PENALTY_CHARS characters longer than its reference is capped at a
    tie. Lengths are in characters."""
    wins = [min(p - 1, 0.5) if out - ref > LENGTH_PENALTY_CHARS else p - 1
            for p, out, ref in zip(preferences, output_lengths, reference_lengths, strict=True)
            if not math.isnan(p)]
    n = len(wins)
    return {"length_penalised_win_rate": 100 * statistics.fmean(wins) if n else math.nan,
            "lp_standard_error": 100 * statistics.stdev(wins) / math.sqrt(n) if n > 1 else math.nan}


def _logloss_continuous(y_true, y_pred, true_prob, true_sample_weight=None):
    import numpy as np

    y_true = np.where(y_true == 1, true_prob, 1 - true_prob)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    losses = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    if true_sample_weight is not None:
        losses = losses * true_sample_weight
    return -np.mean(losses)


def _fit(x, wins, groups, *, C=None):
    """AlpacaEval's fit_LogisticRegressionCV with probabilities as targets: each row appears once
    as a win, weighted by the preference, and once as a loss. With C, a plain fit without CV."""
    import numpy as np
    import sklearn
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GroupKFold

    # Keep every class weight above zero: a CV training fold without a single win (or loss)
    # otherwise misfits in liblinear and skews the choice of C (one win in 1,000 came out at 17%)
    wins = np.clip(wins, PROB_EPS, 1 - PROB_EPS)
    x2, y2 = np.concatenate([x, x]), np.concatenate([np.ones(len(x)), np.zeros(len(x))])
    prob = np.concatenate([wins, 1 - wins])
    # AlpacaEval's penalty="l1", in the l1_ratio form that sklearn >= 1.8 asks for
    kwargs = {"random_state": 123, "dual": False, "solver": "liblinear", "fit_intercept": False}
    with sklearn.config_context(enable_metadata_routing=True):
        if C is not None:
            return LogisticRegression(C=C, l1_ratio=1, **kwargs).fit(x2, y2, sample_weight=prob)
        scorer = make_scorer(_logloss_continuous, response_method="predict_proba",
                             greater_is_better=False).set_score_request(
                                 true_sample_weight=True, true_prob=True)
        model = LogisticRegressionCV(cv=GroupKFold(n_splits=5), scoring=scorer, l1_ratios=(1,),
                                     use_legacy_attributes=False, **kwargs)
        model.set_fit_request(sample_weight=True)
        return model.fit(x2, y2, sample_weight=prob, groups=np.concatenate([groups, groups]),
                         true_sample_weight=None, true_prob=prob)


def _equal_length_win_rate(x, wins, groups, *, C=None):
    """The GLM's win probability at no length difference, where only the intercept is left,
    and the fitted model. Without a single win, or a single loss, there is nothing to fit (the
    GLM would sit at 50%), so the rate is 0 or 1."""
    if not wins.any():
        return 0.0, None
    if (wins == 1).all():
        return 1.0, None
    model = _fit(x, wins, groups, C=C)
    return 1 / (1 + math.exp(-model.coef_[0][1])), model


def length_controlled(preferences: list[float], output_lengths: list[int],
                      reference_lengths: list[int], *, bootstrap: int = 200) -> dict[str, float]:
    """AlpacaEval's length_controlled_minimal win rate. Lengths are in characters. Its standard
    error is a bootstrap over prompts (AlpacaEval's is 0 by construction for this GLM)."""
    import numpy as np

    prefs = np.asarray(preferences, dtype=float)
    keep = ~np.isnan(prefs)
    if keep.sum() < 10:
        return {"length_controlled_win_rate": math.nan, "lc_standard_error": math.nan}
    delta = np.asarray(reference_lengths, dtype=float) - np.asarray(output_lengths, dtype=float)
    std = delta.std(ddof=1)
    x = np.stack([np.tanh(delta / std if std > 0 else 0 * delta), np.ones(len(delta))], axis=1)
    x, wins = x[keep], prefs[keep] - 1
    lc, model = _equal_length_win_rate(x, wins, np.arange(len(wins)))
    # A one-sided sample has no fit, and neither has any of its resamples.
    C = float(np.ravel(model.C_)[0]) if model is not None else None
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(bootstrap):
        idx = rng.integers(0, len(wins), len(wins))
        boots.append(_equal_length_win_rate(x[idx], wins[idx], idx, C=C)[0])
    return {"length_controlled_win_rate": 100 * lc,
            "lc_standard_error": 100 * float(np.std(boots, ddof=1)) if boots else math.nan}
