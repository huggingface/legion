"""Convert AgentEscape trajectory Parquet to a directly consumable SFT messages dataset."""

import argparse
import json
from pathlib import Path

from datasets import Dataset, Features, Json, List, Value

# A message or tool is an open-ended mapping. Struct inference would materialize
# the union of keys, changing tool arguments and the definitions shown to the model.
FEATURES = Features({
    "id": Value("string"),
    "messages": List(Json()),
    "tools": List(Json()),
    "metadata": Json(),
})


def conversation(row):
    messages = json.loads(row["prompt"]) + json.loads(row["completion"])
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            if any(part["type"] != "text" for part in content):
                raise ValueError(f"Non-text content in {row['id']}")
            message["content"] = "".join(part["text"] for part in content)
    return {
        "id": row["id"],
        "messages": messages,
        "tools": json.loads(row["tools"]),
        "metadata": json.loads(row["metadata"]),
    }


def export_messages(source_path, output_path):
    source = Dataset.from_parquet(str(source_path))
    dataset = source.map(
        conversation, remove_columns=source.column_names, features=FEATURES, writer_batch_size=32
    )
    dataset.to_parquet(output_path, batch_size=32)
    reloaded = Dataset.from_parquet(str(output_path))
    for raw, stored in zip(source, reloaded, strict=True):
        if stored != conversation(raw):
            raise ValueError(f"Parquet round trip changed {raw['id']}")
    return reloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", type=Path, help="examples.parquet from trajectory export")
    parser.add_argument("output", type=Path, help="Output messages Parquet file")
    parser.add_argument("--push-to-hub", metavar="NAMESPACE/DATASET", help="Publish privately")
    args = parser.parse_args()
    dataset = export_messages(args.trajectories, args.output)
    if args.push_to_hub:
        commit = dataset.push_to_hub(args.push_to_hub, private=True, split="train")
        print(commit.oid)
