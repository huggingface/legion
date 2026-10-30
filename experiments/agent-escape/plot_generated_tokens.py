"""Plot only generated completion tokens per configured agent, excluding all input."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_token_time import generated_rows
from sweep import plot_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = generated_rows(json.loads(args.results.read_text()))
    fig = plot_results(rows, generated_only=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(args.results.parent / f"normalized-generated-tokens.{ext}",
                    dpi=180, facecolor="white")
    plt.close(fig)
    data = [{k: r[k] for k in ("difficulty", "agents", "trial", "success",
                              "measurement_valid", "generated_tokens", "normalized_tokens")}
            for r in rows]
    (args.results.parent / "generated-token-results.json").write_text(
        json.dumps(data, indent=2) + "\n")
    print(f"Plotted {sum(r['success'] for r in rows)} valid successful trials")


if __name__ == "__main__":
    main()
