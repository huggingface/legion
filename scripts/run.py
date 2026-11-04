"""Run a Shepherd experiment.

uv run scripts/run.py --config examples/pi-hf-qwen.yaml
"""

import argparse
import asyncio
import sys
from pathlib import Path

from shepherd.config import Config
from shepherd.launch import run_experiment
from shepherd.results import safe_error


async def main(config: Config):
    directory, exit_code = await run_experiment(config)
    print(f"Run manifest: {directory / 'run.json'}")
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML")
    args = parser.parse_args()

    try:
        config = Config.load(args.config)
        exit_code = asyncio.run(main(config))
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        exit_code = 1

    sys.exit(exit_code)
