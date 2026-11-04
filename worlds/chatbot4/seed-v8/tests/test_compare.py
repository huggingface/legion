import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare(*args):
    return subprocess.run([sys.executable, str(ROOT / "tools/compare.py"), *args], capture_output=True,
                          text=True, cwd=ROOT, timeout=600, check=False)


class CompareTest(unittest.TestCase):
    def test_a_checkout_compared_with_itself_changes_nothing(self):
        result = compare(str(ROOT), str(ROOT), "--sample", "6", "--seed", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no reply changed", result.stdout)

    def test_score_counts_head_better_and_worse_with_the_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "k.key"
            key.write_text(json.dumps({"seed": 3, "items": ["a", "b", "c"],
                                       "key": ["head=1", "head=2", "head=1"]}))
            result = compare("--score", str(key), "1 1 =")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("head better 1, base better 1, same 1", result.stdout)
        self.assertIn("head worse on: b", result.stdout)


if __name__ == "__main__":
    unittest.main()
