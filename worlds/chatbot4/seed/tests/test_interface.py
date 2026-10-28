import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class InterfaceTest(unittest.TestCase):
    def test_one_nonempty_reply_per_input(self):
        result = subprocess.run([sys.executable, "-m", "bot"], input="hi\n\nbye\n",
                                text=True, capture_output=True, cwd=ROOT, timeout=300, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 3)
        self.assertTrue(all(line.strip() for line in result.stdout.splitlines()))


if __name__ == "__main__":
    unittest.main()
