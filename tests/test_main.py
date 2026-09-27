import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class MainEntryPointTests(unittest.TestCase):
    def test_main_prints_launched_and_succeeds(self):
        completed = subprocess.run(
            [sys.executable, "main.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "launched")
        self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
