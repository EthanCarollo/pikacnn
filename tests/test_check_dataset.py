import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.check_dataset import DEFAULT_DATASET_DIR, scan_dataset


class CheckDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def touch(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic file; image contents are never decoded")

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "tools" / "check_dataset.py"), *map(str, args)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_default_path_points_into_repository_dataset(self):
        self.assertEqual(DEFAULT_DATASET_DIR, ROOT / "data" / "pokemon-128aug")

    def test_success_counts_case_insensitive_extensions_in_sorted_class_order(self):
        self.touch(self.root / "zebra" / "b.PNG")
        self.touch(self.root / "zebra" / "a.jpg")
        self.touch(self.root / "Alpha" / "one.JPEG")
        self.touch(self.root / "zebra" / "notes.txt")

        result = scan_dataset(self.root)

        self.assertTrue(result["valid"])
        self.assertIsNone(result["error"])
        self.assertEqual([c["name"] for c in result["classes"]], ["Alpha", "zebra"])
        self.assertEqual([c["image_count"] for c in result["classes"]], [1, 2])
        self.assertEqual(result["total_images"], 3)
        self.assertEqual(result["classes"][1]["ignored_files"], ["notes.txt"])

    def test_scan_reports_missing_non_directory_empty_and_classless_paths(self):
        missing = scan_dataset(self.root / "missing")
        self.assertIn("missing path", missing["error"])

        regular_file = self.root / "not-a-directory"
        regular_file.write_text("x", encoding="utf-8")
        self.assertIn("invalid path", scan_dataset(regular_file)["error"])

        empty_root = self.root / "empty"
        empty_root.mkdir()
        self.assertIn("empty dataset", scan_dataset(empty_root)["error"])

        classless_root = self.root / "classless"
        classless_root.mkdir()
        (classless_root / "readme.txt").write_text("x", encoding="utf-8")
        classless = scan_dataset(classless_root)
        self.assertIn("no classes", classless["error"])
        self.assertEqual(classless["ignored_files"], ["readme.txt"])

    def test_empty_class_is_invalid(self):
        self.touch(self.root / "valid-class" / "image.png")
        (self.root / "empty-class").mkdir()
        result = scan_dataset(self.root)
        self.assertFalse(result["valid"])
        self.assertEqual(result["error"], "empty classes: empty-class")

    def test_ignored_root_and_class_files_are_reported_but_do_not_hide_images(self):
        self.touch(self.root / "README.txt")
        self.touch(self.root / "pikachu" / "image.jpg")
        self.touch(self.root / "pikachu" / ".DS_Store")

        result = scan_dataset(self.root)

        self.assertTrue(result["valid"])
        self.assertEqual(result["ignored_files"], ["README.txt"])
        self.assertEqual(result["classes"][0]["ignored_files"], [".DS_Store"])
        self.assertEqual(result["total_ignored"], 2)

    def test_json_summary_is_stable_and_sorted(self):
        self.touch(self.root / "z-class" / "z.png")
        self.touch(self.root / "a-class" / "a.png")
        first = self.run_cli("--json", self.root)
        second = self.run_cli("--json", self.root)

        self.assertEqual(first.returncode, 0)
        self.assertEqual(first.stdout, second.stdout)
        data = json.loads(first.stdout)
        self.assertTrue(data["valid"])
        self.assertEqual([c["name"] for c in data["classes"]], ["a-class", "z-class"])

    def test_text_cli_reports_valid_and_invalid_exit_codes(self):
        self.touch(self.root / "class" / "image.png")
        valid = self.run_cli(self.root)
        self.assertEqual(valid.returncode, 0)
        self.assertIn("Status: VALID", valid.stdout)
        self.assertIn("Total images: 1", valid.stdout)

        invalid_root = self.root / "invalid"
        invalid_root.mkdir()
        invalid = self.run_cli(invalid_root)
        self.assertEqual(invalid.returncode, 1)
        self.assertIn("Status: INVALID", invalid.stdout)
        self.assertIn("empty dataset", invalid.stdout)

    def test_json_cli_reports_invalid_dataset_and_exit_code(self):
        invalid_root = self.root / "invalid"
        invalid_root.mkdir()
        result = self.run_cli(invalid_root, "--json")
        self.assertEqual(result.returncode, 1)
        data = json.loads(result.stdout)
        self.assertFalse(data["valid"])
        self.assertIn("empty dataset", data["error"])


if __name__ == "__main__":
    unittest.main()
