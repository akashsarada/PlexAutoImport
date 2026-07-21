import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from helpers.moving import move_file, robust_move


class MovingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @patch("moving.time.sleep")
    def test_move_file_copies_content_and_removes_source(self, _sleep) -> None:
        source = self.root / "source.jpg"
        destination = self.root / "destination.jpg"
        source.write_text("media", encoding="utf-8")

        moved = move_file(str(source), str(destination))

        self.assertTrue(moved)
        self.assertFalse(source.exists())
        self.assertEqual(destination.read_text(encoding="utf-8"), "media")

    def test_move_file_skips_existing_destination(self) -> None:
        source = self.root / "source.jpg"
        destination = self.root / "destination.jpg"
        source.write_text("source", encoding="utf-8")
        destination.write_text("destination", encoding="utf-8")

        with self.assertLogs("moving", level="WARNING") as logs:
            moved = move_file(str(source), str(destination))

        self.assertFalse(moved)
        self.assertIn("Destination file already exists", "\n".join(logs.output))
        self.assertTrue(source.exists())
        self.assertEqual(destination.read_text(encoding="utf-8"), "destination")

    @patch("moving.os.remove", side_effect=[PermissionError, None])
    @patch("moving.shutil.copy2")
    @patch("moving.time.sleep")
    def test_robust_move_retries_source_removal(
        self,
        sleep,
        _copy,
        remove,
    ) -> None:
        robust_move("source", "destination")

        self.assertEqual(remove.call_count, 2)
        self.assertEqual(sleep.call_args_list, [call(0.1), call(1)])


if __name__ == "__main__":
    unittest.main()
