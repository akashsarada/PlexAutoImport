import errno
import os
import shutil
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

    @patch("helpers.moving.shutil.copy2")
    def test_move_file_renames_on_same_filesystem(self, copy) -> None:
        source = self.root / "source.jpg"
        destination = self.root / "destination.jpg"
        source.write_text("media", encoding="utf-8")

        moved = move_file(str(source), str(destination))

        self.assertTrue(moved)
        self.assertFalse(source.exists())
        self.assertEqual(destination.read_text(encoding="utf-8"), "media")
        copy.assert_not_called()

    def test_move_file_copies_across_filesystems(self) -> None:
        source = self.root / "source.mp4"
        destination = self.root / "destination.mp4"
        source.write_text("video", encoding="utf-8")

        with patch(
            "helpers.moving.os.rename",
            side_effect=self._fail_first_rename_with_exdev(),
        ):
            moved = move_file(str(source), str(destination))

        self.assertTrue(moved)
        self.assertFalse(source.exists())
        self.assertEqual(destination.read_text(encoding="utf-8"), "video")
        self.assertFalse((self.root / "destination.mp4.importing").exists())

    def test_robust_move_copies_via_temp_name_then_renames(self) -> None:
        source = self.root / "source.mp4"
        destination = self.root / "destination.mp4"
        source.write_text("video", encoding="utf-8")
        copied_targets: list[str] = []
        real_copy2 = shutil.copy2

        def recording_copy2(src: str, dest: str) -> None:
            copied_targets.append(dest)
            real_copy2(src, dest)

        with patch(
            "helpers.moving.os.rename",
            side_effect=self._fail_first_rename_with_exdev(),
        ), patch("helpers.moving.shutil.copy2", side_effect=recording_copy2):
            mode = robust_move(str(source), str(destination))

        self.assertEqual(mode, "copy")
        self.assertEqual(copied_targets, [str(destination) + ".importing"])
        self.assertEqual(destination.read_text(encoding="utf-8"), "video")

    def test_robust_move_cleans_up_temp_file_when_copy_fails(self) -> None:
        source = self.root / "source.mp4"
        destination = self.root / "destination.mp4"
        source.write_text("video", encoding="utf-8")

        def failing_copy2(src: str, dest: str) -> None:
            Path(dest).write_text("partial", encoding="utf-8")
            raise OSError("disk full")

        with patch(
            "helpers.moving.os.rename",
            side_effect=self._fail_first_rename_with_exdev(),
        ), patch("helpers.moving.shutil.copy2", side_effect=failing_copy2):
            with self.assertRaisesRegex(OSError, "disk full"):
                robust_move(str(source), str(destination))

        self.assertTrue(source.exists())
        self.assertFalse(destination.exists())
        self.assertFalse((self.root / "destination.mp4.importing").exists())

    @staticmethod
    def _fail_first_rename_with_exdev():
        real_rename = os.rename
        failed = []

        def fake_rename(src: str, dest: str) -> None:
            if not failed:
                failed.append(True)
                raise OSError(errno.EXDEV, "cross-device link")
            real_rename(src, dest)

        return fake_rename

    def test_move_file_skips_existing_destination(self) -> None:
        source = self.root / "source.jpg"
        destination = self.root / "destination.jpg"
        source.write_text("source", encoding="utf-8")
        destination.write_text("destination", encoding="utf-8")

        with self.assertLogs("helpers.moving", level="WARNING") as logs:
            moved = move_file(str(source), str(destination))

        self.assertFalse(moved)
        self.assertIn("Destination file already exists", "\n".join(logs.output))
        self.assertTrue(source.exists())
        self.assertEqual(destination.read_text(encoding="utf-8"), "destination")

    @patch("helpers.moving.time.perf_counter", side_effect=[10.0, 10.125])
    @patch("helpers.moving.os.path.getsize", return_value=1234)
    @patch("helpers.moving.robust_move", return_value="rename")
    def test_move_file_logs_mode_size_and_elapsed_time(
        self,
        _move,
        _getsize,
        _perf_counter,
    ) -> None:
        source = self.root / "source.jpg"
        source.touch()

        with self.assertLogs("helpers.moving", level="INFO") as logs:
            moved = move_file(str(source), str(self.root / "destination.jpg"))

        self.assertTrue(moved)
        output = "\n".join(logs.output)
        self.assertIn("mode=rename", output)
        self.assertIn("bytes=1234", output)
        self.assertIn("elapsed_ms=125.0", output)

    @patch("helpers.moving.os.remove", side_effect=[PermissionError, None])
    @patch("helpers.moving.shutil.copy2")
    @patch("helpers.moving.time.sleep")
    def test_robust_move_retries_source_removal(
        self,
        sleep,
        _copy,
        remove,
    ) -> None:
        with patch(
            "helpers.moving.os.rename",
            side_effect=self._fail_first_rename_with_exdev_noop_after(),
        ):
            mode = robust_move("source", "destination")

        self.assertEqual(mode, "copy")
        self.assertEqual(remove.call_count, 2)
        self.assertEqual(sleep.call_args_list, [call(1)])

    @staticmethod
    def _fail_first_rename_with_exdev_noop_after():
        failed = []

        def fake_rename(src: str, dest: str) -> None:
            if not failed:
                failed.append(True)
                raise OSError(errno.EXDEV, "cross-device link")

        return fake_rename

    @patch("helpers.moving.shutil.copy2")
    @patch(
        "helpers.moving.os.rename",
        side_effect=PermissionError(errno.EACCES, "permission denied"),
    )
    def test_robust_move_propagates_non_cross_device_error(
        self,
        _rename,
        copy,
    ) -> None:
        with self.assertRaises(PermissionError):
            robust_move("source", "destination")

        copy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
