import datetime
import os
import tempfile
import unittest
from pathlib import Path

from events import group_events


class EventSorterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_groups_all_same_date_files_at_threshold(self) -> None:
        filenames = [
            "20260721_100001.jpg",
            "20260721_100002.jpg",
            "20260721_100003.jpg",
        ]
        for filename in filenames:
            (self.root / filename).touch()

        moved = group_events(str(self.root), threshold=3, move=self._move)

        event_folder = self.root / "Event on 20260721"
        self.assertEqual(moved, 3)
        self.assertEqual(sorted(path.name for path in event_folder.iterdir()), filenames)
        self.assertFalse(any((self.root / filename).exists() for filename in filenames))

    def test_leaves_date_group_below_threshold_untouched(self) -> None:
        filenames = ["20260721_100001.jpg", "20260721_100002.jpg"]
        for filename in filenames:
            (self.root / filename).touch()

        moved = group_events(str(self.root), threshold=3, move=self._move)

        self.assertEqual(moved, 0)
        self.assertFalse((self.root / "Event on 20260721").exists())
        self.assertTrue(all((self.root / filename).exists() for filename in filenames))

    def test_rejects_non_positive_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            group_events(str(self.root), threshold=0, move=self._move)

    def test_groups_files_without_date_prefix_by_mtime(self) -> None:
        filenames = ["IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"]
        mtime = datetime.datetime(2024, 5, 17, 12, 0, 0).timestamp()
        for filename in filenames:
            path = self.root / filename
            path.touch()
            os.utime(path, (mtime, mtime))

        moved = group_events(str(self.root), threshold=3, move=self._move)

        event_folder = self.root / "Event on 20240517"
        self.assertEqual(moved, 3)
        self.assertEqual(sorted(path.name for path in event_folder.iterdir()), filenames)

    def test_exif_reader_takes_precedence_over_filename_prefix(self) -> None:
        filenames = ["20260721_100001.jpg", "20260721_100002.jpg"]
        for filename in filenames:
            (self.root / filename).touch()

        moved = group_events(
            str(self.root),
            threshold=2,
            move=self._move,
            exif_reader=lambda _: datetime.date(2019, 6, 14),
        )

        event_folder = self.root / "Event on 20190614"
        self.assertEqual(moved, 2)
        self.assertEqual(sorted(path.name for path in event_folder.iterdir()), filenames)

    @staticmethod
    def _move(src: str, dest: str) -> bool:
        Path(src).rename(dest)
        return True


if __name__ == "__main__":
    unittest.main()
