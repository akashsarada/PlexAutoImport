import datetime
import os
import tempfile
import unittest
from pathlib import Path

from helpers.dates import date_from_filename, file_date, parse_exif_date


class ParseExifDateTest(unittest.TestCase):
    def test_parses_exiftool_datetime(self) -> None:
        self.assertEqual(
            parse_exif_date("2019:06:14 12:34:56"),
            datetime.date(2019, 6, 14),
        )

    def test_parses_dash_separated_datetime(self) -> None:
        self.assertEqual(
            parse_exif_date("2019-06-14T12:34:56"),
            datetime.date(2019, 6, 14),
        )

    def test_rejects_non_string_and_malformed_values(self) -> None:
        self.assertIsNone(parse_exif_date(None))
        self.assertIsNone(parse_exif_date(20190614))
        self.assertIsNone(parse_exif_date("not a date"))
        self.assertIsNone(parse_exif_date("2019:13:40 00:00:00"))


class DateFromFilenameTest(unittest.TestCase):
    def test_parses_date_prefix(self) -> None:
        self.assertEqual(
            date_from_filename("/photos/20190614_123456.jpg"),
            datetime.date(2019, 6, 14),
        )

    def test_rejects_missing_or_invalid_prefix(self) -> None:
        self.assertIsNone(date_from_filename("IMG_1234.jpg"))
        self.assertIsNone(date_from_filename("photo.jpg"))
        self.assertIsNone(date_from_filename("20191341_123456.jpg"))


class FileDateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _touch(self, name: str) -> Path:
        path = self.root / name
        path.touch()
        return path

    def test_exif_reader_takes_precedence(self) -> None:
        path = self._touch("20200101_000000.jpg")

        result = file_date(str(path), lambda _: datetime.date(2019, 6, 14))

        self.assertEqual(result, datetime.date(2019, 6, 14))

    def test_falls_back_to_filename_when_exif_missing(self) -> None:
        path = self._touch("20200101_000000.jpg")

        result = file_date(str(path), lambda _: None)

        self.assertEqual(result, datetime.date(2020, 1, 1))

    def test_falls_back_to_filename_when_exif_reader_raises(self) -> None:
        path = self._touch("20200101_000000.jpg")

        def broken_reader(_: str) -> datetime.date:
            raise RuntimeError("exiftool unavailable")

        with self.assertLogs("helpers.dates", level="ERROR"):
            result = file_date(str(path), broken_reader)

        self.assertEqual(result, datetime.date(2020, 1, 1))

    def test_falls_back_to_mtime_when_nothing_else_available(self) -> None:
        path = self._touch("IMG_1234.jpg")
        mtime = datetime.datetime(2018, 3, 5, 12, 0, 0).timestamp()
        os.utime(path, (mtime, mtime))

        result = file_date(str(path))

        self.assertEqual(result, datetime.date(2018, 3, 5))


if __name__ == "__main__":
    unittest.main()
