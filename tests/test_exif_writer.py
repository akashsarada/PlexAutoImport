import unittest
from unittest.mock import MagicMock, call, patch

from aisorter.exif_writer import ExifToolKeywordWriter, write_keywords


class ExifToolKeywordWriterTest(unittest.TestCase):
    @patch("aisorter.exif_writer.ExifToolHelper")
    def test_reuses_one_helper_for_multiple_files(self, helper_class) -> None:
        context = MagicMock()
        helper = MagicMock()
        context.__enter__.return_value = helper
        helper_class.return_value = context

        with ExifToolKeywordWriter() as writer:
            writer("photo.jpg", ["Alice"])
            writer("clip.mp4", ["Family"])

        helper_class.assert_called_once_with()
        self.assertEqual(
            helper.set_tags.call_args_list,
            [
                call(
                    "photo.jpg",
                    tags={"EXIF:Keywords": ["Alice"], "XMP:Subject": ["Alice"]},
                    params=["-overwrite_original", "-P"],
                ),
                call(
                    "clip.mp4",
                    tags={
                        "XMP:Subject": ["Family"],
                        "Keys:Keywords": ["Family"],
                        "ItemList:Keyword": ["Family"],
                    },
                    params=["-overwrite_original", "-P"],
                ),
            ],
        )
        context.__exit__.assert_called_once_with(None, None, None)

    @patch("aisorter.exif_writer.ExifToolHelper")
    def test_empty_keywords_do_not_start_helper(self, helper_class) -> None:
        with ExifToolKeywordWriter() as writer:
            writer("photo.jpg", [])

        helper_class.assert_not_called()

    @patch("aisorter.exif_writer.ExifToolHelper")
    def test_failure_propagates_and_restarts_helper_for_next_file(self, helper_class) -> None:
        first_context = MagicMock()
        first_helper = MagicMock()
        first_helper.set_tags.side_effect = RuntimeError("process failed")
        first_context.__enter__.return_value = first_helper

        second_context = MagicMock()
        second_helper = MagicMock()
        second_context.__enter__.return_value = second_helper
        helper_class.side_effect = [first_context, second_context]

        with self.assertLogs("aisorter.exif_writer", level="WARNING"):
            with ExifToolKeywordWriter() as writer:
                with self.assertRaisesRegex(RuntimeError, "process failed"):
                    writer("first.jpg", ["Alice"])
                writer("second.jpg", ["Bob"])

        self.assertEqual(helper_class.call_count, 2)
        second_helper.set_tags.assert_called_once()
        first_context.__exit__.assert_called_once_with(None, None, None)
        second_context.__exit__.assert_called_once_with(None, None, None)

    @patch("aisorter.exif_writer.ExifToolHelper")
    def test_start_failure_closes_partial_context(self, helper_class) -> None:
        context = MagicMock()
        context.__enter__.side_effect = RuntimeError("start failed")
        helper_class.return_value = context

        with self.assertLogs("aisorter.exif_writer", level="WARNING"):
            with ExifToolKeywordWriter() as writer:
                with self.assertRaisesRegex(RuntimeError, "start failed"):
                    writer("photo.jpg", ["Alice"])

        context.__exit__.assert_called_once_with(None, None, None)

    @patch("aisorter.exif_writer.ExifToolHelper")
    def test_write_keywords_preserves_standalone_api(self, helper_class) -> None:
        context = MagicMock()
        helper = MagicMock()
        context.__enter__.return_value = helper
        helper_class.return_value = context

        write_keywords("photo.jpg", ["Alice"])

        helper.set_tags.assert_called_once()
        context.__exit__.assert_called_once_with(None, None, None)


if __name__ == "__main__":
    unittest.main()
