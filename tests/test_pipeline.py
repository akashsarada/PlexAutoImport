import unittest
from unittest.mock import Mock, patch

import numpy as np

from aisorter.pipeline import AISorterPipeline


class PipelineLoggingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = object.__new__(AISorterPipeline)
        self.pipeline._category_input_name = "input"
        self.pipeline._category_session = Mock()
        self.pipeline._face_detector = Mock()
        self.pipeline._face_identifier = Mock()
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.pipeline._load_image = Mock(return_value=(image, image))

    @patch("aisorter.pipeline.write_keywords")
    def test_logs_completion_after_all_executed_stages(self, write_keywords) -> None:
        self.pipeline._category_session.run.return_value = [
            np.array([[10.0, 10.0, -10.0]], dtype=np.float32)
        ]
        self.pipeline._face_detector.detect.return_value = [
            {"bbox": (0.0, 0.0, 4.0, 4.0), "confidence": 0.9}
        ]
        self.pipeline._face_identifier.identify.return_value = "Alice"

        with self.assertLogs("aisorter.pipeline", level="INFO") as logs:
            result = self.pipeline.process_image("photo.jpg")

        output = "\n".join(logs.output)
        self.assertIn("Stage 1/4 category classification complete", output)
        self.assertIn("Stage 2/4 face detection complete", output)
        self.assertIn("Stage 3/4 face identification complete", output)
        self.assertIn("Stage 4/4 keyword write complete", output)
        self.assertEqual(result["identities"], ["Alice"])
        write_keywords.assert_called_once_with("photo.jpg", ["Alice", "cars", "people"])

    @patch("aisorter.pipeline.write_keywords")
    def test_logs_skipped_conditional_stages(self, _write_keywords) -> None:
        self.pipeline._category_session.run.return_value = [
            np.array([[-10.0, -10.0, 10.0]], dtype=np.float32)
        ]

        with self.assertLogs("aisorter.pipeline", level="INFO") as logs:
            self.pipeline.process_image("scenery.jpg")

        output = "\n".join(logs.output)
        self.assertIn("Stage 1/4 category classification complete", output)
        self.assertIn("Stage 2/4 face detection skipped", output)
        self.assertIn("Stage 3/4 face identification skipped", output)
        self.assertIn("Stage 4/4 keyword write complete", output)
        self.pipeline._face_detector.detect.assert_not_called()
        self.pipeline._face_identifier.identify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
