import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from aisorter.models.face_identifier import FaceIdentifier


class FaceClusteringTest(unittest.TestCase):
    @patch("aisorter.models.face_identifier.ort.InferenceSession")
    @patch("aisorter.models.face_identifier._download_if_missing")
    def setUp(self, mock_download, mock_session) -> None:
        self.identifier = FaceIdentifier(model_path="dummy.onnx")
        self.identifier._references = {}
        self.identifier._reference_dir = Path("/dummy/references")
        # Mock the embedding method to return predefined vectors
        self.identifier._embed = MagicMock()

    @patch("aisorter.models.face_identifier.cv2.imwrite", return_value=True)
    @patch("aisorter.models.face_identifier.Path.mkdir")
    def test_identify_no_match_saves_to_unknown(
        self,
        mock_mkdir,
        mock_imwrite,
    ) -> None:
        self.identifier._references = {
            "Alice": np.array([1.0] + [0.0] * 127, dtype=np.float32)
        }
        self.identifier._embed.return_value = np.array(
            [0.0, 1.0] + [0.0] * 126,
            dtype=np.float32,
        )
        face_crop = np.zeros((10, 10, 3), dtype=np.uint8)

        result = self.identifier.identify(
            face_crop,
            source_path="img1.jpg",
            face_index=0,
        )

        self.assertIsNone(result)
        mock_mkdir.assert_called_with(parents=True, exist_ok=True)
        expected_path = Path("/dummy/references/unknown/img1_face_0.jpg")
        mock_imwrite.assert_called_once()
        written_path = mock_imwrite.call_args.args[0]
        self.assertEqual(Path(written_path), expected_path)

    @patch("aisorter.models.face_identifier.cv2.imwrite", return_value=True)
    @patch("aisorter.models.face_identifier.Path.mkdir")
    @patch("aisorter.models.face_identifier.Path.exists", return_value=False)
    @patch("aisorter.models.face_identifier.Path.rename")
    def test_identify_matches_unknown_creates_unknown_x(
        self,
        mock_rename,
        _mock_exists,
        _mock_mkdir,
        mock_imwrite,
    ) -> None:
        query_embedding = np.array([0.0, 1.0] + [0.0] * 126, dtype=np.float32)
        matched_embedding = np.array([0.0, 0.99] + [0.0] * 126, dtype=np.float32)
        matched_path = Path("/dummy/references/unknown/old_img_face_1.jpg")
        self.identifier._references = {
            "Alice": np.array([1.0, 0.0] + [0.0] * 126, dtype=np.float32)
        }
        self.identifier._unknown_paths = [matched_path]
        self.identifier._unknown_matrix = matched_embedding.reshape(1, 128)
        self.identifier._embed.return_value = query_embedding
        face_crop = np.zeros((10, 10, 3), dtype=np.uint8)

        result = self.identifier.identify(
            face_crop,
            source_path="new_img.jpg",
            face_index=2,
        )

        self.assertEqual(result, "unknown_1")
        new_dir = Path("/dummy/references/unknown_1")
        mock_rename.assert_called_once_with(new_dir / matched_path.name)
        mock_imwrite.assert_called_once_with(
            str(new_dir / "new_img_face_2.jpg"),
            face_crop,
        )
        self.assertIn("unknown_1", self.identifier._references)


class UnknownEmbeddingCacheTest(unittest.TestCase):
    """Verify the in-memory unknown-face embedding cache."""

    @patch("aisorter.models.face_identifier.ort.InferenceSession")
    @patch("aisorter.models.face_identifier._download_if_missing")
    def setUp(self, mock_download, mock_session) -> None:
        self.identifier = FaceIdentifier(model_path="dummy.onnx")
        self.identifier._references = {}
        self.identifier._reference_dir = Path("/dummy/references")
        self.identifier._embed = MagicMock()

    @patch("aisorter.models.face_identifier.cv2.imread")
    @patch("aisorter.models.face_identifier.Path.is_dir", return_value=True)
    @patch("aisorter.models.face_identifier.Path.iterdir")
    def test_rebuild_unknown_cache_embeds_each_file_once(
        self, mock_iterdir, mock_is_dir, mock_imread
    ) -> None:
        img1 = Path("/dummy/references/unknown/face_a.jpg")
        img2 = Path("/dummy/references/unknown/face_b.jpg")
        mock_iterdir.return_value = [img1, img2]
        mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        emb1 = np.array([1.0, 0.0] + [0.0] * 126, dtype=np.float32)
        emb2 = np.array([0.0, 1.0] + [0.0] * 126, dtype=np.float32)
        self.identifier._embed.side_effect = [emb1, emb2]

        self.identifier._rebuild_unknown_cache(Path("/dummy/references/unknown"))

        self.assertEqual(self.identifier._embed.call_count, 2)
        self.assertEqual(self.identifier._unknown_matrix.shape, (2, 128))
        self.assertEqual(self.identifier._unknown_paths, [img1, img2])

    @patch("aisorter.models.face_identifier.cv2.imwrite", return_value=True)
    @patch("aisorter.models.face_identifier.Path.mkdir")
    @patch("aisorter.models.face_identifier.Path.is_dir", return_value=False)
    def test_save_to_unknown_appends_to_cache(
        self, mock_is_dir, mock_mkdir, mock_imwrite
    ) -> None:
        self.identifier._references = {"Alice": np.array([1.0, 0.0] + [0.0] * 126, dtype=np.float32)}
        query_emb = np.array([0.0, 1.0] + [0.0] * 126, dtype=np.float32)
        self.identifier._embed.return_value = query_emb

        self.identifier.identify(np.zeros((10, 10, 3), dtype=np.uint8), source_path="img.jpg", face_index=0)

        self.assertEqual(len(self.identifier._unknown_paths), 1)
        self.assertEqual(self.identifier._unknown_matrix.shape, (1, 128))

    @patch("aisorter.models.face_identifier.cv2.imwrite", return_value=True)
    @patch("aisorter.models.face_identifier.Path.mkdir")
    @patch("aisorter.models.face_identifier.Path.is_dir", return_value=False)
    def test_cache_used_on_second_query_no_extra_embed(
        self, _mock_is_dir, _mock_mkdir, _mock_imwrite
    ) -> None:
        self.identifier._references = {
            "Alice": np.array([1.0, 0.0] + [0.0] * 126, dtype=np.float32)
        }
        first_embedding = np.array([0.0, 1.0] + [0.0] * 126, dtype=np.float32)
        second_embedding = np.array([0.0, 0.0, 1.0] + [0.0] * 125, dtype=np.float32)
        self.identifier._embed.side_effect = [first_embedding, second_embedding]
        face_crop = np.zeros((10, 10, 3), dtype=np.uint8)

        self.identifier.identify(face_crop, source_path="img1.jpg", face_index=0)
        self.identifier._embed.reset_mock()
        self.identifier.identify(face_crop, source_path="img2.jpg", face_index=0)

        self.assertEqual(self.identifier._embed.call_count, 1)
        self.assertEqual(self.identifier._unknown_matrix.shape, (2, 128))

    @patch("aisorter.models.face_identifier.cv2.imwrite", return_value=True)
    @patch("aisorter.models.face_identifier.Path.mkdir")
    @patch("aisorter.models.face_identifier.Path.rename")
    @patch("aisorter.models.face_identifier.Path.exists", return_value=False)
    @patch("aisorter.models.face_identifier.Path.is_dir", return_value=False)
    def test_cluster_creation_removes_entry_from_cache(
        self, mock_is_dir, mock_exists, mock_rename, mock_mkdir, mock_imwrite
    ) -> None:
        self.identifier._references = {}
        existing_emb = np.array([0.5, 0.5] + [0.0] * 126, dtype=np.float32)
        existing_emb /= np.linalg.norm(existing_emb)
        existing_path = Path("/dummy/references/unknown/old_face.jpg")
        self.identifier._unknown_paths = [existing_path]
        self.identifier._unknown_matrix = existing_emb.reshape(1, 128)

        query_emb = existing_emb.copy()
        self.identifier._embed.return_value = query_emb

        result = self.identifier.identify(
            np.zeros((10, 10, 3), dtype=np.uint8), source_path="new.jpg", face_index=0
        )

        self.assertEqual(result, "unknown_1")
        self.assertEqual(len(self.identifier._unknown_paths), 0)
        self.assertEqual(self.identifier._unknown_matrix.shape[0], 0)
        self.assertIn("unknown_1", self.identifier._references)


    @patch("aisorter.models.face_identifier.cv2.imread")
    def test_completed_unknown_cluster_is_loaded_after_restart(self, mock_imread) -> None:
        mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        embedding = np.array([1.0] + [0.0] * 127, dtype=np.float32)
        self.identifier._embed.return_value = embedding

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_dir = Path(temp_dir)
            (reference_dir / "unknown").mkdir()
            cluster_dir = reference_dir / "unknown_1"
            cluster_dir.mkdir()
            (cluster_dir / "face.jpg").touch()

            self.identifier._load_dir(reference_dir, depth=0)

        self.assertIn("unknown_1", self.identifier._references)
        np.testing.assert_array_equal(self.identifier._references["unknown_1"], embedding)

    @patch("aisorter.models.face_identifier.cv2.imwrite", return_value=True)
    @patch("aisorter.models.face_identifier.Path.mkdir")
    @patch("aisorter.models.face_identifier.Path.exists", return_value=False)
    @patch("aisorter.models.face_identifier.Path.rename", side_effect=OSError("move failed"))
    def test_move_failure_keeps_existing_cache_entry(
        self,
        _mock_rename,
        _mock_exists,
        _mock_mkdir,
        _mock_imwrite,
    ) -> None:
        embedding = np.array([1.0] + [0.0] * 127, dtype=np.float32)
        matched_path = Path("/dummy/references/unknown/old_face.jpg")
        self.identifier._unknown_paths = [matched_path]
        self.identifier._unknown_matrix = embedding.reshape(1, 128)
        self.identifier._embed.return_value = embedding

        result = self.identifier.identify(
            np.zeros((10, 10, 3), dtype=np.uint8),
            source_path="new.jpg",
        )

        self.assertIsNone(result)
        self.assertIn(matched_path, self.identifier._unknown_paths)
        self.assertNotIn("unknown_1", self.identifier._references)
        self.assertEqual(self.identifier.new_people_found, 0)

    @patch("aisorter.models.face_identifier.cv2.imwrite", return_value=False)
    @patch("aisorter.models.face_identifier.Path.mkdir")
    @patch("aisorter.models.face_identifier.Path.exists", return_value=False)
    @patch("aisorter.models.face_identifier.Path.rename")
    def test_cluster_write_failure_rolls_back_cached_face(
        self,
        mock_rename,
        _mock_exists,
        _mock_mkdir,
        _mock_imwrite,
    ) -> None:
        embedding = np.array([1.0] + [0.0] * 127, dtype=np.float32)
        matched_path = Path("/dummy/references/unknown/old_face.jpg")
        self.identifier._unknown_paths = [matched_path]
        self.identifier._unknown_matrix = embedding.reshape(1, 128)
        self.identifier._embed.return_value = embedding

        result = self.identifier.identify(
            np.zeros((10, 10, 3), dtype=np.uint8),
            source_path="new.jpg",
        )

        self.assertIsNone(result)
        self.assertEqual(mock_rename.call_count, 2)
        self.assertEqual(self.identifier._unknown_paths, [matched_path])
        self.assertNotIn("unknown_1", self.identifier._references)
        self.assertEqual(self.identifier.new_people_found, 0)


if __name__ == "__main__":
    unittest.main()
