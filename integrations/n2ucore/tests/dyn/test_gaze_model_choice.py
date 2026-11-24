"""Tests for the model the GazeDirection node loads: the ViT-B GazeDINO on the GPU, its distilled ViT-S student
on the CPU.

  a. ``GazeDirection._modelFile``: device CPU gives the student, GPU the model; on macOS with coremltools the
     Core ML package of either when it is there, the int8 PyTorch weights otherwise;
  b. ``architectureOf``: ViT-B or ViT-S from the width of the head's first layer, anything else refused;
  c. ``validateArtifact``: an artifact whose declared architecture does not match its weights is refused;
  d. ``GazeDinoEngine._buildModel``: the trunk and the head width follow the architecture;
  e. the artifacts in N2uCore/res/models/gazedino: model.joblib is the ViT-B, student.joblib the ViT-S.

Built without model, camera or event loop; (e) reads the real artifacts and is skipped when they are missing.

Run from project root:
    python3 -m unittest tests.dyn.test_gaze_model_choice -v
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
from torch import nn

from N2uCore.dyn.eyegaze import GazeDirection, GazeDirectionTicker
from N2uCore.inference import gazeartifact
from N2uCore.inference.gazeartifact import ARCHITECTURE, ARCHITECTURE_SMALL, architectureOf, validateArtifact
from N2uCore.inference.gazedinoengine import GazeDinoEngine

MODELS = Path(__file__).resolve().parents[2] / "N2uCore" / "res" / "models" / "gazedino"


def _modelFile(student, platform, coremltools, existing):
    """GazeDirection._modelFile with the platform, coremltools and the files on disk set by the test."""
    manager = MagicMock()
    manager.absolutePath.side_effect = lambda relative: "/res/" + relative
    with patch("N2uCore.dyn.eyegaze.sys.platform", platform), \
            patch("N2uCore.dyn.eyegaze.importlib.util.find_spec", return_value=object() if coremltools else None), \
            patch("N2uCore.dyn.eyegaze.ResourceManager", return_value=manager), \
            patch("N2uCore.dyn.eyegaze.Path.exists", lambda path: str(path) in existing):
        return GazeDirection._modelFile(student)


def _artifact(width, architecture):
    """A minimal float32 artifact: the head's LayerNorm of the given width, and a declared architecture."""
    return {
        "format": gazeartifact.FORMAT,
        "architecture": dict(architecture),
        "preprocessing": dict(gazeartifact.PREPROCESSING),
        "output": dict(gazeartifact.OUTPUT),
        "state_dict": {"head.0.weight": np.ones(width, np.float32)},
        "metadata": {"source_sha256": "0" * 64, "validation_error_deg": None},
    }


class TestModelFile(unittest.TestCase):
    """(a) which file the node hands to the ticker."""

    ALL = {"/res/" + p for p in (GazeDirectionTicker.COREML_PATH, GazeDirectionTicker.STUDENT_COREML_PATH)}

    def test_gpu_on_macos_is_the_core_ml_model(self):
        self.assertEqual(_modelFile(False, "darwin", True, self.ALL), "/res/" + GazeDirectionTicker.COREML_PATH)

    def test_cpu_on_macos_is_the_core_ml_student(self):
        self.assertEqual(_modelFile(True, "darwin", True, self.ALL), "/res/" + GazeDirectionTicker.STUDENT_COREML_PATH)

    def test_without_coremltools_the_pytorch_weights(self):
        self.assertEqual(_modelFile(False, "darwin", False, self.ALL), "/res/" + GazeDirectionTicker.MODEL_PATH)
        self.assertEqual(_modelFile(True, "darwin", False, self.ALL), "/res/" + GazeDirectionTicker.STUDENT_PATH)

    def test_elsewhere_the_pytorch_weights(self):
        self.assertEqual(_modelFile(False, "linux", True, self.ALL), "/res/" + GazeDirectionTicker.MODEL_PATH)
        self.assertEqual(_modelFile(True, "win32", True, self.ALL), "/res/" + GazeDirectionTicker.STUDENT_PATH)

    def test_missing_core_ml_student_falls_back_to_its_weights(self):
        existing = {"/res/" + GazeDirectionTicker.COREML_PATH}
        self.assertEqual(_modelFile(True, "darwin", True, existing), "/res/" + GazeDirectionTicker.STUDENT_PATH)

    def test_device_prop_picks_the_student(self):
        node = GazeDirection.__new__(GazeDirection)
        node._memProvider, node._device = MagicMock(), "cpu"
        with patch.object(GazeDirection, "_modelFile", side_effect=lambda student: "student" if student else "model"):
            node._props = {"device": "CPU"}
            self.assertEqual(node._tickerArgs()["modelPath"], "student")
            node._props = {"device": "GPU"}
            self.assertEqual(node._tickerArgs()["modelPath"], "model")


class TestArchitecture(unittest.TestCase):
    """(b) and (c)."""

    def test_width_gives_the_architecture(self):
        self.assertEqual(architectureOf({"head.0.weight": np.ones(768, np.float32)}), ARCHITECTURE)
        self.assertEqual(architectureOf({"head.0.weight": np.ones(384, np.float32)}), ARCHITECTURE_SMALL)

    def test_other_widths_are_refused(self):
        self.assertIsNone(architectureOf({"head.0.weight": np.ones(1024, np.float32)}))
        self.assertIsNone(architectureOf({}))

    def test_matching_artifacts_are_valid(self):
        self.assertIsNotNone(validateArtifact(_artifact(768, ARCHITECTURE)))
        self.assertIsNotNone(validateArtifact(_artifact(384, ARCHITECTURE_SMALL)))

    def test_declared_architecture_must_match_the_weights(self):
        with patch.object(gazeartifact.log, "err"):
            self.assertIsNone(validateArtifact(_artifact(384, ARCHITECTURE)))
            self.assertIsNone(validateArtifact(_artifact(768, ARCHITECTURE_SMALL)))


class TestBuildModel(unittest.TestCase):
    """(d) the network the engine builds for the weights."""

    def test_trunk_and_head_follow_the_architecture(self):
        engine = GazeDinoEngine.__new__(GazeDinoEngine)
        for architecture in (ARCHITECTURE, ARCHITECTURE_SMALL):
            with patch("N2uCore.inference.gazedinoengine.torch.hub.load", return_value=nn.Identity()) as load:
                model = engine._buildModel(Path("hub"), architecture)
            self.assertEqual(load.call_args[0][1], architecture["backbone"])
            self.assertEqual(model.head[0].normalized_shape, (architecture["head_dims"][0],))
            self.assertEqual(model.head[1].in_features, architecture["head_dims"][0])


@unittest.skipUnless((MODELS / "model.joblib").exists() and (MODELS / "student.joblib").exists(),
                     "gaze artifacts not in N2uCore/res/models/gazedino")
class TestShippedArtifacts(unittest.TestCase):
    """(e) the files the node loads."""

    def test_model_is_the_vit_b_and_student_the_vit_s(self):
        for name, architecture in (("model.joblib", ARCHITECTURE), ("student.joblib", ARCHITECTURE_SMALL)):
            artifact = validateArtifact(joblib.load(MODELS / name))
            self.assertIsNotNone(artifact, name)
            self.assertEqual(artifact["architecture"], architecture, name)


if __name__ == "__main__":
    unittest.main()
