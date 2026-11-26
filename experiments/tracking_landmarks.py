"""The landmarks of N2uCore's tracking node, for the scripts that run GazeDinoEngine without it.

Since N2uCore_dev a5c5e98 the engine takes the 478 FaceLandmarker points from the tracking node's
`Landmarks` channel instead of running MediaPipe itself. This class gives the same array: FaceLandmarker set
up as `AbstractTracking` (models/face_landmarker.task, image mode, one face, the VisibleTracking default
confidences 0.7 / 0.5, CPU), raw normalized (x, y, z) as float32, all zeros when there is no face.
"""

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

POINTS = 478


class TrackingLandmarks:
    MODEL = "N2uCore/res/models/face_landmarker.task"

    def __init__(self, n2ucore, detection=0.7, tracking=0.5, matrices=False):
        model = Path(n2ucore) / self.MODEL
        if not model.is_file():
            raise FileNotFoundError(model)
        self._mesher = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
                output_face_blendshapes=False,
                # the head pose matrix is an extra output of the same model: the landmarks do not change
                output_facial_transformation_matrixes=matrices,
                min_face_detection_confidence=detection,
                min_tracking_confidence=tracking,
                num_faces=1,
            )
        )

    def __call__(self, rgb):
        """(478, 3) float32 normalized landmarks of the RGB uint8 frame, zeros without a face."""
        return self.detect(rgb)[0]

    def detect(self, rgb):
        """The landmarks as __call__, and the 4x4 facial transformation matrix (head pose) when built with
        matrices=True, else None."""
        result = self._mesher.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if not result.face_landmarks:
            return np.zeros((POINTS, 3), np.float32), None
        mesh = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], np.float32)
        matrices = result.facial_transformation_matrixes
        return mesh, (np.asarray(matrices[0], np.float64) if matrices else None)

    def close(self):
        self._mesher.close()
