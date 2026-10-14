import os
from typing import Optional
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from core.interfaces import IEyeTracker, PupilPoints

class MediaPipeEyeTracker(IEyeTracker):
    # Индексы центров зрачков: 468 - левый, 473 - правый
    LEFT_PUPIL_INDEX = 468
    RIGHT_PUPIL_INDEX = 473

    def __init__(self, model_path: str = "face_landmarker.task") -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Файл модели не найден по пути: {model_path}. "
                "Скачай его командой: curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray) -> Optional[PupilPoints]:
        if frame is None:
            return None

        # MediaPipe Tasks ожидает RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self._detector.detect(mp_image)

        if not detection_result.face_landmarks:
            return None

        landmarks = detection_result.face_landmarks[0]
        if len(landmarks) <= self.RIGHT_PUPIL_INDEX:
            return None

        left_pupil = landmarks[self.LEFT_PUPIL_INDEX]
        right_pupil = landmarks[self.RIGHT_PUPIL_INDEX]

        return PupilPoints(
            left=(float(left_pupil.x), float(left_pupil.y)),
            right=(float(right_pupil.x), float(right_pupil.y)),
        )

    def close(self) -> None:
        self._detector.close()