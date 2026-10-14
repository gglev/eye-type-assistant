from typing import Optional, Tuple
import cv2
import numpy as np
from core.interfaces import PupilPoints

# Отрисовывает зрачки на кадре
class FrameVisualizer:

    def __init__(
        self, color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 4, thickness: int = -1,
    ) -> None:
        
        self.color = color
        self.radius = radius
        self.thickness = thickness

    def draw_pupils(self, frame: np.ndarray, pupils: Optional [PupilPoints]) -> np.ndarray:
        if pupils is None or frame is None:
            return frame

        h, w = frame.shape[:2]

        # Переводим нормализованные координаты [0.0, 1.0] в пиксели кадра для отрисовки
        lx, ly = int(pupils.left[0] * w), int(pupils.left[1] * h)
        rx, ry = int(pupils.right[0] * w), int(pupils.right[1] * h)

        # Рисуем кружки в центрах зрачков
        cv2.circle(frame, (lx, ly), self.radius, self.color, self.thickness)
        cv2.circle(frame, (rx, ry), self.radius, self.color, self.thickness)

        return frame