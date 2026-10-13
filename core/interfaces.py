from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# хранит координаты зрачков в виде кортежей (x, y) для левого и правого глаза
@dataclass (frozen=True)
class PupilPoints: 
    left: Tuple[float, float]
    right: Tuple[float, float]


# точка на мониторе в реальных пикселях 
@dataclass (frozen=True)
class ScreenPoint:
    x: int
    y: int 

class ICameraStream(ABC):

    #start camera
    @abstractmethod
    def start(self) -> "ICameraStream":
        pass

    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        pass

    #stop camera
    @abstractmethod
    def stop(self) -> None:
        pass

class IGazeCalibrator(ABC):

    @abstractmethod
    def add_calibration_point(self,pupil_norm: Tuple[float, float], screen_pt: ScreenPoint) -> None:
        pass

    @abstractmethod
    def calibrate(self) -> bool:
        pass

    @abstractmethod
    def map_to_screen(self, pupil_norm: Tuple[float, float]) -> ScreenPoint:
        pass

    @property
    @abstractmethod
    def is_calibrated(self) -> bool:
        pass