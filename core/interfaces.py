from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class PupilPoints:
    # Normalized coordinates for eyes
    left: Tuple[float, float]
    right: Tuple[float, float]


class ICameraStream(ABC):
    @abstractmethod
    def start(self) -> "ICameraStream":
        pass
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        pass
    @abstractmethod
    def stop(self) -> None:
        pass

class IEyeTracker(ABC):
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Optional[PupilPoints]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass