import threading
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from core.interfaces import ICameraStream

class OpenCVCameraStream(ICameraStream):

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.capture: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.running = False

        # Storage of newest frame
        self.frame: Optional[np.ndarray] = None
        self.ret: bool = False

        #Link to background thread
        self.thread: Optional[threading.Thread] = None

    def start(self) -> "OpenCVCameraStream":
        with self.lock:
            if not self.running:
                self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
                if not self.capture.isOpened():
                # if 0 is busy try connect 1
                    fallback = 1 if self.camera_index == 0 else 0
                    self.capture = cv2.VideoCapture(fallback, cv2.CAP_AVFOUNDATION)
                    if self.capture.isOpened():
                        self.camera_index = fallback
                # if 0 and 1 are busy raise error
                if not self.capture.isOpened():
                    raise RuntimeError(f"Failed to open camera with index {self.camera_index}")

                # warm up frame capture and set is active (running) flag
                self.ret, self.frame = self.capture.read()
                self.running = True

                # run a daemon thread 
                self.thread = threading.Thread(target=self._capture_worker,name = "CameraWorker", daemon=True)
                self.thread.start()
        return self

    def _capture_worker(self) -> None:
        # while camera is working
        while self.running:
            if self.capture is None:
                break   

            ret, frame = self.capture.read()

            if not ret:
                continue
            # doing mirrored for better assistance
            mirrored = cv2.flip(frame, 1)

            with self.lock:
                self.ret = ret
                self.frame = mirrored

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.running or not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self) -> None:
        with self.lock:
            self.running = False

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        # close the cam
        with self.lock:
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            self.frame = None
            self.ret = False