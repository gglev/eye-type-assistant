import time
import cv2
from core.camera import OpenCVCameraStream
from core.eye_tracker import MediaPipeEyeTracker
from core.visualizer import FrameVisualizer


def main():
  print("Инициализация камеры и нейросети")
  camera = OpenCVCameraStream(camera_index=0).start()
  tracker = MediaPipeEyeTracker()
  visualizer = FrameVisualizer()

  time.sleep(0.3)

  print("Трекинг запущен. Нажми 'q' для выхода.")
  try:
    while True:
      success, frame = camera.read()
      if not success or frame is None:
        continue

      # Получаем нормализованные зрачки от модели
      pupils = tracker.process_frame(frame)
      visualizer.draw_pupils(frame, pupils)

      cv2.imshow("Eye Type Assistant", frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      
  finally:
    print("Завершение работы...")
    camera.stop()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()