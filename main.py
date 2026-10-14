# main.py
import time
import cv2
from core.camera import OpenCVCameraStream
from core.eye_tracker import MediaPipeEyeTracker


def main():
  print("Инициализация камеры и нейросети...")
  camera = OpenCVCameraStream(camera_index=0).start()
  tracker = MediaPipeEyeTracker()
  time.sleep(0.3)

  print("Трекинг запущен. Нажми 'q' для выхода.")
  try:
    while True:
      success, frame = camera.read()
      if not success or frame is None:
        continue

      # Получаем нормализованные зрачки от модели
      pupils = tracker.process_frame(frame)

      if pupils is not None:
        h, w, _ = frame.shape

        # Переводим нормализованные координаты [0.0, 1.0] в пиксели кадра для отрисовки
        lx, ly = int(pupils.left[0] * w), int(pupils.left[1] * h)
        rx, ry = int(pupils.right[0] * w), int(pupils.right[1] * h)

        # Рисуем зеленые кружки в центрах зрачков
        cv2.circle(frame, (lx, ly), 4, (0, 255, 0), -1)
        cv2.circle(frame, (rx, ry), 4, (0, 255, 0), -1)

      cv2.imshow("Eye Tracking Test", frame)

      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
  finally:
    print("Завершение работы...")
    camera.stop()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()