import argparse
import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np


def parse_camera_source(camera: str):
    value = camera.strip()
    if value.isdigit():
        return int(value)
    return value

    
def parse_resolution(resolution: str):
    try:
        w_str, h_str = resolution.lower().split("x", maxsplit=1)
        width, height = int(w_str), int(h_str)
    except Exception as exc:
        raise ValueError(f"Invalid resolution format: {resolution}") from exc

    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution must be positive, got: {resolution}")
    return width, height


class Sensor:
    def get(self):
        raise NotImplementedError


class SensorX(Sensor):
    '''SensorX class'''
    def __init__(self, delay: float):
        self.delay = delay
        self.data = 0

    def get(self) -> int:
        time.sleep(self.delay)
        self.data += 1
        return self.data


class SensorCam(Sensor):
    def __init__(self, camera_name: str, resolution: str):
        width, height = parse_resolution(resolution)
        self.cap = cv2.VideoCapture(camera_name)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {camera_name}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ok, _ = self.cap.read()
        if not ok:
            self.cap.release()
            raise RuntimeError("Camera initialized but cannot read frames")

    def get(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def __del__(self):
        cap = getattr(self, "cap", None)
        if cap is not None:
            cap.release()


class WindowImage:
    def __init__(self, fps: float, name: str = "Sensor View"):
        if fps <= 0:
            raise ValueError("FPS must be > 0")
        self.fps = fps
        self.name = name
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)

    def show(self, img) -> bool:
        cv2.imshow(self.name, img)
        key = cv2.waitKey(max(1, int(1000.0 / self.fps))) & 0xFF
        return key != ord("q")

    def __del__(self):
        try:
            cv2.destroyWindow(self.name)
        except Exception:
            cv2.destroyAllWindows()


def setup_logging():
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "sensor_manager.log"),
            logging.StreamHandler(),
        ],
    )


def sensor_worker(sensor_name: str, sensor: Sensor, data_queue: queue.Queue, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            value = sensor.get()
            try:
                data_queue.put_nowait(value)
            except queue.Full:
                try:
                    data_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    data_queue.put_nowait(value)
                except queue.Full:
                    pass
        except Exception as exc:
            logging.exception("Sensor '%s' worker failed: %s", sensor_name, exc)
            stop_event.set()
            break


def compose_frame(frame, sensor_values: dict) -> np.ndarray:
    rendered = frame.copy()
    y = 40
    for key in ("sensor_100hz", "sensor_10hz", "sensor_1hz"):
        value = sensor_values.get(key)
        text = f"{key}: {value if value is not None else 'N/A'}"
        cv2.putText(
            rendered,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 35
    return rendered


def main():
    parser = argparse.ArgumentParser(description="Sensor manager with threaded sensor reading.")
    parser.add_argument("--camera", default="0", help="Camera index or video path, e.g. 0 for windows.")
    parser.add_argument("--resolution", default="1920x1080", help="Desired camera resolution")
    parser.add_argument("--fps", type=float, default=60.0, help="Display frequency (frames per second).")
    args = parser.parse_args()

    setup_logging()
    logging.info("Starting SensorManager")

    stop_event = threading.Event()
    camera_source = parse_camera_source(args.camera)

    try:
        sensor_cam = SensorCam(camera_source, args.resolution)
        window = WindowImage(args.fps)
    except Exception as exc:
        logging.exception("Initialization failed: %s", exc)
        return 1

    sensors = {
        "sensor_100hz": SensorX(delay=0.01),
        "sensor_10hz": SensorX(delay=0.1),
        "sensor_1hz": SensorX(delay=1.0),
    }
    queues = {name: queue.Queue(maxsize=1) for name in sensors}
    camera_queue = queue.Queue(maxsize=1)

    threads = [
        threading.Thread(
            target=sensor_worker,
            args=("camera", sensor_cam, camera_queue, stop_event),
            name="sensor-camera",
            daemon=True,
        )
    ]
    for name, sensor in sensors.items():
        threads.append(
            threading.Thread(
                target=sensor_worker,
                args=(name, sensor, queues[name], stop_event),
                name=f"sensor-{name}",
                daemon=True,
            )
        )

    for thread in threads:
        thread.start()

    last_frame = None
    sensor_values = {name: None for name in sensors}

    try:
        while not stop_event.is_set():
            try:
                while True:
                    last_frame = camera_queue.get_nowait()
            except queue.Empty:
                pass

            for name, q in queues.items():
                try:
                    while True:
                        sensor_values[name] = q.get_nowait()
                except queue.Empty:
                    pass

            if last_frame is None:
                time.sleep(0.005)
                continue

            output = compose_frame(last_frame, sensor_values)
            if not window.show(output):
                stop_event.set()
                break
    except KeyboardInterrupt:
        stop_event.set()
    except Exception as exc:
        logging.exception("Runtime failure: %s", exc)
        stop_event.set()

    for thread in threads:
        thread.join(timeout=1.0)

    del window
    del sensor_cam
    logging.info("SensorManager finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())