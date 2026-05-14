"""
Task 5: YOLOv8s-pose on CPU — single vs multi-threaded frame pipeline.

Pipeline (multi): capture/read → input queue → worker threads (each owns a YOLO
instance) → output queue → reorder by frame index → write video.

Bonus: webcam realtime with the same pool + ordered display buffer.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

TARGET_W = 640
TARGET_H = 480
MODEL_NAME = "yolov8s-pose.pt"


def parse_camera_source(camera: str) -> int | str:
    value = camera.strip()
    if value.isdigit():
        return int(value)
    return value


def setup_logging() -> None:
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "pose_task5.log"),
            logging.StreamHandler(),
        ],
    )


def resize_to_target(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if (w, h) == (TARGET_W, TARGET_H):
        return frame
    return cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


class VideoCaptureRaii:

    def __init__(self, source: int | str | Path):
        self._cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"Cannot open video source: {source!r}")

    @property
    def cap(self) -> cv2.VideoCapture:
        return self._cap

    def read(self) -> tuple[bool, np.ndarray]:
        return self._cap.read()

    def get_prop(self, prop: int) -> float:
        return float(self._cap.get(prop))

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None  # type: ignore[assignment]

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class VideoWriterRaii:
    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if not self._writer.isOpened():
            self._writer.release()
            raise RuntimeError(f"Cannot open VideoWriter for {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _torch_threads_per_worker(num_workers: int) -> int:
    cpu = os.cpu_count() or 1
    return max(1, cpu // max(1, num_workers))


def _make_predict_fn_in_this_thread(num_workers: int) -> Callable[[np.ndarray], np.ndarray]:
    """Each calling thread gets its own YOLO instance (thread-safe pattern)."""
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(_torch_threads_per_worker(num_workers))

    model = YOLO(MODEL_NAME)

    def predict_bgr(frame_bgr: np.ndarray) -> np.ndarray:
        results = model.predict(frame_bgr, verbose=False, imgsz=TARGET_W, device="cpu")
        return results[0].plot()

    return predict_bgr


def worker_loop(
    in_q: queue.Queue[tuple[int, np.ndarray] | None],
    out_q: queue.Queue[tuple[int, np.ndarray]],
    stop_on_error: threading.Event,
    num_workers: int,
) -> None:
    predict = _make_predict_fn_in_this_thread(num_workers)
    while not stop_on_error.is_set():
        item = in_q.get()
        if item is None:
            break
        idx, frame = item
        try:
            plotted = predict(frame)
            out_q.put((idx, plotted))
        except Exception:
            logging.exception("Worker inference failed at frame %s", idx)
            stop_on_error.set()
            break


def producer_video(
    cap: VideoCaptureRaii,
    in_q: queue.Queue[tuple[int, np.ndarray] | None],
    num_workers: int,
    frame_count_out: list[int],
    maxsize: int = 64,
) -> None:
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_to_target(frame)
        in_q.put((idx, frame))
        idx += 1
    frame_count_out[0] = idx
    for _ in range(num_workers):
        in_q.put(None)


def process_video_multithreaded(
    video_path: Path,
    output_path: Path,
    num_workers: int,
    in_queue_maxsize: int = 64,
) -> tuple[float, int]:
    cap = VideoCaptureRaii(video_path)
    try:
        fps = cap.get_prop(cv2.CAP_PROP_FPS) or 25.0
    except Exception:
        fps = 25.0

    in_q: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(maxsize=in_queue_maxsize)
    out_q: queue.Queue[tuple[int, np.ndarray]] = queue.Queue()
    stop_err = threading.Event()
    frame_count: list[int] = [0]

    workers = [
        threading.Thread(
            target=worker_loop,
            args=(in_q, out_q, stop_err, num_workers),
            name=f"pose-worker-{i}",
            daemon=True,
        )
        for i in range(num_workers)
    ]
    for w in workers:
        w.start()

    t0 = time.perf_counter()
    prod = threading.Thread(
        target=producer_video,
        args=(cap, in_q, num_workers, frame_count, in_queue_maxsize),
        name="frame-producer",
        daemon=True,
    )
    prod.start()
    prod.join()
    cap.close()
    for w in workers:
        w.join()
    t1 = time.perf_counter()

    if stop_err.is_set():
        raise RuntimeError("Processing stopped due to worker error; see log.")

    total = frame_count[0]
    if total == 0:
        raise RuntimeError("No frames decoded from input video.")

    frames_by_idx: dict[int, np.ndarray] = {}
    for _ in range(total):
        idx, fr = out_q.get()
        frames_by_idx[idx] = fr

    writer = VideoWriterRaii(output_path, fps, (TARGET_W, TARGET_H))
    try:
        for i in range(total):
            writer.write(frames_by_idx[i])
    finally:
        writer.close()

    elapsed = t1 - t0
    return elapsed, total


def process_video_singlethreaded(video_path: Path, output_path: Path) -> tuple[float, int]:
    cap = VideoCaptureRaii(video_path)
    try:
        fps = cap.get_prop(cv2.CAP_PROP_FPS) or 25.0
    except Exception:
        fps = 25.0

    predict = _make_predict_fn_in_this_thread(1)
    plotted_list: list[np.ndarray] = []

    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_to_target(frame)
        plotted_list.append(predict(frame))
    t1 = time.perf_counter()
    cap.close()

    if not plotted_list:
        raise RuntimeError("No frames decoded from input video.")

    writer = VideoWriterRaii(output_path, fps, (TARGET_W, TARGET_H))
    try:
        for fr in plotted_list:
            writer.write(fr)
    finally:
        writer.close()

    return t1 - t0, len(plotted_list)


def benchmark_threads(video_path: Path, output_dir: Path, thread_list: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, float, int, float]] = []
    for n in thread_list:
        out = output_dir / f"bench_workers_{n}.mp4"
        elapsed, total = process_video_multithreaded(video_path, out, num_workers=n)
        per_frame_ms = (elapsed / total) * 1000.0 if total else 0.0
        rows.append((n, elapsed, total, per_frame_ms))
        logging.info("benchmark workers=%s elapsed=%.3fs frames=%s (%.2f ms/frame)", n, elapsed, total, per_frame_ms)

    best = min(rows, key=lambda r: r[1])
    print("\n=== benchmark (lower elapsed is better) ===")
    print("workers\telapsed_s\tframes\tms_per_frame")
    for n, elapsed, total, ms in rows:
        print(f"{n}\t{elapsed:.4f}\t{total}\t{ms:.2f}")
    print(f"\nBest worker count on this machine (by total time): {best[0]} ({best[1]:.4f} s)")


def camera_realtime(
    camera: str,
    num_workers: int,
    display_fps: float,
    record_path: Optional[Path],
) -> int:
    """
    Сapture thread fills input queue; workers annotate; display thread
    consumes frames in original index order (reorder buffer).
    """
    source = parse_camera_source(camera)
    cam = VideoCaptureRaii(source)
    cam.cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cam.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

    in_q: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(maxsize=32)
    out_q: queue.Queue[tuple[int, np.ndarray]] = queue.Queue()
    stop_event = threading.Event()
    stop_err = threading.Event()

    def capture_worker() -> None:
        idx = 0
        while not stop_event.is_set():
            try:
                ok, frame = cam.read()
                if not ok:
                    time.sleep(0.005)
                    continue
                frame = resize_to_target(frame)
                while not stop_event.is_set():
                    try:
                        in_q.put((idx, frame), timeout=0.05)
                        idx += 1
                        break
                    except queue.Full:
                        continue
            except Exception:
                logging.exception("Capture failed")
                stop_err.set()
                stop_event.set()
                break

    workers = [
        threading.Thread(
            target=worker_loop,
            args=(in_q, out_q, stop_err, num_workers),
            name=f"pose-worker-{i}",
            daemon=True,
        )
        for i in range(num_workers)
    ]
    for w in workers:
        w.start()

    cap_thread = threading.Thread(target=capture_worker, name="camera-capture", daemon=True)
    cap_thread.start()

    window_name = "YOLOv8s-pose (realtime, ordered)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    writer: VideoWriterRaii | None = None
    if record_path is not None:
        writer = VideoWriterRaii(record_path, display_fps, (TARGET_W, TARGET_H))

    next_show = 0
    pending: dict[int, np.ndarray] = {}
    last_shown: Optional[np.ndarray] = None
    t_stats0 = time.perf_counter()
    shown = 0

    try:
        while not stop_event.is_set() and not stop_err.is_set():
            try:
                while True:
                    idx, fr = out_q.get_nowait()
                    pending[idx] = fr
            except queue.Empty:
                pass

            if next_show in pending:
                img = pending.pop(next_show)
                next_show += 1
                last_shown = img
                if writer is not None:
                    writer.write(img)
                cv2.putText(
                    img,
                    f"workers={num_workers} ordered_idx={next_show - 1}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, img)
                shown += 1
            elif last_shown is not None:
                cv2.imshow(window_name, last_shown)

            key = cv2.waitKey(max(1, int(1000.0 / display_fps))) & 0xFF
            if key == ord("q"):
                stop_event.set()
                break

            if pending and (max(pending.keys()) - next_show) > 120:
                logging.warning(
                    "Reorder buffer depth=%s (next=%s); consider fewer workers or faster CPU",
                    max(pending.keys()) - next_show,
                    next_show,
                )

    finally:
        stop_event.set()
        cap_thread.join(timeout=3.0)
        for _ in workers:
            in_q.put(None)
        for w in workers:
            w.join(timeout=120.0)
        if writer is not None:
            writer.close()
        cam.close()
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            cv2.destroyAllWindows()

    dt = time.perf_counter() - t_stats0
    fps_achieved = shown / dt if dt > 0 else 0.0
    logging.info("Realtime session: shown=%s frames in %.3f s (~%.2f FPS)", shown, dt, fps_achieved)
    print(f"Displayed frames: {shown}, wall time: {dt:.3f} s, approximate FPS: {fps_achieved:.2f}")
    return 0 if not stop_err.is_set() else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLOv8s-pose CPU: single vs multi-threaded video/camera.")
    p.add_argument("--video", type=Path, help="Путь к входному видео (ожидается 640x480; иначе ресайз).")
    p.add_argument(
        "--mode",
        choices=("single", "multi"),
        help="Режим: один поток или пул потоков с очередями и восстановлением порядка.",
    )
    p.add_argument("--output", type=Path, help="Имя / путь выходного видео с keypoints.")
    p.add_argument("--threads", type=int, default=None, help="Число рабочих потоков (multi). По умолчанию: min(8, CPU).")
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Прогнать multi-режим для набора чисел потоков и вывести таблицу времени (ускорение подбирается по минимуму).",
    )
    p.add_argument(
        "--benchmark-threads",
        type=str,
        default="1,2,4,6,8,12,16",
        help="Список через запятую для --benchmark.",
    )
    p.add_argument(
        "--camera",
        action="store_true",
        help="Доп. задание: веб-камера в реальном времени (ordered display + пул потоков).",
    )
    p.add_argument("--camera-index", default="0", help="Индекс или строка источника камеры (как в SensorManager).")
    p.add_argument("--display-fps", type=float, default=60.0, help="Частота опроса окна (waitKey), >0.")
    p.add_argument(
        "--record",
        type=Path,
        default=None,
        help="При --camera: опционально писать упорядоченное видео в файл.",
    )
    return p


def main() -> int:
    setup_logging()
    args = build_arg_parser().parse_args()

    if args.camera:
        n = args.threads if args.threads is not None else min(8, os.cpu_count() or 4)
        if n < 1:
            raise SystemExit("--threads must be >= 1")
        logging.info("Starting realtime camera with %s workers", n)
        return camera_realtime(args.camera_index, n, args.display_fps, args.record)

    if args.video is None:
        raise SystemExit("Укажите --video или включите --camera.")

    if args.benchmark:
        cpus = os.cpu_count() or 4
        thread_list = sorted({int(x.strip()) for x in args.benchmark_threads.split(",") if x.strip()})
        thread_list = [t for t in thread_list if t >= 1]
        if not thread_list:
            thread_list = [1, 2, 4, min(8, cpus)]
        out_dir = args.output.parent if args.output else Path("benchmark_out")
        out_dir.mkdir(parents=True, exist_ok=True)
        benchmark_threads(args.video, out_dir, thread_list)
        return 0

    if args.mode is None or args.output is None:
        raise SystemExit("Для файла нужны --mode и --output (или используйте --benchmark).")

    cpus = os.cpu_count() or 4
    threads = args.threads if args.threads is not None else min(8, cpus)
    if threads < 1:
        raise SystemExit("--threads must be >= 1")

    if args.mode == "single":
        elapsed, total = process_video_singlethreaded(args.video, args.output)
    else:
        elapsed, total = process_video_multithreaded(args.video, args.output, num_workers=threads)

    ms = (elapsed / total) * 1000.0 if total else 0.0
    print(f"Время обработки всех кадров: {elapsed:.4f} s ({total} кадров, {ms:.2f} ms/кадр)")
    print(f"Выходное видео: {args.output.resolve()}")
    logging.info("Done mode=%s threads=%s elapsed=%.4fs frames=%s", args.mode, threads, elapsed, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
