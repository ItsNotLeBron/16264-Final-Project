import cv2
import time
from multiprocessing import Process, Queue, Event, set_start_method

class Camera:
    """
    Robust camera stream producer using multiprocessing.
    Frames are pushed into a Queue. If full, the oldest frame is dropped.
    Supports local devices and network streams (HTTP/RTSP) with automatic retry.
    """

    def __init__(self, src=0, fps=1, queue_size=5):
        """
        :param src:       OpenCV capture source (0 for webcam, URL for IP cam/RTSP)
        :param fps:       Target frames per second
        :param queue_size: Max number of frames to buffer
        """
        self.src         = src
        self.fps         = fps
        self.queue_size  = queue_size

        self.frame_queue = Queue(maxsize=queue_size)
        self.stop_event  = Event()
        self.process     = None

    def start(self):
        """Spawn the camera process without upfront validation."""
        # Ensure safe start method on all platforms
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass

        # Spawn the capture loop
        self.process = Process(
            target=self._capture_loop,
            args=(self.frame_queue, self.stop_event, self.src, self.fps),
            daemon=True 
        )
        self.process.start()

    def _capture_loop(self, frame_queue, stop_event, src, fps):
        """
        Opens the source and reads frames. On failure it retries every second.
        """
        interval = 1.0 / fps
        cap = None

        try:
            while not stop_event.is_set():
                # (Re)open if needed
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        # couldn’t open yet—retry in a second
                        time.sleep(1.0)
                        continue

                # try to read a frame
                ret, frame = cap.read()
                if not ret or frame is None:
                    # connection may have dropped—reset and retry
                    cap.release()
                    cap = None
                    time.sleep(1.0)
                    continue

                # enqueue (dropping oldest if full)
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except:
                        pass
                frame_queue.put(frame)

                # precise FPS pacing with quick stop checks
                end_time = time.time() + interval
                while time.time() < end_time:
                    if stop_event.is_set():
                        break
                    time.sleep(0.01)

            if cap is not None:
                cap.release()
        except Exception as e:
            print(f"Error in capture loop: {e}")
            if cap is not None:
                cap.release()
        finally:
            if cap is not None:
                cap.release()

    def get_frame(self, timeout=None):
        """
        Retrieve the next available frame.
        :param timeout: Seconds to wait (None = block indefinitely)
        :return: cv2 image array, or None if queue empty / timed out
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except:
            return None

    def stop(self, join_timeout=2):
        """
        Signal shutdown and wait for the process to exit.
        If it doesn’t exit in `join_timeout` seconds, forcibly terminate.
        """
        self.stop_event.set()
        if self.process:
            self.process.join(join_timeout)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
            