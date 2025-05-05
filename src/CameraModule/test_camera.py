import cv2

from camera_module import Camera


if __name__ == "__main__":
    cam = Camera(src="http://172.26.98.218:8080/video", fps=10, queue_size=5)
    try:
        cam.start()
        while True:
            frame = cam.get_frame(timeout=1)
            if frame is None:
                continue
            cv2.imshow("Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted! Stopping camera...")

    finally:
        cam.stop()    
        cv2.destroyAllWindows()
