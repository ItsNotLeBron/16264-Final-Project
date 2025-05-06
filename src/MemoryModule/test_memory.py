# memory_module_demo.py

from pathlib import Path
import numpy as np
import cv2
import datetime
from memory_module import MemoryModule


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    logs_dir = script_dir / "memory_logs_demo"

    mem = MemoryModule(storage_dir=str(logs_dir))

    # Simulate three sightings of "test_object" with locations
    now = datetime.datetime.now()
    demo_sightings = [
        {
            "label": "test_object",
            "track_id": 1,
            "bbox": (10, 20, 50, 60),
            "timestamp": now,
            "location": (37.7749, -122.4194),  # example lat, lon
        },
        {
            "label": "test_object",
            "track_id": 1,
            "bbox": (15, 25, 50, 60),
            "timestamp": now + datetime.timedelta(seconds=1),
            "location": (37.7750, -122.4195),
        },
        {
            "label": "test_object",
            "track_id": 2,
            "bbox": (100, 40, 80, 90),
            "timestamp": now + datetime.timedelta(seconds=2),
            "location": (37.7751, -122.4196),
        },
    ]

    for s in demo_sightings:
        mem.store_sighting(s)

    # Query last-seen and full history
    last = mem.get_last_seen("test_object")
    print("Last seen:", last)

    all_events = mem.get_sightings("test_object")
    print("All sightings:")
    for ev in all_events:
        print(" ", ev)

    # Annotate an empty frame with the latest bounding box and location
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    annotated = mem.annotate_frame(frame, "test_object")
    output_path = script_dir / "annotated_demo.png"
    cv2.imwrite(str(output_path), annotated)
    print(f"Wrote {output_path} with the last bbox and location annotation")
