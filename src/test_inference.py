# inference_module_demo.py

import os
from datetime import datetime, timedelta
from pathlib import Path

from MemoryModule.memory_module import MemoryModule
from InferenceModule.inference_module import InferenceModule, Place


def main():
    # Prepare demo directories
    script_dir = Path(__file__).parent
    logs_dir = script_dir / "memory_logs_inference_demo"

    # Initialize MemoryModule (fresh demo)
    mem = MemoryModule(storage_dir=str(logs_dir))

    # Define manual zones: bedside and desk
    bedside_center = (37.7749, -122.4194)
    desk_center    = (37.7755, -122.4185)
    zones = [
        Place("bedside", bedside_center, radius_m=5.0),
        Place("desk",    desk_center,    radius_m=5.0)
    ]

    # Initialize InferenceModule with manual zones
    inf = InferenceModule(mem, manual_zones=zones, cluster_radius_m=5.0, freshness_seconds=600)

    # Simulate sightings for 'test_object'
    # Two sightings by the bedside at 2:15 and 2:45 AM
    t1 = datetime(2025, 5, 6, 2, 15)
    t2 = datetime(2025, 5, 6, 2, 45)
    # One sighting at the desk at  9:30 AM
    t3 = datetime(2025, 5, 6, 9, 30)

    mem.store_sighting({
        "label": "test_object",
        "track_id": 1,
        "bbox": (0, 0, 0, 0),  # dummy
        "timestamp": t1,
        "location": bedside_center
    })
    mem.store_sighting({
        "label": "test_object",
        "track_id": 2,
        "bbox": (0, 0, 0, 0),
        "timestamp": t2,
        "location": bedside_center
    })
    mem.store_sighting({
        "label": "test_object",
        "track_id": 3,
        "bbox": (0, 0, 0, 0),
        "timestamp": t3,
        "location": desk_center
    })

    # Train the time-based model for 'test_object'
    inf.train_time_model("test_object")

    # Test predictions at different times
    test_times = [
        datetime(2025, 5, 6, 2, 30),  # between first two: bedside
        datetime(2025, 5, 6, 3, 0),   # shortly after: bedside
        datetime(2025, 5, 6, 9, 45),  # after desk sighting
        datetime(2025, 5, 6, 12, 0)   # midday: no data for this bin
    ]

    for tt in test_times:
        loc = inf.predict_location("test_object", at_time=tt)
        explanation = inf.explain_prediction("test_object", at_time=tt)
        print(f"At {tt.isoformat(' ')} → Predicted location: {loc}")
        print(f"Explanation: {explanation}\n")


if __name__ == "__main__":
    main()
