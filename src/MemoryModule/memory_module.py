import os
import threading
import csv
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import datetime

import cv2
import numpy as np


class MemoryModule:
    """
    MemoryModule stores and retrieves object sighting events.

    Each sighting is logged to a per-label text file in CSV format:
        datetime_iso,track_id,x,y,w,h,lat,lon
    An in-memory cache mirrors these events for fast querying.
    """

    def __init__(self, storage_dir: str = "memory_logs"):
        """
        :param storage_dir: Directory where per-object logs are stored.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        # label -> List of event dicts
        self._cache: Dict[str, List[Dict]] = {}

        # Load existing logs into cache
        self._load_existing()

    def _load_existing(self) -> None:
        for file in self.storage_dir.glob("*.txt"):
            label = file.stem
            events: List[Dict] = []
            with file.open("r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        dt = datetime.datetime.fromisoformat(row[0])
                        track_id = int(row[1])
                        x, y, w, h = map(int, row[2:6])
                        lat, lon = map(float, row[6:8])
                        events.append({
                            "label": label,
                            "timestamp": dt,
                            "track_id": track_id,
                            "bbox": (x, y, w, h),
                            "location": (lat, lon),
                        })
                    except Exception:
                        continue
            if events:
                self._cache[label] = events

    def store_sighting(self, sighting: Dict) -> None:
        """
        Store a single sighting event.

        :param sighting: Dict with keys:
                         - label: str
                         - track_id: int
                         - bbox: Tuple[x, y, w, h]
                         - timestamp: datetime.datetime
                         - location: Tuple[lat, lon]
        """
        label = sighting["label"]
        ts: datetime.datetime = sighting["timestamp"]
        track_id = sighting["track_id"]
        x, y, w, h = sighting["bbox"]
        lat, lon = sighting.get("location", (0.0, 0.0))

        file_path = self.storage_dir / f"{label}.txt"
        with self._lock:
            # Append to log file
            with file_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ts.isoformat(), track_id, x, y, w, h, lat, lon])

            # Update in-memory cache
            self._cache.setdefault(label, []).append({
                "label": label,
                "timestamp": ts,
                "track_id": track_id,
                "bbox": (x, y, w, h),
                "location": (lat, lon),
            })

    def get_last_seen(self, label: str) -> Optional[Dict]:
        """
        Get the most recent sighting for a given label.

        :param label: Object label
        :return: Sighting dict or None if no sightings
        """
        events = self._cache.get(label)
        if not events:
            return None
        return events[-1]

    def get_sightings(
        self,
        label: Optional[str] = None,
        since: Optional[datetime.datetime] = None
    ) -> List[Dict]:
        """
        Retrieve all sighting events, optionally filtered by label and datetime.

        :param label: Specific label to filter, or None for all labels
        :param since: Datetime threshold (inclusive)
        :return: List of sighting dicts
        """
        results: List[Dict] = []
        labels = [label] if label else list(self._cache.keys())
        for lbl in labels:
            for ev in self._cache.get(lbl, []):
                if since is None or ev["timestamp"] >= since:
                    results.append(ev.copy())
        return results

    def annotate_frame(
        self,
        frame: np.ndarray,
        label: str,
        event_index: int = -1
    ) -> np.ndarray:
        """
        Draw the bounding box of a specified sighting on a copy of the frame.

        :param frame: Original image
        :param label: Object label
        :param event_index: Index of event to annotate (default latest)
        :return: Annotated image copy
        """
        events = self._cache.get(label)
        if not events:
            return frame.copy()

        # Select event
        try:
            ev = events[event_index]
        except IndexError:
            ev = events[-1]

        x, y, w, h = ev["bbox"]
        lat, lon = ev["location"]
        out = frame.copy()
        # Draw bounding box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Prepare annotation text
        ts_str = ev['timestamp'].isoformat(sep=' ', timespec='seconds')
        text1 = f"{label}:{ev['track_id']} @ {ts_str}"
        text2 = f"Loc: ({lat:.5f}, {lon:.5f})"
        # Draw texts
        cv2.putText(out, text1, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(out, text2, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return out
