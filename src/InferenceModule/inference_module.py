import math
from typing import List, Tuple, Optional, Dict
from datetime import datetime

import numpy as np
from sklearn.cluster import DBSCAN

from MemoryModule.memory_module import MemoryModule


# Earth radius in meters
EARTH_RADIUS_M = 6371000.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the Haversine distance between two GPS points in meters.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


class Place:
    def __init__(self, name: str, center: Tuple[float, float], radius_m: float = 3.0):
        """
        A manually-defined zone.
        :param name:     Zone name (e.g., "bedside")
        :param center:   (lat, lon)
        :param radius_m: Radius in meters around center
        """
        self.name = name
        self.center = center
        self.radius_m = radius_m

    def contains(self, lat: float, lon: float) -> bool:
        """
        Check if a point (lat, lon) falls within this zone.
        """
        dist = haversine_distance(lat, lon, self.center[0], self.center[1])
        return dist <= self.radius_m


class ZoneModel:
    """
    Combines manual zones with an automatic clustering fallback.
    """
    def __init__(
        self,
        manual_zones: Optional[List[Place]] = None,
        cluster_radius_m: float = 3.0
    ):
        self.manual_zones = manual_zones or []
        self.cluster_radius_m = cluster_radius_m
        self._auto_centroids: Dict[str, Tuple[float, float]] = {}

    def fit(self, points: List[Tuple[float, float]]):
        """
        Cluster GPS points into automatic zones using DBSCAN with haversine.

        :param points: List of (lat, lon)
        """
        if not points:
            return
        # Convert to radians for haversine metric
        coords = np.radians(np.array(points))
        # eps in radians
        eps = self.cluster_radius_m / EARTH_RADIUS_M
        db = DBSCAN(eps=eps, min_samples=1, metric='haversine')
        labels = db.fit_predict(coords)

        # Compute centroids per cluster
        clusters: Dict[int, List[Tuple[float, float]]] = {}
        for lbl, (lat, lon) in zip(labels, points):
            clusters.setdefault(lbl, []).append((lat, lon))
        self._auto_centroids = {
            f"cluster_{lbl}": (
                float(np.mean([p[0] for p in pts])),
                float(np.mean([p[1] for p in pts]))
            )
            for lbl, pts in clusters.items()
        }

    def assign(self, lat: float, lon: float) -> str:
        """
        Assign a (lat, lon) to a manual zone or an automatic cluster.

        :return: zone name or cluster_id
        """
        # Check manual zones first
        for zone in self.manual_zones:
            if zone.contains(lat, lon):
                return zone.name
        # Fallback: find nearest auto-centroid within radius
        best_label = None
        best_dist = float('inf')
        for label, (clat, clon) in self._auto_centroids.items():
            dist = haversine_distance(lat, lon, clat, clon)
            if dist < best_dist:
                best_dist = dist
                best_label = label
        if best_label and best_dist <= self.cluster_radius_m:
            return best_label
        # If too far from any cluster, mark unknown
        return "unknown"


class TimeModel:
    """
    Stores P(being in zone | hour) probabilities and centroids.
    """
    def __init__(self):
        # hour -> zone -> probability
        self.probs: Dict[int, Dict[str, float]] = {}
        # zone -> centroid (lat, lon)
        self.centroids: Dict[str, Tuple[float, float]] = {}


class InferenceModule:
    def __init__(
        self,
        memory_module: MemoryModule,
        manual_zones: Optional[List[Place]] = None,
        cluster_radius_m: float = 3.0,
        freshness_seconds: int = 300
    ):
        self.mem = memory_module
        self.zoner = ZoneModel(manual_zones, cluster_radius_m)
        self.freshness = freshness_seconds
        self._time_models: Dict[str, TimeModel] = {}

    def define_zone(self, place: Place):
        """Add a manual zone."""
        self.zoner.manual_zones.append(place)

    def train_time_model(self, label: str) -> None:
        """
        Build clustering and P(being in zone | hour) from memory logs.
        """
        events = self.mem.get_sightings(label)
        points = [(ev['location'][0], ev['location'][1]) for ev in events]
        # Cluster points
        self.zoner.fit(points)
        # Prepare count structure: hour -> zone -> count
        counts: Dict[int, Dict[str, int]] = {h: {} for h in range(24)}
        for ev in events:
            hour = ev['timestamp'].hour
            zone = self.zoner.assign(ev['location'][0], ev['location'][1])
            counts[hour][zone] = counts[hour].get(zone, 0) + 1
            # store centroids
        model = TimeModel()
        model.centroids = {**{zone.name: zone.center for zone in self.zoner.manual_zones},
                           **self.zoner._auto_centroids}
        # Normalize to probabilities
        for hour, zone_counts in counts.items():
            total = sum(zone_counts.values())
            prob_dict: Dict[str, float] = {}
            if total > 0:
                for zone, cnt in zone_counts.items():
                    prob_dict[zone] = cnt / total
            model.probs[hour] = prob_dict
        self._time_models[label] = model

    def get_history(
        self,
        label: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> List[Dict]:
        """Return raw sighting events from memory."""
        events = self.mem.get_sightings(label, since)
        if until:
            events = [ev for ev in events if ev['timestamp'] <= until]
        return events

    def last_seen(self, label: str) -> Optional[Dict]:
        """Return the most recent sighting."""
        return self.mem.get_last_seen(label)

    def predict_location(
        self,
        label: str,
        at_time: Optional[datetime] = None
    ) -> Tuple[float, float]:
        """
        Return a (lat, lon) estimate for label at at_time.
        """
        now = at_time or datetime.now()
        last = self.last_seen(label)
        if last:
            age = (now - last['timestamp']).total_seconds()
            if age <= self.freshness:
                return last['location']
        # Otherwise, use time model
        if label not in self._time_models:
            self.train_time_model(label)
        model = self._time_models[label]
        hour = now.hour
        probs = model.probs.get(hour, {})
        if not probs:
            # fallback: use centroid of last seen or first centroid
            if last:
                return last['location']
            # any centroid
            return next(iter(model.centroids.values()))
        # pick zone with max probability
        zone = max(probs, key=probs.get)
        return model.centroids.get(zone, list(model.centroids.values())[0])

    def explain_prediction(
        self,
        label: str,
        at_time: Optional[datetime] = None
    ) -> str:
        """
        Generate a human-readable rationale for predict_location.
        """
        now = at_time or datetime.now()
        last = self.last_seen(label)
        if last and (now - last['timestamp']).total_seconds() <= self.freshness:
            return (f"Your {label} was seen {int((now - last['timestamp']).total_seconds())} seconds ago "
                    f"at location {last['location']}. Returning that location.")
        if label not in self._time_models:
            self.train_time_model(label)
        model = self._time_models[label]
        hour = now.hour
        probs = model.probs.get(hour, {})
        if not probs:
            return f"No data for hour {hour}. Unable to infer {label} location."
        zone = max(probs, key=probs.get)
        p = probs[zone] * 100
        centroid = model.centroids[zone]
        return (f"No recent sightings. Between hour {hour} and {hour+1}, your {label} was in '{zone}' "
                f"{p:.1f}% of the time (centroid at {centroid}).")
