"""
trajectory_engine.py
=====================================================================
Multi-Camera ANPR Trajectory Reconstruction Engine
SIH 2026 - City-Wide AI Engine for Multi-Camera ANPR Trajectory Tracking

Provides:
  - Data access layer for ANPR detections + fixed camera locations
  - Fuzzy plate-number query interface (tolerant of OCR misreads)
  - Chronological trajectory reconstruction with deduplication
  - Inter-camera transition metrics (distance, time, speed)
  - Anomaly detection (teleportation, low confidence, large gaps)
  - JSON-serializable Trajectory output
  - Interactive Folium map + static Matplotlib fallback
  - Optional FastAPI query endpoint

Author: SIH 2026 Team
Python: 3.10+
Dependencies: pandas, folium, geopy, matplotlib (optional), fastapi+uvicorn (optional)
=====================================================================
"""

from __future__ import annotations

import difflib
import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd

try:
    from geopy.distance import geodesic
    _HAVE_GEOPY = True
except ImportError:
    _HAVE_GEOPY = False

try:
    import folium
    _HAVE_FOLIUM = True
except ImportError:
    _HAVE_FOLIUM = False


# =====================================================================
# 1. CONFIG
# =====================================================================

@dataclass
class EngineConfig:
    """Tunable thresholds for reconstruction & anomaly detection."""
    dedup_window_seconds: int = 15          # merge same-camera hits within this window
    plausible_max_speed_kmph: float = 140.0  # above this => teleportation flag
    min_speed_for_check_kmph: float = 0.0
    large_gap_minutes: int = 180             # unexplained gap flag threshold
    min_confidence: float = 0.55             # below this => low-confidence flag
    fuzzy_plate_max_edits: int = 2           # allowed char difference for plate match
    fuzzy_match_cutoff: float = 0.75         # difflib similarity ratio cutoff


# =====================================================================
# 2. CORE DATA STRUCTURES
# =====================================================================

@dataclass
class TrajectoryPoint:
    detection_id: Any
    camera_id: str
    camera_name: str
    latitude: float
    longitude: float
    timestamp: datetime
    confidence: float
    vehicle_type: Optional[str] = None
    anomaly_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class TransitionAnomaly:
    from_camera: str
    to_camera: str
    from_time: datetime
    to_time: datetime
    distance_km: float
    time_gap_minutes: float
    implied_speed_kmph: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["from_time"] = self.from_time.isoformat()
        d["to_time"] = self.to_time.isoformat()
        return d


@dataclass
class Trajectory:
    plate_number: str
    points: List[TrajectoryPoint] = field(default_factory=list)
    anomalies: List[TransitionAnomaly] = field(default_factory=list)
    total_distance_km: float = 0.0
    total_duration_minutes: float = 0.0
    average_speed_kmph: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plate_number": self.plate_number,
            "points": [p.to_dict() for p in self.points],
            "anomalies": [a.to_dict() for a in self.anomalies],
            "total_distance_km": round(self.total_distance_km, 3),
            "total_duration_minutes": round(self.total_duration_minutes, 2),
            "average_speed_kmph": round(self.average_speed_kmph, 2),
            "num_points": len(self.points),
            "num_anomalies": len(self.anomalies),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# =====================================================================
# 3. DISTANCE UTILITY
# =====================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km. Falls back to pure-Python haversine
    if geopy is not installed (so the module never hard-fails)."""
    if _HAVE_GEOPY:
        return geodesic((lat1, lon1), (lat2, lon2)).km

    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# =====================================================================
# 4. DATA ACCESS LAYER
# =====================================================================

class ANPRDataStore:
    """
    Wraps the detections + cameras tables. Designed so the pandas
    in-memory implementation can later be swapped for a PostGIS /
    SQL-backed one without changing the engine's calling code -
    just reimplement these methods against a real DB.
    """

    def __init__(self, detections: pd.DataFrame, cameras: pd.DataFrame):
        self.detections = detections.copy()
        self.detections["timestamp"] = pd.to_datetime(self.detections["timestamp"])
        self.cameras = cameras.copy().set_index("camera_id", drop=False)

    @classmethod
    def from_csv(cls, detections_csv: str, cameras_csv: str) -> "ANPRDataStore":
        return cls(pd.read_csv(detections_csv), pd.read_csv(cameras_csv))

    def all_plate_numbers(self) -> List[str]:
        return sorted(self.detections["plate_number"].unique().tolist())

    def query_exact(
        self,
        plate_number: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        df = self.detections[self.detections["plate_number"] == plate_number]
        if start_time:
            df = df[df["timestamp"] >= start_time]
        if end_time:
            df = df[df["timestamp"] <= end_time]
        return df.sort_values("timestamp")

    def query_fuzzy(
        self,
        plate_number: str,
        max_edits: int = 2,
        cutoff: float = 0.75,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Tolerant of OCR misreads. Uses difflib.SequenceMatcher ratio as a
        cheap proxy for edit-distance similarity (avoids extra dependency
        on python-Levenshtein while still catching 1-2 char OCR errors).
        """
        candidates = self.all_plate_numbers()
        matches = []
        for cand in candidates:
            ratio = difflib.SequenceMatcher(None, plate_number.upper(), cand.upper()).ratio()
            edit_est = int(round((1 - ratio) * max(len(plate_number), len(cand))))
            if ratio >= cutoff or edit_est <= max_edits:
                matches.append(cand)

        if not matches:
            return self.detections.iloc[0:0]  # empty frame, same schema

        df = self.detections[self.detections["plate_number"].isin(matches)]
        if start_time:
            df = df[df["timestamp"] >= start_time]
        if end_time:
            df = df[df["timestamp"] <= end_time]
        return df.sort_values("timestamp")

    def query_by_camera(self, camera_id: str) -> pd.DataFrame:
        return self.detections[self.detections["camera_id"] == camera_id].sort_values("timestamp")

    def query_by_bbox(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> pd.DataFrame:
        cams_in_box = self.cameras[
            (self.cameras["latitude"] >= min_lat)
            & (self.cameras["latitude"] <= max_lat)
            & (self.cameras["longitude"] >= min_lon)
            & (self.cameras["longitude"] <= max_lon)
        ]
        return self.detections[self.detections["camera_id"].isin(cams_in_box["camera_id"])]

    def camera_info(self, camera_id: str) -> pd.Series:
        return self.cameras.loc[camera_id]


# =====================================================================
# 5. TRAJECTORY RECONSTRUCTION ENGINE
# =====================================================================

class TrajectoryReconstructionEngine:
    """
    Core query-based tracking interface. Given a plate number (exact or
    fuzzy) and an optional time window, reconstructs a chronologically
    ordered, deduplicated, anomaly-annotated Trajectory.
    """

    def __init__(self, store: ANPRDataStore, config: Optional[EngineConfig] = None):
        self.store = store
        self.config = config or EngineConfig()

    # -----------------------------------------------------------------
    # Public query interface
    # -----------------------------------------------------------------
    def get_trajectory(
        self,
        plate_number: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        fuzzy: bool = True,
    ) -> Trajectory:
        if fuzzy:
            raw = self.store.query_fuzzy(
                plate_number,
                max_edits=self.config.fuzzy_plate_max_edits,
                cutoff=self.config.fuzzy_match_cutoff,
                start_time=start_time,
                end_time=end_time,
            )
        else:
            raw = self.store.query_exact(plate_number, start_time, end_time)

        if raw.empty:
            return Trajectory(plate_number=plate_number)

        deduped = self._deduplicate(raw)
        points = self._attach_camera_metadata(deduped)
        points, anomalies = self._compute_transitions_and_anomalies(points)
        trajectory = self._assemble_trajectory(plate_number, points, anomalies)
        return trajectory

    def get_trajectories_for_bbox(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> Dict[str, Trajectory]:
        """Reconstruct trajectories for every plate seen inside a bounding box."""
        df = self.store.query_by_bbox(min_lat, max_lat, min_lon, max_lon)
        result = {}
        for plate in sorted(df["plate_number"].unique()):
            result[plate] = self.get_trajectory(plate, fuzzy=False)
        return result

    # -----------------------------------------------------------------
    # Internal steps
    # -----------------------------------------------------------------
    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse near-simultaneous re-triggers at the same camera."""
        if df.empty:
            return df
        df = df.sort_values(["camera_id", "timestamp"]).copy()
        keep_mask = []
        last_seen: Dict[str, pd.Timestamp] = {}
        window = timedelta(seconds=self.config.dedup_window_seconds)

        for _, row in df.iterrows():
            cam = row["camera_id"]
            ts = row["timestamp"]
            prev = last_seen.get(cam)
            if prev is not None and (ts - prev) <= window:
                keep_mask.append(False)  # duplicate trigger, skip
            else:
                keep_mask.append(True)
                last_seen[cam] = ts

        df = df[keep_mask]
        return df.sort_values("timestamp")

    def _attach_camera_metadata(self, df: pd.DataFrame) -> List[TrajectoryPoint]:
        points: List[TrajectoryPoint] = []
        for _, row in df.iterrows():
            try:
                cam = self.store.camera_info(row["camera_id"])
            except KeyError:
                continue  # skip detections whose camera isn't registered
            points.append(
                TrajectoryPoint(
                    detection_id=row.get("detection_id"),
                    camera_id=row["camera_id"],
                    camera_name=cam.get("camera_name", row["camera_id"]),
                    latitude=float(cam["latitude"]),
                    longitude=float(cam["longitude"]),
                    timestamp=row["timestamp"],
                    confidence=float(row.get("confidence_score", 1.0)),
                    vehicle_type=row.get("vehicle_type"),
                )
            )
        return sorted(points, key=lambda p: p.timestamp)

    def _compute_transitions_and_anomalies(
        self, points: List[TrajectoryPoint]
    ) -> Tuple[List[TrajectoryPoint], List[TransitionAnomaly]]:
        anomalies: List[TrajectoryPoint] = []
        cfg = self.config

        for i, p in enumerate(points):
            if p.confidence < cfg.min_confidence:
                p.anomaly_flags.append("low_confidence")

        transition_anomalies: List[TransitionAnomaly] = []
        for i in range(1, len(points)):
            prev, curr = points[i - 1], points[i]
            dist_km = haversine_km(prev.latitude, prev.longitude, curr.latitude, curr.longitude)
            gap_seconds = (curr.timestamp - prev.timestamp).total_seconds()
            gap_minutes = gap_seconds / 60.0
            speed_kmph = (dist_km / (gap_seconds / 3600.0)) if gap_seconds > 0 else float("inf")

            reasons = []
            if speed_kmph > cfg.plausible_max_speed_kmph:
                reasons.append("implausible_speed_teleportation")
            if gap_minutes > cfg.large_gap_minutes:
                reasons.append("large_unexplained_time_gap")

            if reasons:
                reason_str = ", ".join(reasons)
                curr.anomaly_flags.append(reason_str)
                transition_anomalies.append(
                    TransitionAnomaly(
                        from_camera=prev.camera_id,
                        to_camera=curr.camera_id,
                        from_time=prev.timestamp,
                        to_time=curr.timestamp,
                        distance_km=round(dist_km, 3),
                        time_gap_minutes=round(gap_minutes, 2),
                        implied_speed_kmph=round(speed_kmph, 2) if speed_kmph != float("inf") else -1,
                        reason=reason_str,
                    )
                )

        return points, transition_anomalies

    def _assemble_trajectory(
        self, plate_number: str, points: List[TrajectoryPoint], anomalies: List[TransitionAnomaly]
    ) -> Trajectory:
        total_distance = 0.0
        for i in range(1, len(points)):
            total_distance += haversine_km(
                points[i - 1].latitude, points[i - 1].longitude,
                points[i].latitude, points[i].longitude,
            )

        if len(points) >= 2:
            duration_minutes = (points[-1].timestamp - points[0].timestamp).total_seconds() / 60.0
        else:
            duration_minutes = 0.0

        avg_speed = (total_distance / (duration_minutes / 60.0)) if duration_minutes > 0 else 0.0

        return Trajectory(
            plate_number=plate_number,
            points=points,
            anomalies=anomalies,
            total_distance_km=total_distance,
            total_duration_minutes=duration_minutes,
            average_speed_kmph=avg_speed,
        )


# =====================================================================
# 6. VISUALIZATION
# =====================================================================

class TrajectoryVisualizer:
    """Renders a Trajectory as an interactive Folium map or a static plot."""

    @staticmethod
    def to_folium_map(trajectory: Trajectory, zoom_start: int = 12) -> "folium.Map":
        if not _HAVE_FOLIUM:
            raise ImportError("folium is not installed. Run: pip install folium")
        if not trajectory.points:
            raise ValueError("Trajectory has no points to plot.")

        center_lat = sum(p.latitude for p in trajectory.points) / len(trajectory.points)
        center_lon = sum(p.longitude for p in trajectory.points) / len(trajectory.points)
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="cartodbpositron")

        coords = [(p.latitude, p.longitude) for p in trajectory.points]
        folium.PolyLine(coords, color="#2b6cb0", weight=4, opacity=0.8).add_to(fmap)

        for idx, p in enumerate(trajectory.points, start=1):
            is_anomalous = len(p.anomaly_flags) > 0
            color = "red" if is_anomalous else "green" if idx == 1 else (
                "blue" if idx == len(trajectory.points) else "orange"
            )
            popup_html = (
                f"<b>Stop #{idx}</b><br>"
                f"Camera: {p.camera_name} ({p.camera_id})<br>"
                f"Time: {p.timestamp.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                f"Confidence: {p.confidence:.2f}<br>"
                f"{'<b style=color:red>ANOMALY: ' + '; '.join(p.anomaly_flags) + '</b>' if is_anomalous else ''}"
            )
            folium.Marker(
                location=[p.latitude, p.longitude],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"#{idx} {p.camera_name}",
                icon=folium.Icon(color=color, icon="camera", prefix="fa"),
            ).add_to(fmap)

        title_html = (
            f'<h4 style="position:fixed; top:10px; left:60px; z-index:9999; '
            f'background:white; padding:6px 10px; border-radius:4px;">'
            f'Trajectory: {trajectory.plate_number} | '
            f'{trajectory.total_distance_km:.1f} km | '
            f'{trajectory.total_duration_minutes:.0f} min | '
            f'{len(trajectory.anomalies)} anomalies</h4>'
        )
        fmap.get_root().html.add_child(folium.Element(title_html))
        return fmap

    @staticmethod
    def save_html(trajectory: Trajectory, path: str) -> str:
        fmap = TrajectoryVisualizer.to_folium_map(trajectory)
        fmap.save(path)
        return path

    @staticmethod
    def to_matplotlib(trajectory: Trajectory, save_path: Optional[str] = None):
        """Static fallback plot, useful for reports / non-interactive contexts."""
        import matplotlib.pyplot as plt

        if not trajectory.points:
            raise ValueError("Trajectory has no points to plot.")

        lats = [p.latitude for p in trajectory.points]
        lons = [p.longitude for p in trajectory.points]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(lons, lats, "-o", color="#2b6cb0", linewidth=2, markersize=6)

        for idx, p in enumerate(trajectory.points, start=1):
            color = "red" if p.anomaly_flags else "black"
            ax.annotate(
                f"{idx}. {p.camera_name}\n{p.timestamp.strftime('%H:%M:%S')}",
                (p.longitude, p.latitude),
                textcoords="offset points", xytext=(6, 6), fontsize=8, color=color,
            )

        ax.set_title(f"Trajectory: {trajectory.plate_number}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", alpha=0.4)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# =====================================================================
# 7. OPTIONAL FASTAPI QUERY ENDPOINT
# =====================================================================

def build_fastapi_app(engine: TrajectoryReconstructionEngine):
    """
    Optional REST wrapper: GET /trajectory/{plate_number}?start=&end=&fuzzy=
    Run with: uvicorn trajectory_engine:app --reload
    (only import fastapi/uvicorn if actually used, to keep core module lightweight)
    """
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="ANPR Trajectory Reconstruction API")

    @app.get("/trajectory/{plate_number}")
    def get_trajectory(
        plate_number: str,
        start: Optional[str] = Query(None, description="ISO 8601 start datetime"),
        end: Optional[str] = Query(None, description="ISO 8601 end datetime"),
        fuzzy: bool = Query(True),
    ):
        start_dt = datetime.fromisoformat(start) if start else None
        end_dt = datetime.fromisoformat(end) if end else None
        traj = engine.get_trajectory(plate_number, start_dt, end_dt, fuzzy=fuzzy)
        if not traj.points:
            raise HTTPException(status_code=404, detail="No detections found for this plate/time range.")
        return traj.to_dict()

    @app.get("/trajectory/{plate_number}/map", response_class=HTMLResponse)
    def get_trajectory_map(plate_number: str, fuzzy: bool = Query(True)):
        traj = engine.get_trajectory(plate_number, fuzzy=fuzzy)
        if not traj.points:
            raise HTTPException(status_code=404, detail="No detections found for this plate.")
        fmap = TrajectoryVisualizer.to_folium_map(traj)
        return fmap.get_root().render()

    return app


# =====================================================================
# 8. DEMO / MOCK DATA
# =====================================================================

def _build_mock_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """6 cameras across a mock city + 8 detections for one plate, including
    one deliberately implausible ('teleportation') jump for anomaly demo."""

    cameras = pd.DataFrame([
        {"camera_id": "CAM01", "camera_name": "City Center Junction", "latitude": 13.0827, "longitude": 80.2707},
        {"camera_id": "CAM02", "camera_name": "Ring Road North",      "latitude": 13.0950, "longitude": 80.2750},
        {"camera_id": "CAM03", "camera_name": "Highway Toll Gate",    "latitude": 13.1100, "longitude": 80.2900},
        {"camera_id": "CAM04", "camera_name": "IT Corridor Bridge",   "latitude": 13.0500, "longitude": 80.2300},
        {"camera_id": "CAM05", "camera_name": "Airport Approach",     "latitude": 12.9900, "longitude": 80.1700},
        {"camera_id": "CAM06", "camera_name": "Old Town Market",      "latitude": 13.0700, "longitude": 80.2600},
    ])

    base_time = datetime(2026, 8, 20, 8, 0, 0)
    plate = "TN09AB1234"

    detections = pd.DataFrame([
        {"detection_id": 1, "plate_number": plate, "camera_id": "CAM01",
         "timestamp": base_time,                              "confidence_score": 0.96, "vehicle_type": "car"},
        {"detection_id": 2, "plate_number": plate, "camera_id": "CAM01",  # duplicate trigger (dedup test)
         "timestamp": base_time + timedelta(seconds=4),        "confidence_score": 0.91, "vehicle_type": "car"},
        {"detection_id": 3, "plate_number": plate, "camera_id": "CAM06",
         "timestamp": base_time + timedelta(minutes=9),        "confidence_score": 0.88, "vehicle_type": "car"},
        {"detection_id": 4, "plate_number": plate, "camera_id": "CAM02",
         "timestamp": base_time + timedelta(minutes=20),       "confidence_score": 0.93, "vehicle_type": "car"},
        # anomalous jump: CAM02 -> CAM05 (far away) in just 3 minutes = implausible speed
        {"detection_id": 5, "plate_number": plate, "camera_id": "CAM05",
         "timestamp": base_time + timedelta(minutes=23),       "confidence_score": 0.80, "vehicle_type": "car"},
        {"detection_id": 6, "plate_number": plate, "camera_id": "CAM03",
         "timestamp": base_time + timedelta(minutes=45),       "confidence_score": 0.40, "vehicle_type": "car"},  # low confidence
        {"detection_id": 7, "plate_number": plate, "camera_id": "CAM04",
         "timestamp": base_time + timedelta(minutes=70),       "confidence_score": 0.97, "vehicle_type": "car"},
        {"detection_id": 8, "plate_number": plate, "camera_id": "CAM06",
         "timestamp": base_time + timedelta(hours=5, minutes=30), "confidence_score": 0.95, "vehicle_type": "car"},  # large gap
    ])

    return detections, cameras


def run_demo():
    print("=" * 70)
    print("ANPR Trajectory Reconstruction Engine — Demo")
    print("=" * 70)

    detections, cameras = _build_mock_data()
    store = ANPRDataStore(detections, cameras)
    engine = TrajectoryReconstructionEngine(store, EngineConfig())

    # Exact query
    trajectory = engine.get_trajectory("TN09AB1234", fuzzy=False)
    print(trajectory.to_json())

    # Fuzzy query demo (simulate an OCR misread: 0/O confusion, 1 char off)
    print("\n--- Fuzzy query for misread plate 'TN09AB1235' ---")
    fuzzy_trajectory = engine.get_trajectory("TN09AB1235", fuzzy=True)
    print(f"Matched {len(fuzzy_trajectory.points)} points via fuzzy match.")

    # Map export
    if _HAVE_FOLIUM:
        out_path = "trajectory_map.html"
        TrajectoryVisualizer.save_html(trajectory, out_path)
        print(f"\nInteractive map saved to: {out_path}")
    else:
        print("\n(folium not installed — skipping interactive map export)")

    # Static fallback plot
    try:
        fig = TrajectoryVisualizer.to_matplotlib(trajectory, save_path="trajectory_static.png")
        print("Static plot saved to: trajectory_static.png")
    except ImportError:
        print("(matplotlib not installed — skipping static plot)")


if __name__ == "__main__":
    run_demo()
