"""
bridge.py
=====================================================================
Connects the live ANPR camera script (events.json output) to the
Trajectory Reconstruction Engine.

Flow:
  multi_camera_output/events.json  --> pandas DataFrame  --> engine
=====================================================================
"""

import json
import pandas as pd

from trajectory_engine import (
    ANPRDataStore,
    TrajectoryReconstructionEngine,
    EngineConfig,
    TrajectoryVisualizer,
)

EVENTS_JSON_PATH = "multi_camera_output/events.json"

# The ANPR script only stores camera URLs, not GPS coordinates.
# Add real lat/lon here for each camera_id used in CAMERAS in the ANPR script.
CAMERA_LOCATIONS = {
    "CAM_01": {"camera_name": "Camera 1", "latitude": 13.0827, "longitude": 80.2707},
    "CAM_02": {"camera_name": "Camera 2", "latitude": 13.0950, "longitude": 80.2750},
    "CAM_03": {"camera_name": "Camera 3", "latitude": 13.1100, "longitude": 80.2900},
}


def load_events_as_detections(json_path: str) -> pd.DataFrame:
    """Reads events.json (from the ANPR script) and reshapes it into
    the detections DataFrame the trajectory engine expects."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = data["events"]
    rows = []
    for i, e in enumerate(events):
        rows.append({
            "detection_id": i,
            "plate_number": e["plate"],
            "camera_id": e["camera_id"],
            "timestamp": e["timestamp"],
            "confidence_score": e["ocr_confidence"],
        })
    return pd.DataFrame(rows)


def build_cameras_df() -> pd.DataFrame:
    """Turns CAMERA_LOCATIONS into the cameras DataFrame the engine expects."""
    rows = []
    for cam_id, info in CAMERA_LOCATIONS.items():
        rows.append({"camera_id": cam_id, **info})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 1. Load ANPR script's output
    detections = load_events_as_detections(EVENTS_JSON_PATH)
    cameras = build_cameras_df()

    if detections.empty:
        print("No events found in events.json yet. Run the ANPR script first.")
        exit()

    print(f"Loaded {len(detections)} detections across {detections['plate_number'].nunique()} plates.")

    # 2. Plug into the trajectory engine
    store = ANPRDataStore(detections, cameras)
    engine = TrajectoryReconstructionEngine(store, EngineConfig())

    # 3. Pick a plate to track (first one seen, for demo purposes)
    plate_number = detections["plate_number"].iloc[0]
    trajectory = engine.get_trajectory(plate_number, fuzzy=False)

    print(trajectory.to_json())

    # 4. Save an interactive map you can open in a browser
    TrajectoryVisualizer.save_html(trajectory, "trajectory_map.html")
    print("Map saved to trajectory_map.html")
