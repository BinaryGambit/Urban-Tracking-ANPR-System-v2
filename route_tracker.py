import json
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import folium

EVENTS_FILE = Path("multi_camera_output/events.json")
CAMERA_FILE = Path("camera_config.json")

OUTPUT_DIR = Path("map_ouput")
ROUTES_FILE = OUTPUT_DIR / "routes.json"
MAP_FILE = OUTPUT_DIR / "map.html"

with open(CAMERA_FILE,"r",encoding="utf-8") as f:
    cameras = json.load(f)

with open(EVENTS_FILE,"r",encoding="utf-8") as f:
    data = json.load(f)

events = data.get("events",[])

print(f"Loaded {len(events)} ANPR Events")

vehicles = defaultdict(list)

for event in events:
    plate = event.get("Plate")

    camera_id = event.get("camera_id")

    timestamp = event.get("timestamp")

    if not plate:
        continue

    if camera_id not in cameras:
        print(f"WARNING: {camera_id}",f"not found in camera_config.json")
        continue
    if not timestamp:
        continue

def haversine(lat1,lon1,lat2,lon2):
    R = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)

    dlon = math.rradians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        +
        math.cos(lat2)
        +
        math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


routes = []

for plate, plate_events in vehicles.items():
    plate_events.sort(key=lambda x: x['timestamp'])

    visits = []

    last_camera = camera_id

    if len(visits) < 2:
        continue

    route_points = []

    transitions=[]

    for event in visits:
        camera_id = event["camera_id"]

        camera = cameras[camera_id]

        route_points.append({
            "camera_id":camera_id,
            "camera_name":camera["name"],
            "latitude":camera["latitude"],
            "longitude":camera["longitude"],
            "timestamp":event["timestamp"],
            "plate":plate
        })

    for i in range(len(route_points) - 1):
        current = route_points[i]

        next_point = route_points[i+1]

        t1 = datetime.fromisoformat(
            current["timestamp"]
        )
        t2 = datetime.fromisoformat(
            next_point["timestamp"]
        )

        travel_time = (
            t2 - t1
        ).total_seconds()

        distance = haversine(
            current["latitude"],
            current["longitude"],
            next_point["latitude"],
            next_point["longitude"]
        )

        speed = None

        if travel_time > 0:
            speed = (distance/(travel_time/3600))

        transitions.append({
            "from_camera":current["camera_id"],
            "to_camera":next_point["camera_id"],
            "start_time":current["timestamp"],
            "end_time":next_point["timestamp"],
            "travel_time_seconds":travel_time,
            "distance_km":round(distance, 3),
            "estimated_speed_kmph":
                    round(speed,2)
                    if speed is not None
                    else None
        })

        routes.append({
            "plate":plate,
            "camera_count":len(route_points),
            "route":route_points,
            "transitions":transitions
        })

route_data = {
    "total_vehicles":len(routes),
    "vehicles":routes
}

OUTPUT_DIR.mkdir(
    exist_ok = True
)

with open(ROUTES_FILE,"w",encoding="utf-8") as f:
    json.dump(
        route_data,
        f,
        indent=4
    )

print(f"Routes saved to: {ROUTES_FILE}")

if cameras:
    first_camera = next(iter(cameras.values()))
    map_center = [
        first_camera["latitude"],
        first_camera["longitude"]
    ]

else:
    map_center = [
        20.5937,
        78.9629
    ]

m = folium.Map(
    location=map_center,
    zoom_start=14
)

for camera_id, camera in cameras.items():
    folium.Marker(

        location=[
            camera["latitude"],
            camera["longitude"]
        ],

        popup=(
            f"<b>{camera_id}</b><br>"
            f"{camera['name']}"
        ),

        tooltip=camera_id,

        icon=folium.Icon(
            icon="camera",
            prefix="fa"
        )

    ).add_to(m)

for vehicle in routes:

    plate = vehicle["plate"]

    points = vehicle["route"]


    coordinates = [

        [
            point["latitude"],
            point["longitude"]
        ]

        for point in points
    ]


    folium.PolyLine(

        coordinates,

        tooltip=(
            f"Vehicle: {plate}"
        ),

        popup=(
            f"<b>Vehicle:</b> {plate}<br>"
            f"<b>Cameras:</b> "
            f"{len(points)}"
        ),

        weight=5

    ).add_to(m)


    for point in points:

        popup_html = (

            f"<b>Vehicle:</b> {plate}<br>"

            f"<b>Camera:</b> "
            f"{point['camera_id']}<br>"

            f"<b>Time:</b> "
            f"{point['timestamp']}"

        )


        folium.CircleMarker(

            location=[
                point["latitude"],
                point["longitude"]
            ],

            radius=7,

            popup=popup_html,

            tooltip=plate,

            fill=True

        ).add_to(m)



m.save(
    str(MAP_FILE)
)


print(
    f"Map saved to: {MAP_FILE}"
)

print("\n\n=============\nROUTE TRACKING COMPLETE\n=============\n\n")