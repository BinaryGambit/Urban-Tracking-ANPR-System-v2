from flask import Flask, request, jsonify
from datetime import datetime
import json
import os

app = Flask(__name__)

GPS_FILE = "camera_locations.json"
HISTORY_FILE = "camera_history.json"


def load_json(filename, default):
    if not os.path.exists(filename):
        return default

    try:
        with open(filename, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


@app.route("/gps", methods=["POST"])
def receive_gps():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "status": "error",
            "message": "No JSON received"
        }), 400

    camera_id = data.get("camera_id", "UNKNOWN")

    
    data["server_received"] = datetime.now().isoformat()

    
    locations = load_json(GPS_FILE, {})

    locations[camera_id] = data

    save_json(GPS_FILE, locations)

    

    history = load_json(HISTORY_FILE, {})

    if camera_id not in history:
        history[camera_id] = []

    history[camera_id].append(data)

    save_json(HISTORY_FILE, history)

    

    print(
        f"[{camera_id}] "
        f"Lat: {data.get('latitude')} | "
        f"Lon: {data.get('longitude')} | "
        f"Accuracy: {data.get('accuracy')}m | "
        f"Time: {data.get('timestamp')}"
    )

    return jsonify({
        "status": "ok",
        "camera_id": camera_id
    })




@app.route("/gps", methods=["GET"])
def get_gps():

    return jsonify(
        load_json(GPS_FILE, {})
    )




@app.route("/gps/history", methods=["GET"])
def get_history():

    return jsonify(
        load_json(HISTORY_FILE, {})
    )




@app.route("/gps/history/<camera_id>", methods=["GET"])
def get_camera_history(camera_id):

    history = load_json(HISTORY_FILE, {})

    return jsonify(
        history.get(camera_id, [])
    )




if __name__ == "__main__":

    print("=" * 50)
    print("UrbanTracker GPS Server")
    print("=" * 50)

    print()
    print("Latest GPS:")
    print("http://10.200.195.29:5000/gps")

    print()
    print("All GPS history:")
    print("http://10.200.195.29:5000/gps/history")

    print()
    print("CAM_01 history:")
    print("http://10.200.195.29:5000/gps/history/CAM_01")

    print()
    print("Server running...")
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )