import os
import cv2
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except Exception:
    PYDECK_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

ENV_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")

ROBOFLOW_API_URL = os.getenv(
    "ROBOFLOW_API_URL",
    "https://serverless.roboflow.com",
)

ROAD_DAMAGE_MODEL_ID = os.getenv(
    "ROAD_DAMAGE_MODEL_ID",
    "moulya-d/road-damage-saoua-wesks-2-yolov8s-t1",
)

VEHICLE_MODEL_ID = os.getenv(
    "VEHICLE_MODEL_ID",
    "rfdetr-small",
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
FRAME_DIR = OUTPUT_DIR / "frames"
ANNOTATED_DIR = OUTPUT_DIR / "annotated_frames"
HISTORY_CSV = OUTPUT_DIR / "road_history.csv"

for folder in [UPLOAD_DIR, OUTPUT_DIR, FRAME_DIR, ANNOTATED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_DAMAGE_CONFIDENCE = 0.10
DEFAULT_VEHICLE_CONFIDENCE = 0.30

DEFAULT_MAX_DAMAGE_AREA_RATIO = 0.80
DEFAULT_MAX_DAMAGE_WIDTH_RATIO = 1.00
DEFAULT_MAX_DAMAGE_HEIGHT_RATIO = 1.00

VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
}

VEHICLE_WEIGHTS_TONS = {
    "car": 1.5,
    "truck": 10.0,
    "bus": 12.0,
    "motorcycle": 0.25,
    "bicycle": 0.10,
}


# ============================================================
# ROBOFLOW CLIENT
# ============================================================

@st.cache_resource
def make_client(api_key: str, confidence: float):
    client = InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=api_key,
    )

    config = InferenceConfiguration(
        confidence_threshold=confidence,
    )

    client.configure(config)
    return client


# ============================================================
# FILE AND VIDEO HELPERS
# ============================================================

def clear_old_outputs():
    for folder in [FRAME_DIR, ANNOTATED_DIR]:
        for file in folder.glob("*"):
            try:
                file.unlink()
            except Exception:
                pass


def save_uploaded_video(uploaded_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = uploaded_file.name.replace(" ", "_")
    video_path = UPLOAD_DIR / f"{timestamp}_{filename}"

    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    return video_path


def extract_frames(video_path, every_n_frames=5, max_frames=100):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = total_frames / fps if total_frames > 0 else 0

    frames = []
    frame_index = 0
    sample_index = 0

    while True:
        success, frame_bgr = cap.read()

        if not success:
            break

        if frame_index % every_n_frames == 0:
            frame_path = FRAME_DIR / f"frame_{sample_index:05d}.jpg"
            cv2.imwrite(str(frame_path), frame_bgr)

            frames.append(
                {
                    "frame_index": frame_index,
                    "sample_index": sample_index,
                    "timestamp_s": frame_index / fps,
                    "frame_bgr": frame_bgr,
                    "frame_path": frame_path,
                }
            )

            sample_index += 1

            if sample_index >= max_frames:
                break

        frame_index += 1

    cap.release()
    return frames, fps, duration_seconds


# ============================================================
# PREDICTION PARSING
# ============================================================

def get_predictions(result):
    if result is None:
        return []

    if isinstance(result, dict):
        preds = result.get("predictions", [])
        if isinstance(preds, list):
            return preds

    if isinstance(result, list):
        return result

    return []


def normalize_detection(det):
    if not isinstance(det, dict):
        return None

    x = det.get("x")
    y = det.get("y")
    w = det.get("width")
    h = det.get("height")

    if w is None or h is None:
        x_min = det.get("x_min")
        y_min = det.get("y_min")
        x_max = det.get("x_max")
        y_max = det.get("y_max")

        if None not in [x_min, y_min, x_max, y_max]:
            w = float(x_max) - float(x_min)
            h = float(y_max) - float(y_min)
            x = float(x_min) + w / 2
            y = float(y_min) + h / 2

    cls = det.get("class", det.get("class_name", det.get("label", "object")))
    conf = det.get("confidence", det.get("score", 0.0))

    try:
        return {
            "x": float(x),
            "y": float(y),
            "width": float(w),
            "height": float(h),
            "confidence": float(conf),
            "class_name": str(cls),
            "raw": det,
        }
    except Exception:
        return None


def normalize_detections(predictions):
    output = []

    for det in predictions:
        nd = normalize_detection(det)
        if nd is not None:
            output.append(nd)

    return output


# ============================================================
# FILTERS
# ============================================================

def filter_damage_detections(
    detections,
    image_width,
    image_height,
    min_confidence,
    max_area_ratio,
    max_width_ratio,
    max_height_ratio,
):
    valid = []

    frame_area = image_width * image_height

    for det in detections:
        conf = det["confidence"]
        w = det["width"]
        h = det["height"]
        area = w * h

        if conf < min_confidence:
            continue

        if area > frame_area * max_area_ratio:
            continue

        if w > image_width * max_width_ratio:
            continue

        if h > image_height * max_height_ratio:
            continue

        valid.append(det)

    return valid


def filter_vehicle_detections(detections, min_confidence):
    valid = []

    for det in detections:
        cls = det["class_name"].lower().strip()
        conf = det["confidence"]

        if conf < min_confidence:
            continue

        if cls not in VEHICLE_CLASSES:
            continue

        valid.append(det)

    return valid


# ============================================================
# DRAWING
# ============================================================

def draw_detections(frame_bgr, detections, color, label_prefix=""):
    output = frame_bgr.copy()

    for det in detections:
        x = det["x"]
        y = det["y"]
        w = det["width"]
        h = det["height"]
        conf = det["confidence"]
        cls = det["class_name"]

        x1 = int(round(x - w / 2))
        y1 = int(round(y - h / 2))
        x2 = int(round(x + w / 2))
        y2 = int(round(y + h / 2))

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(output.shape[1] - 1, x2)
        y2 = min(output.shape[0] - 1, y2)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

        if label_prefix:
            label = f"{label_prefix}: {cls} {conf:.2f}"
        else:
            label = f"{cls} {conf:.2f}"

        label_w = min(x1 + 430, output.shape[1] - 1)
        label_y1 = max(0, y1 - 32)
        cv2.rectangle(output, (x1, label_y1), (label_w, y1), color, -1)

        cv2.putText(
            output,
            label,
            (x1 + 5, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def draw_metric_overlay(frame_bgr, damage_count, vehicle_count):
    output = frame_bgr.copy()

    text = f"Potholes / Road Damage: {damage_count} | Vehicles: {vehicle_count}"

    cv2.rectangle(output, (10, 10), (850, 62), (0, 0, 0), -1)

    cv2.putText(
        output,
        text,
        (20, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def build_annotated_video(frame_paths, output_video_path, fps=10):
    if not frame_paths:
        return None

    first = cv2.imread(str(frame_paths[0]))

    if first is None:
        return None

    height, width = first.shape[:2]

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for path in frame_paths:
        frame = cv2.imread(str(path))

        if frame is None:
            continue

        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        writer.write(frame)

    writer.release()
    return output_video_path


# ============================================================
# HISTORY, LOCATION, COST, LIFESPAN, PRIORITY
# ============================================================

def normalize_location_name(location_name):
    location_name = str(location_name).strip().lower()
    location_name = location_name.replace(" ", "_")
    location_name = "".join(
        ch for ch in location_name
        if ch.isalnum() or ch in ["_", "-", ","]
    )
    return location_name or "unknown_road"


def load_history():
    if HISTORY_CSV.exists():
        try:
            return pd.read_csv(HISTORY_CSV)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def append_history(record):
    history = load_history()
    new_row = pd.DataFrame([record])

    if history.empty:
        updated = new_row
    else:
        updated = pd.concat([history, new_row], ignore_index=True)

    updated.to_csv(HISTORY_CSV, index=False)
    return updated


def get_location_history(history_df, location_id):
    if history_df.empty or "location_id" not in history_df.columns:
        return pd.DataFrame()

    df = history_df[history_df["location_id"] == location_id].copy()

    if "analysis_time" in df.columns:
        df["analysis_time"] = pd.to_datetime(df["analysis_time"], errors="coerce")
        df = df.sort_values("analysis_time")

    return df


def compute_damage_trend(location_history, current_damage_index):
    if location_history.empty or len(location_history) < 1:
        return "first_upload", 1.00, 0.0

    previous = location_history.iloc[-1]
    previous_index = float(previous.get("damage_index", 0.0))

    delta = current_damage_index - previous_index

    if previous_index <= 0:
        if current_damage_index > 0:
            return "new_damage_detected", 1.20, delta
        return "stable", 1.00, delta

    percent_change = (delta / previous_index) * 100.0

    if percent_change >= 30:
        return "rapidly_worsening", 1.50, delta
    elif percent_change >= 10:
        return "worsening", 1.25, delta
    elif percent_change <= -10:
        return "improving", 0.85, delta
    else:
        return "stable", 1.00, delta


def estimate_traffic_load(vehicle_detections, duration_seconds, assumed_speed_kmph):
    duration_minutes = max(duration_seconds / 60.0, 1 / 60.0)

    type_counts = {}
    total_load = 0.0

    for det in vehicle_detections:
        cls = det["class_name"].lower().strip()
        type_counts[cls] = type_counts.get(cls, 0) + 1

    for cls, count in type_counts.items():
        weight = VEHICLE_WEIGHTS_TONS.get(cls, 1.0)
        frequency_per_minute = count / duration_minutes
        total_load += weight * assumed_speed_kmph * frequency_per_minute

    return total_load, type_counts


def estimate_repair_cost(
    damage_count,
    damage_area_proxy,
    traffic_load,
    base_inspection_cost=2500,
    cost_per_damage=1200,
    cost_per_area_unit=0.03,
    traffic_load_cost_factor=0.50,
):
    cost = (
        base_inspection_cost
        + damage_count * cost_per_damage
        + damage_area_proxy * cost_per_area_unit
        + traffic_load * traffic_load_cost_factor
    )

    return max(cost, 0.0)


def estimate_remaining_life_days(
    damage_count,
    damage_area_proxy,
    traffic_load,
    rainfall_mm,
    damage_trend_multiplier,
):
    base_life = 365.0

    damage_penalty = damage_count * 10.0
    area_penalty = min(damage_area_proxy / 6000.0, 150.0)
    traffic_penalty = min(traffic_load / 80.0, 120.0)
    rain_penalty = rainfall_mm * 1.0

    total_penalty = (
        damage_penalty
        + area_penalty
        + traffic_penalty
        + rain_penalty
    ) * damage_trend_multiplier

    remaining = base_life - total_penalty

    return max(remaining, 7.0)


def calculate_priority_score(
    remaining_life_days,
    traffic_density,
    damage_area_proxy,
    damage_count,
    damage_trend_multiplier,
):
    remaining_life_days = max(remaining_life_days, 1.0)

    severity = damage_count + (damage_area_proxy / 10000.0)

    score = (
        severity
        * max(traffic_density, 1.0)
        * damage_trend_multiplier
        / remaining_life_days
    )

    return score


# ============================================================
# MAP AND DIGITAL TWIN
# ============================================================

def render_exact_location_map(road_name, city, lat_text, lon_text, priority_score):
    if not lat_text.strip() or not lon_text.strip():
        st.info("Add latitude and longitude to highlight the exact road location on the map.")
        return

    try:
        lat = float(lat_text)
        lon = float(lon_text)
    except Exception:
        st.warning("Latitude or longitude format is invalid. Example: 12.9716, 77.5946")
        return

    map_df = pd.DataFrame(
        {
            "lat": [lat],
            "lon": [lon],
            "road": [road_name],
            "city": [city],
            "priority_score": [priority_score],
        }
    )

    if PYDECK_AVAILABLE:
        color = [255, 0, 0, 180] if priority_score >= 1 else [255, 165, 0, 180]

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_radius=90,
            get_fill_color=color,
            pickable=True,
        )

        text_layer = pdk.Layer(
            "TextLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_text="road",
            get_size=16,
            get_color=[0, 0, 0],
            get_angle=0,
            get_alignment_baseline="'bottom'",
        )

        view_state = pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=16,
            pitch=45,
        )

        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/streets-v11",
            initial_view_state=view_state,
            layers=[layer, text_layer],
            tooltip={
                "text": "Road: {road}\nCity: {city}\nPriority: {priority_score}"
            },
        )

        st.pydeck_chart(deck)
    else:
        st.map(map_df[["lat", "lon"]])
        st.caption("Install pydeck for highlighted marker styling: pip install pydeck")


def create_uploaded_road_virtual_twin(frame_bgr, damage_detections, vehicle_detections):
    """
    Creates a virtual road twin from the user's actual uploaded video frame.

    This is not true photogrammetry.
    It uses the actual uploaded road frame as the visual source and projects
    pothole/road-damage detections into a 3D-style road surface.
    """
    if frame_bgr is None:
        return None

    h, w = frame_bgr.shape[:2]

    twin_w = 120
    twin_h = max(60, int(twin_w * h / w))

    small = cv2.resize(frame_bgr, (twin_w, twin_h))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gray_norm = gray / 255.0

    # Create a subtle road surface elevation from image texture.
    z = gray_norm * 0.15

    # Increase local elevation where road damage is detected.
    for det in damage_detections:
        cx = int(np.clip(det["x"] / w * twin_w, 0, twin_w - 1))
        cy = int(np.clip(det["y"] / h * twin_h, 0, twin_h - 1))

        bw = max(2, int(det["width"] / w * twin_w))
        bh = max(2, int(det["height"] / h * twin_h))

        x1 = max(0, cx - bw // 2)
        x2 = min(twin_w, cx + bw // 2)
        y1 = max(0, cy - bh // 2)
        y2 = min(twin_h, cy + bh // 2)

        z[y1:y2, x1:x2] += 1.0

    x = np.linspace(0, 1, twin_w)
    y = np.linspace(0, 1, twin_h)

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=gray,
            colorscale="gray",
            opacity=0.92,
            showscale=False,
            name="Uploaded road surface",
        )
    )

    # Red markers for potholes / damage.
    damage_x = []
    damage_y = []
    damage_z = []
    damage_text = []

    for det in damage_detections:
        dx = det["x"] / w
        dy = det["y"] / h
        dz = 1.3
        damage_x.append(dx)
        damage_y.append(dy)
        damage_z.append(dz)
        damage_text.append(f"{det['class_name']} {det['confidence']:.2f}")

    if damage_x:
        fig.add_trace(
            go.Scatter3d(
                x=damage_x,
                y=damage_y,
                z=damage_z,
                mode="markers+text",
                marker=dict(size=8, color="red"),
                text=damage_text,
                textposition="top center",
                name="Pothole / damage",
            )
        )

    # Green markers for vehicles.
    vehicle_x = []
    vehicle_y = []
    vehicle_z = []
    vehicle_text = []

    for det in vehicle_detections:
        vx = det["x"] / w
        vy = det["y"] / h
        vz = 0.45
        vehicle_x.append(vx)
        vehicle_y.append(vy)
        vehicle_z.append(vz)
        vehicle_text.append(f"{det['class_name']} {det['confidence']:.2f}")

    if vehicle_x:
        fig.add_trace(
            go.Scatter3d(
                x=vehicle_x,
                y=vehicle_y,
                z=vehicle_z,
                mode="markers",
                marker=dict(size=5, color="lime"),
                text=vehicle_text,
                name="Vehicles",
            )
        )

    fig.update_layout(
        title="Virtual Road Digital Twin, Built From Uploaded Video Frame",
        scene=dict(
            xaxis_title="Road frame width",
            yaxis_title="Road frame height",
            zaxis_title="Damage severity / surface variation",
            camera=dict(
                eye=dict(x=1.4, y=-1.6, z=1.1)
            ),
        ),
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


def create_damage_severity_timeline_surface(df):
    if df.empty:
        return None

    times = df["timestamp_s"].values

    if len(times) < 2:
        times = np.array([0, 1])

    road_width_bands = np.linspace(0, 1, 20)

    z = []

    for _, row in df.iterrows():
        damage = float(row.get("damage_area_proxy", 0.0))
        count = float(row.get("filtered_damage_count", 0.0))

        severity = min((damage / 50000.0) + count, 10.0)

        profile = severity * np.exp(-((road_width_bands - 0.5) ** 2) / 0.08)

        z.append(profile)

    z = np.array(z).T

    fig = go.Figure(
        data=[
            go.Surface(
                x=times,
                y=road_width_bands,
                z=z,
                colorscale="Turbo",
                colorbar=dict(title="Damage severity"),
            )
        ]
    )

    fig.update_layout(
        title="Damage Severity Timeline Surface",
        scene=dict(
            xaxis_title="Video time, seconds",
            yaxis_title="Road width, normalized",
            zaxis_title="Damage severity",
        ),
        height=650,
    )

    return fig


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="AI Road Assessment Dashboard",
    layout="wide",
)

st.title("AI-Based Road Assessment and Traffic Load Analysis")
st.caption(
    "Pothole detection, traffic load estimation, repair cost, remaining lifespan, historical comparison, exact map marker, and virtual digital twin"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Settings")

st.sidebar.subheader("Road Location")

road_name = st.sidebar.text_input(
    "Road / junction name",
    value="",
    placeholder="Example: MG Road Junction",
)

road_city = st.sidebar.text_input(
    "City / area",
    value="",
    placeholder="Example: Bengaluru",
)

road_latitude = st.sidebar.text_input(
    "Latitude",
    value="",
    placeholder="Example: 12.9716",
)

road_longitude = st.sidebar.text_input(
    "Longitude",
    value="",
    placeholder="Example: 77.5946",
)

st.sidebar.markdown("---")

st.sidebar.subheader("Roboflow API")

api_key_from_ui = st.sidebar.text_input(
    "Roboflow API Key",
    value=ENV_API_KEY if ENV_API_KEY and not ENV_API_KEY.startswith("YOUR_") else "",
    type="password",
)

ROBOFLOW_API_KEY = api_key_from_ui.strip()

if ROBOFLOW_API_KEY:
    st.sidebar.success("API key loaded")
else:
    st.sidebar.warning("API key missing")

st.sidebar.markdown("---")

st.sidebar.subheader("Models")

road_damage_model_id = st.sidebar.text_input(
    "Road damage model ID",
    value=ROAD_DAMAGE_MODEL_ID,
)

vehicle_model_id = st.sidebar.text_input(
    "Vehicle model ID",
    value=VEHICLE_MODEL_ID,
)

st.sidebar.markdown("---")

st.sidebar.subheader("Video Sampling")

every_n_frames = st.sidebar.slider(
    "Analyze every Nth frame",
    min_value=1,
    max_value=30,
    value=5,
)

max_frames = st.sidebar.slider(
    "Maximum frames to analyze",
    min_value=5,
    max_value=300,
    value=80,
)

st.sidebar.markdown("---")

st.sidebar.subheader("Pothole Detection")

damage_confidence = st.sidebar.slider(
    "Pothole / damage confidence",
    min_value=0.01,
    max_value=0.90,
    value=DEFAULT_DAMAGE_CONFIDENCE,
    step=0.01,
)

max_area_ratio = st.sidebar.slider(
    "Max damage box area ratio",
    min_value=0.05,
    max_value=1.00,
    value=DEFAULT_MAX_DAMAGE_AREA_RATIO,
    step=0.05,
    help="Higher keeps larger pothole boxes. Lower removes full-frame false positives.",
)

max_width_ratio = st.sidebar.slider(
    "Max damage box width ratio",
    min_value=0.10,
    max_value=1.00,
    value=DEFAULT_MAX_DAMAGE_WIDTH_RATIO,
    step=0.05,
)

max_height_ratio = st.sidebar.slider(
    "Max damage box height ratio",
    min_value=0.10,
    max_value=1.00,
    value=DEFAULT_MAX_DAMAGE_HEIGHT_RATIO,
    step=0.05,
)

st.sidebar.markdown("---")

st.sidebar.subheader("Traffic and Environment")

vehicle_confidence = st.sidebar.slider(
    "Vehicle confidence",
    min_value=0.05,
    max_value=0.90,
    value=DEFAULT_VEHICLE_CONFIDENCE,
    step=0.05,
)

assumed_speed_kmph = st.sidebar.slider(
    "Assumed average speed, km/h",
    min_value=5,
    max_value=80,
    value=25,
)

rainfall_mm = st.sidebar.number_input(
    "Rainfall, mm",
    min_value=0.0,
    value=0.0,
    step=1.0,
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_video = st.file_uploader(
    "Upload road video",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_video is None:
    st.info("Upload a road video to begin.")
    st.stop()


# ============================================================
# RUN
# ============================================================

if st.button("Run full analysis", type="primary"):
    if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY.startswith("YOUR_"):
        st.error("Paste your real Roboflow API key in the sidebar.")
        st.stop()

    if not road_name.strip():
        st.error("Enter the road or junction name in the sidebar before running analysis.")
        st.stop()

    if not road_latitude.strip() or not road_longitude.strip():
        st.error("Enter latitude and longitude so the exact road location can be highlighted.")
        st.stop()

    try:
        float(road_latitude)
        float(road_longitude)
    except Exception:
        st.error("Latitude and longitude must be numeric. Example: 12.9716 and 77.5946")
        st.stop()

    location_label = f"{road_name.strip()}, {road_city.strip()}".strip(", ")
    location_id = normalize_location_name(location_label)

    clear_old_outputs()

    video_path = save_uploaded_video(uploaded_video)
    st.success(f"Uploaded: {video_path.name}")

    try:
        damage_client = make_client(ROBOFLOW_API_KEY, damage_confidence)
        vehicle_client = make_client(ROBOFLOW_API_KEY, vehicle_confidence)
    except Exception as e:
        st.error(f"Could not create Roboflow clients: {e}")
        st.stop()

    try:
        with st.spinner("Extracting frames..."):
            frames, source_fps, duration_seconds = extract_frames(
                video_path,
                every_n_frames=every_n_frames,
                max_frames=max_frames,
            )
    except Exception as e:
        st.error(f"Could not extract frames: {e}")
        st.stop()

    if not frames:
        st.error("No frames extracted from video.")
        st.stop()

    st.write(f"Extracted {len(frames)} frames from {duration_seconds:.1f} seconds.")

    rows = []
    annotated_paths = []
    all_vehicle_detections = []
    api_errors = []

    progress = st.progress(0)
    status = st.empty()

    debug_first_damage_raw = []
    debug_first_damage_filtered = []

    representative_frame = None
    representative_damage = []
    representative_vehicles = []
    representative_score = -1

    for i, frame_item in enumerate(frames):
        status.write(f"Processing frame {i + 1}/{len(frames)}")

        frame_bgr = frame_item["frame_bgr"]
        height, width = frame_bgr.shape[:2]

        model_input = frame_bgr

        # -----------------------------
        # Road damage inference
        # -----------------------------
        try:
            damage_result = damage_client.infer(
                model_input,
                model_id=road_damage_model_id,
            )
            raw_damage = normalize_detections(get_predictions(damage_result))

        except Exception as e:
            raw_damage = []
            api_errors.append(f"Road damage model error on frame {i}: {e}")

        filtered_damage = filter_damage_detections(
            raw_damage,
            image_width=width,
            image_height=height,
            min_confidence=damage_confidence,
            max_area_ratio=max_area_ratio,
            max_width_ratio=max_width_ratio,
            max_height_ratio=max_height_ratio,
        )

        if i == 0:
            debug_first_damage_raw = raw_damage[:5]
            debug_first_damage_filtered = filtered_damage[:5]

        # -----------------------------
        # Vehicle inference
        # -----------------------------
        try:
            vehicle_result = vehicle_client.infer(
                model_input,
                model_id=vehicle_model_id,
            )
            raw_vehicles = normalize_detections(get_predictions(vehicle_result))

        except Exception as e:
            raw_vehicles = []
            api_errors.append(f"Vehicle model error on frame {i}: {e}")

        filtered_vehicles = filter_vehicle_detections(
            raw_vehicles,
            min_confidence=vehicle_confidence,
        )

        all_vehicle_detections.extend(filtered_vehicles)

        # -----------------------------
        # Frame metrics
        # -----------------------------
        damage_count = len(filtered_damage)
        vehicle_count = len(filtered_vehicles)

        damage_area_proxy = (
            float(np.sum([d["width"] * d["height"] for d in filtered_damage]))
            if filtered_damage
            else 0.0
        )

        vehicle_classes = [d["class_name"] for d in filtered_vehicles]

        frame_score = damage_count + (damage_area_proxy / 10000.0) + (vehicle_count * 0.05)

        if frame_score > representative_score:
            representative_score = frame_score
            representative_frame = frame_bgr.copy()
            representative_damage = list(filtered_damage)
            representative_vehicles = list(filtered_vehicles)

        # -----------------------------
        # Visualization
        # -----------------------------
        annotated = frame_bgr.copy()

        annotated = draw_detections(
            annotated,
            filtered_damage,
            color=(0, 0, 255),
            label_prefix="damage",
        )

        annotated = draw_detections(
            annotated,
            filtered_vehicles,
            color=(0, 180, 0),
            label_prefix="vehicle",
        )

        annotated = draw_metric_overlay(
            annotated,
            damage_count=damage_count,
            vehicle_count=vehicle_count,
        )

        annotated_path = ANNOTATED_DIR / f"annotated_{frame_item['sample_index']:05d}.jpg"
        cv2.imwrite(str(annotated_path), annotated)
        annotated_paths.append(annotated_path)

        rows.append(
            {
                "frame_index": frame_item["frame_index"],
                "timestamp_s": round(frame_item["timestamp_s"], 2),
                "raw_damage_count": len(raw_damage),
                "filtered_damage_count": damage_count,
                "damage_area_proxy": round(damage_area_proxy, 2),
                "vehicle_count": vehicle_count,
                "vehicle_classes": ", ".join(vehicle_classes),
            }
        )

        progress.progress((i + 1) / len(frames))

    status.write("Analysis complete.")

    df = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "road_assessment_results.csv"
    df.to_csv(csv_path, index=False)

    annotated_video_path = OUTPUT_DIR / "annotated_output.mp4"
    output_fps = max(source_fps / every_n_frames, 5)
    build_annotated_video(annotated_paths, annotated_video_path, fps=output_fps)

    # ========================================================
    # AGGREGATE METRICS
    # ========================================================

    total_raw_damage_count = int(df["raw_damage_count"].sum())
    total_damage_count = int(df["filtered_damage_count"].sum())
    total_damage_area = float(df["damage_area_proxy"].sum())
    total_vehicle_count = int(df["vehicle_count"].sum())

    duration_minutes = max(duration_seconds / 60.0, 1 / 60.0)
    traffic_density = total_vehicle_count / duration_minutes

    traffic_load, vehicle_type_counts = estimate_traffic_load(
        all_vehicle_detections,
        duration_seconds=duration_seconds,
        assumed_speed_kmph=assumed_speed_kmph,
    )

    damage_index = total_damage_count + (total_damage_area / 10000.0)

    history_before = load_history()
    location_history_before = get_location_history(history_before, location_id)

    damage_trend, trend_multiplier, damage_delta = compute_damage_trend(
        location_history_before,
        current_damage_index=damage_index,
    )

    remaining_life_days = estimate_remaining_life_days(
        damage_count=total_damage_count,
        damage_area_proxy=total_damage_area,
        traffic_load=traffic_load,
        rainfall_mm=rainfall_mm,
        damage_trend_multiplier=trend_multiplier,
    )

    estimated_cost = estimate_repair_cost(
        damage_count=total_damage_count,
        damage_area_proxy=total_damage_area,
        traffic_load=traffic_load,
    )

    priority_score = calculate_priority_score(
        remaining_life_days=remaining_life_days,
        traffic_density=traffic_density,
        damage_area_proxy=total_damage_area,
        damage_count=total_damage_count,
        damage_trend_multiplier=trend_multiplier,
    )

    analysis_record = {
        "analysis_time": datetime.now().isoformat(timespec="seconds"),
        "location_id": location_id,
        "road_name": road_name.strip(),
        "city_area": road_city.strip(),
        "latitude": road_latitude.strip(),
        "longitude": road_longitude.strip(),
        "video_file": video_path.name,
        "duration_seconds": round(duration_seconds, 2),
        "raw_damage_count": total_raw_damage_count,
        "filtered_damage_count": total_damage_count,
        "damage_area_proxy": round(total_damage_area, 2),
        "damage_index": round(damage_index, 4),
        "vehicle_count": total_vehicle_count,
        "traffic_density_per_min": round(traffic_density, 2),
        "traffic_load_proxy": round(traffic_load, 2),
        "damage_trend": damage_trend,
        "damage_delta_vs_previous": round(damage_delta, 4),
        "trend_multiplier": trend_multiplier,
        "remaining_life_days": round(remaining_life_days, 2),
        "estimated_repair_cost_inr": round(estimated_cost, 2),
        "priority_score": round(priority_score, 4),
    }

    history_after = append_history(analysis_record)
    location_history_after = get_location_history(history_after, location_id)

    # ========================================================
    # DASHBOARD
    # ========================================================

    st.header("Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Potholes / damage", total_damage_count)
    c2.metric("Vehicles", total_vehicle_count)
    c3.metric("Traffic density", f"{traffic_density:.1f}/min")
    c4.metric("Remaining life", f"{remaining_life_days:.0f} days")
    c5.metric("Priority score", f"{priority_score:.2f}")

    c6, c7, c8 = st.columns(3)

    c6.metric("Estimated repair cost", f"₹{estimated_cost:,.0f}")
    c7.metric("Damage trend", damage_trend.replace("_", " ").title())
    c8.metric("Compared with previous", f"{damage_delta:+.2f}")

    if total_raw_damage_count > 0 and total_damage_count == 0:
        st.warning(
            "The model returned road-damage detections, but the app filter removed them. "
            "Increase max area, width, or height ratios, or lower damage confidence."
        )

    if total_raw_damage_count == 0:
        st.warning(
            "The road damage model returned zero pothole or road-damage detections. "
            "Try lowering damage confidence, using closer road-level footage, or checking the road damage model ID."
        )

    if api_errors:
        with st.expander("API errors"):
            for err in api_errors[:20]:
                st.error(err)

    with st.expander("Debug, first frame damage predictions"):
        st.write("Raw first-frame damage predictions sample:")
        st.write(debug_first_damage_raw)

        st.write("Filtered first-frame damage predictions sample:")
        st.write(debug_first_damage_filtered)

    st.subheader("Exact Road Location Highlighted on Map")

    render_exact_location_map(
        road_name=road_name.strip(),
        city=road_city.strip(),
        lat_text=road_latitude,
        lon_text=road_longitude,
        priority_score=priority_score,
    )

    st.subheader("Road Location Details")

    loc_col1, loc_col2 = st.columns(2)

    with loc_col1:
        st.write("Road / junction:", road_name.strip())
        st.write("City / area:", road_city.strip() or "Not provided")
        st.write("Latitude:", road_latitude.strip())
        st.write("Longitude:", road_longitude.strip())

    with loc_col2:
        st.write("Location ID:", location_id)
        st.write("Priority score:", round(priority_score, 4))
        st.write("Damage trend:", damage_trend.replace("_", " ").title())

    st.subheader("Annotated Video")

    if annotated_video_path.exists():
        st.video(str(annotated_video_path))

        with open(annotated_video_path, "rb") as video_file:
            st.download_button(
                label="Download annotated video",
                data=video_file,
                file_name="annotated_road_assessment.mp4",
                mime="video/mp4",
                key="download_annotated_video_main",
            )
    else:
        st.error("Annotated video was not created.")

    st.subheader("Virtual Digital Twin of Uploaded Road")

    if representative_frame is not None:
        preview_rgb = cv2.cvtColor(representative_frame, cv2.COLOR_BGR2RGB)
        st.image(
            preview_rgb,
            caption="Actual uploaded road frame used as the virtual twin base",
            use_container_width=True,
        )

        twin_fig = create_uploaded_road_virtual_twin(
            frame_bgr=representative_frame,
            damage_detections=representative_damage,
            vehicle_detections=representative_vehicles,
        )

        if twin_fig is not None:
            st.plotly_chart(twin_fig, use_container_width=True)

        st.caption(
            "This virtual twin uses the exact uploaded road frame as the base and projects detected damage/vehicles into a 3D-style surface. "
            "For true survey-grade 3D, add COLMAP or depth reconstruction using close multi-view road-level footage."
        )
    else:
        st.info("No representative frame available to create the virtual road twin.")

    st.subheader("Damage Severity Timeline Surface")

    timeline_twin_fig = create_damage_severity_timeline_surface(df)

    if timeline_twin_fig is not None:
        st.plotly_chart(timeline_twin_fig, use_container_width=True)
    else:
        st.info("Not enough frame data to build the timeline surface.")

    st.subheader("Trends")

    t1, t2 = st.columns(2)

    with t1:
        fig_damage = px.line(
            df,
            x="timestamp_s",
            y=["raw_damage_count", "filtered_damage_count"],
            title="Road Damage Detections Over Time",
            markers=True,
        )
        st.plotly_chart(fig_damage, use_container_width=True)

    with t2:
        fig_vehicle = px.line(
            df,
            x="timestamp_s",
            y="vehicle_count",
            title="Vehicle Count Over Time",
            markers=True,
        )
        st.plotly_chart(fig_vehicle, use_container_width=True)

    st.subheader("Comparison With Previous Uploads")

    if len(location_history_after) > 1:
        hist_fig = px.line(
            location_history_after,
            x="analysis_time",
            y=["damage_index", "traffic_density_per_min", "priority_score"],
            markers=True,
            title=f"Historical Trend for {location_label}",
        )
        st.plotly_chart(hist_fig, use_container_width=True)

        st.dataframe(location_history_after.tail(10), use_container_width=True)
    else:
        st.info("This is the first saved upload for this road location.")

    st.subheader("Vehicle Type Counts")

    if vehicle_type_counts:
        vehicle_df = pd.DataFrame(
            [
                {"vehicle_type": k, "count": v}
                for k, v in vehicle_type_counts.items()
            ]
        )

        fig_vehicle_types = px.bar(
            vehicle_df,
            x="vehicle_type",
            y="count",
            title="Detected Vehicle Types",
        )

        st.plotly_chart(fig_vehicle_types, use_container_width=True)
    else:
        st.info("No vehicles detected.")

    st.subheader("Maintenance Priority Record")

    ranking_df = pd.DataFrame([analysis_record])
    st.dataframe(ranking_df, use_container_width=True)

    st.subheader("Frame-Level Results")

    st.dataframe(df, use_container_width=True)

    st.subheader("Downloads")

    with open(csv_path, "rb") as f:
        st.download_button(
            "Download frame-level CSV",
            data=f,
            file_name="road_assessment_results.csv",
            mime="text/csv",
        )

    if HISTORY_CSV.exists():
        with open(HISTORY_CSV, "rb") as f:
            st.download_button(
                "Download full road history CSV",
                data=f,
                file_name="road_history.csv",
                mime="text/csv",
            )

    if annotated_video_path.exists():
        with open(annotated_video_path, "rb") as f:
            st.download_button(
                "Download annotated video again",
                data=f,
                file_name="annotated_road_assessment.mp4",
                mime="video/mp4",
                key="download_annotated_video_bottom",
            )





