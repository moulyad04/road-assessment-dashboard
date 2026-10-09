# AI-Based Road Assessment and Traffic Load Analysis

A Streamlit dashboard that analyses a road video and estimates how urgently the road needs repair. It detects potholes and road damage, counts vehicles, and estimates traffic load, repair cost, remaining road life and a maintenance priority score. Results are saved per location, so each new upload is compared with earlier ones.

Detection runs on [Roboflow](https://roboflow.com) hosted models.

---

## Features

- **Pothole and road-damage detection** on sampled video frames, with filters that discard oversized (false-positive) boxes
- **Vehicle detection** (car, truck, bus, motorcycle, bicycle) and a traffic-load estimate
- **Repair cost estimate** in INR
- **Remaining road life** in days
- **Maintenance priority score** for ranking roads
- **History and trend tracking** per road: first upload, stable, improving, worsening, rapidly worsening, or new damage detected
- **Map marker** for the exact road location (needs latitude and longitude)
- **Virtual digital twin**: a 3D-style surface built from the worst frame, with damage and vehicles marked
- **Damage severity timeline** surface over the video's duration
- **Annotated output video** and downloadable CSVs (frame-level results and full road history)

---

## How it works

```
Upload video
    |
Extract every Nth frame (up to a maximum)
    |
For each frame:
    Road-damage model  -> filter by confidence and box size
    Vehicle model      -> keep only vehicle classes above the confidence threshold
    Draw boxes and counts on the frame
    |
Combine all frames into totals
    |
Compare with this road's previous uploads (trend)
    |
Estimate traffic load, remaining life, repair cost and priority
    |
Save to history, then show the dashboard
```

### Formulas

All values are estimates built from detection counts and box areas. The box area is in pixels, so it only works as a relative measure.

```
damage_index   = damage_count + damage_area / 10,000

traffic_load   = sum over vehicle types of
                 (type weight in tons x assumed speed x count per minute)

repair_cost    = 2,500
               + damage_count x 1,200
               + damage_area x 0.03
               + traffic_load x 0.50

remaining_life = 365 - (10 x damage_count
                      + min(damage_area / 6,000, 150)
                      + min(traffic_load / 80, 120)
                      + rainfall_mm) x trend_multiplier
                 (never below 7 days)

priority_score = (damage_count + damage_area / 10,000)
                 x max(traffic_density, 1) x trend_multiplier
                 / remaining_life
```

**Trend multiplier** (compares this upload's damage index with the previous one for the same location):

| Change | Trend | Multiplier |
|---|---|---|
| No earlier upload | first upload | 1.00 |
| Previous index was 0, now above 0 | new damage detected | 1.20 |
| Up 30% or more | rapidly worsening | 1.50 |
| Up 10% to 30% | worsening | 1.25 |
| Down 10% or more | improving | 0.85 |
| Otherwise | stable | 1.00 |

**Vehicle weights used (tons):** car 1.5, truck 10, bus 12, motorcycle 0.25, bicycle 0.1.

---

## Requirements

- Python 3.9 or newer
- A Roboflow API key
- A road-damage model on Roboflow, and a vehicle model (the default is `rfdetr-small`)

Python packages:

```
streamlit
opencv-python
numpy
pandas
plotly
python-dotenv
inference-sdk
pydeck
```

---

## Setup

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 2. Install packages
pip install streamlit opencv-python numpy pandas plotly python-dotenv inference-sdk pydeck

# 3. Create a .env file next to the app (see below)

# 4. Run
streamlit run app.py
```

Replace `app.py` with the name of your script file.

### `.env` file

```
ROBOFLOW_API_KEY=your_roboflow_api_key
ROBOFLOW_API_URL=https://serverless.roboflow.com
ROAD_DAMAGE_MODEL_ID=your-workspace/your-road-damage-model/1
VEHICLE_MODEL_ID=rfdetr-small
```

Only the API key is required. The other three have defaults in the code. You can also paste the key and model IDs into the sidebar. **Never commit your `.env` file or API key.**

---

## Using the app

1. In the sidebar, enter the **road or junction name**, **city**, **latitude** and **longitude**. Name, latitude and longitude are required to run.
2. Check the API key and model IDs.
3. Adjust the settings if needed:
   - **Video sampling:** analyse every Nth frame, and a maximum frame count. A higher N and a lower maximum run faster and make fewer API calls.
   - **Pothole detection:** confidence and the maximum box area, width and height ratios. Lower the ratios if full-frame boxes appear. Raise them if real potholes are being removed.
   - **Traffic and environment:** vehicle confidence, assumed average speed and rainfall.
4. Upload a video (`mp4`, `mov`, `avi` or `mkv`).
5. Click **Run full analysis**.

If the model finds damage but the filters remove all of it, or the model finds nothing at all, the dashboard shows a warning that says what to adjust. A debug section shows the first frame's raw and filtered predictions.

Each frame needs two Roboflow calls, so a long video with a small frame step makes many requests.

---

## Output

Files are written to these folders, which are created automatically:

| Path | Contents |
|---|---|
| `uploads/` | Uploaded videos |
| `outputs/frames/` | Extracted frames (cleared on every run) |
| `outputs/annotated_frames/` | Frames with boxes drawn (cleared on every run) |
| `outputs/annotated_output.mp4` | Annotated video |
| `outputs/road_assessment_results.csv` | Frame-level results for the latest run |
| `outputs/road_history.csv` | One row per analysis, for every road |

History is matched by the road name and city, so spell them the same way each time to get comparisons.

---

## Limitations

- **The numbers are estimates, not engineering measurements.** Damage area is in image pixels, with no real-world scale. The cost constants, vehicle weights, life formula and priority score are placeholder assumptions. Calibrate them with local data before relying on them for decisions.
- **Traffic load depends on the assumed speed** you set in the sidebar and on the same vehicle being counted in many frames. Treat it as a relative measure.
- **The digital twin is not photogrammetry.** It projects detections onto a surface made from one frame's brightness. For survey-grade 3D, use multi-view reconstruction such as COLMAP.
- **Detection quality depends on your Roboflow model** and on footage taken close to the road surface.
- **The map** uses a Mapbox style, which may need a Mapbox token (`MAPBOX_API_KEY`) to display tiles. Without pydeck the app falls back to a basic map.
- Frames are sent to Roboflow's hosted API, so an internet connection is required.

---

## Author

Moulya D
