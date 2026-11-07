# 🎯 AttentionAI & CampusGuard — Real-Time Focus Tracker & Multi-Camera Sentinel Monitor

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-FF6F00.svg)](https://developers.google.com/mediapipe)
[![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced real-time computer vision system inspired by **AttentionAI** and engineered for college campus classrooms, study halls, and personal focus tracking. The system pairs Google MediaPipe’s dense 478-point 3D facial landmark mesh with Ultralytics YOLO11 object detection and YOLO11-Pose keypoint tracking to provide millisecond-accurate focus analysis, eye aspect ratio (EAR), blink rates, gaze tracking, head pose (Yaw/Pitch/Roll via solvePnP), mobile phone violation alerts, and cumulative session donut analytics.

---

## 📸 Interface Preview & Layout

The dashboard features a high-performance, two-panel split screen with a retro-modern dark/beige aesthetic:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🎓 AttentionAI - Real-Time Focus Tracker  ● Live  [🔊 Audio]  [↺ Reset Session]  [🎨 Theme]  [📱 Connect Phone]│
├─────────────────────────────────────────────┬───────────────────────────────────────────────────────────────┤
│ 📹 LIVE FEED                       19.4 FPS │ CURRENT STATE                                     🎯          │
│ ┌─────────────────────────────────────────┐ │ Focused                                                       │
│ │ [HUD Overlay]                           │ │ [████████████████████░░░░] Confidence: 94%                    │
│ │ FPS: 19.4    Yaw:   -0.8°               │ ├───────────────────────────────────────────────────────────────┤
│ │ EAR: 0.284   Pitch: +1.2°               │ │  EYE ASPECT RATIO  │     BLINKS/MIN     │    GAZE DIRECTION   │
│ │ Blinks: 18/m Roll:  -0.4°               │ │       0.284        │         18         │       CENTER ◉      │
│ │ Gaze: center                            │ │     Eyes Open      │   Target: 12-24    │     Forward View    │
│ │                                         │ ├───────────────────────────────────────────────────────────────┤
│ │        ┌─────────────┐                  │ │ HEAD POSE (DEGREES)                                           │
│ │        │ Focused 94% │                  │ │ Yaw:   -0.8°  [──────────|──────────]                         │
│ │       ╔╧═════════════╧╗                 │ │ Pitch: +1.2°  [──────────|──────────]                         │
│ │       ║   (·)   (·)   ║ ◄─ Eye Dots     │ │ Roll:  -0.4°  [──────────|──────────]                         │
│ │       ║        ▼      ║                 │ ├───────────────────────────────────────────────────────────────┤
│ │       ╚═══════════════╝ ◄─ Face Box     │ │ SESSION SUMMARY                                      00:04:12 │
│ └─────────────────────────────────────────┘ │       ╭──────╮        ● Focused:    88.5%                     │
│ [▶ Start Camera] [📷 Snapshot] [Cam Select] │      │  89%  │        ● Distracted:  8.2%                     │
│                                             │       ╰──────╯        ● Drowsy:      3.3%                     │
└─────────────────────────────────────────────┴───────────────────────────────────────────────────────────────┘
```

---

## ✨ Core Features & Technical Highlights

### 1. 👁️ MediaPipe 478-Point Facial Landmark & Gaze Engine
* **478 3D Fiducials**: Runs Google MediaPipe's `face_landmarker.task` with dense mesh extraction.
* **Eye Aspect Ratio (EAR)**: Computes Euclidean geometric ratio across 6 contour landmarks per eye:
  $$\text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \times ||p_1 - p_4||}$$
* **Multi-Modal Eye Closure & Drowsiness**:
  * Evaluates both geometric **EAR ($< 0.23$)** and neural blendshapes (`eyeBlinkLeft`, `eyeBlinkRight` $> 0.45$).
  * Triggers **`Drowsy`** (red bounding box & audio alarm) when eyes remain closed $\ge 7$ frames ($\approx 0.45$s) or upon forward head slumping/nodding ($\text{Pitch} < -16^\circ$ with eyes closed $\ge 4$ frames).
  * Smooth step-down frame decay (`max(0, count - 2)`) prevents camera exposure jitter from resetting the alert.
* **3D Head Pose via Perspective-n-Point (`solvePnP` & `RQDecomp3x3`)**:
  * Projects 6 canonical 3D facial coordinates (Nose tip, Chin, Eye outer corners, Mouth corners) into the camera's intrinsic matrix.
  * Decomposes the rotation matrix into normalized Euler angles: **Yaw** (side-to-side), **Pitch** (up/down), and **Roll** (tilt).
* **Gaze Direction Classification**:
  * Real-time iris vector and blendshape tracking flags **Center**, **Left**, **Right**, **Up**, and **Down** gaze states.

### 2. 📱 YOLO11 Mobile Phone Violation Detection
* **Object Detection (`yolo11n.pt`)**: Real-time detection of smartphones and tablets.
* **Temporal Streak Confirmation**: Requires $\ge 2$ consecutive frames of phone visibility before triggering critical alarms, eliminating false positives from hands or desk items.
* **Visual Bounding Boxes**: Draws bright red brackets around detected phones with solid banner: `PHONE DETECTED {conf}%`.

### 3. 🧘 Attention State Machine
| State | Badge Color | Triggers | Visual / Audio Output |
| :--- | :--- | :--- | :--- |
| **`Focused`** | 🟩 **Green** (`#10b981`) | Upright posture ($|\text{Yaw}| \le 25^\circ$, $-24^\circ \le \text{Pitch} \le 22^\circ$), forward gaze, eyes open | Solid green face box, `Focused {conf}%` badge, target counter |
| **`Distracted`**| 🟧 **Amber** (`#f59e0b`) | Head turned to side ($|\text{Yaw}| > 25^\circ$), gaze averted left/right, looking up/down, or mobile phone present | Amber box, `Distracted {conf}%` badge, violation description |
| **`Drowsy`** | 🟥 **Crimson** (`#ef4444`)| Eyes closed $\ge 7$ frames, head nodding off on desk, or sleeping posture | Red face box, `Drowsy {conf}%` badge, 440 Hz beep, spoken warning |

### 4. 📊 Cumulative Session Analytics & Dynamic Donut Chart
* **Live Donut Chart**: Rendered on HTML5 Canvas displaying the exact proportion of time spent in `Focused`, `Distracted`, and `Drowsy` states.
* **Instant Session Reset**: One-click **↺ Reset Session** button synchronizes backend metrics, zeroes the elapsed timer, and restores 100% Focused baseline with built-in warm-up smoothing.
* **Snapshot Export**: Capture timestamped high-resolution `.png` audit snapshots directly from the live feed.

### 5. 🌐 Multi-Camera Sentinel Network (Fleet Monitoring)
* **Phone Camera Integration**: Click **📱 Connect Phone Cam** to generate a dynamic QR code. Scan it with any iPhone, Android, or tablet on your local Wi-Fi to turn mobile devices into remote classroom sentinel cameras.
* **Classroom Grid View**: Monitor multiple feeds simultaneously with aggregate metrics (active students, phones detected, sleeping/slumping students, and overall engagement index).

---

## 🛠️ Architecture & Data Flow

```mermaid
flowchart TD
    A[Webcam / Phone Video Stream] --> B[Offscreen Canvas 640x480]
    B -->|WebSocket Binary JPEG Pacing| C[FastAPI Server :5050]
    
    subgraph Vision Pipeline
        C --> D[YOLO11 Object Detector - Phone Detection]
        C --> E[MediaPipe FaceLandmarker - 478 Mesh Points]
        C --> F[YOLO11-Pose - Skeletal Keypoints]
        
        E --> G[EAR Calculation & Eye Blendshapes]
        E --> H[solvePnP & RQDecomp3x3 - Yaw/Pitch/Roll]
        E --> I[Iris Vectors - Gaze Direction]
    end
    
    Vision Pipeline --> J[Attention State Classifier]
    J --> K[Session State Accumulator & Donut Engine]
    J --> L[Annotated Frame Encoder]
    
    L -->|WebSocket JSON Telemetry + JPEG Frame| M[AttentionAI Dashboard UI]
    K -->|Donut % & Session Timer| M
```

---

## ⚡ Quick Start

### 1. Prerequisites & Environment Setup
* Python 3.10, 3.11, or 3.12 recommended.
* Virtual environment (optional but recommended):

```bash
git clone https://github.com/your-username/campus-monitor.git
cd campus-monitor
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note for Windows Users**: The repository pins `numpy<2.0.0` (e.g. `numpy==1.26.4`) to maintain full binary C-extension compatibility with TensorFlow and MediaPipe.

### 3. Start the Server
Run the convenience root launcher:
```bash
python run.py
```

The server automatically binds to **port 5050** (or scans for 8080/8001 if occupied):
```
=================================================================
🎓 CAMPUSGUARD & ATTENTION-AI FOCUS MONITOR
👉 Local Dashboard:   http://localhost:5050
👉 Network / Phone:   http://192.168.1.15:5050
=================================================================
```

### 4. Open the Live Dashboard
1. Open your browser and navigate to: [**`http://localhost:5050`**](http://localhost:5050).
2. Click **▶ Start Camera** and grant webcam permissions when prompted.
3. Test your focus:
   * **Looking forward**: Face box turns **green** (`Focused 94%`).
   * **Looking to the side**: Turns **amber** (`Distracted`).
   * **Closing your eyes for $\approx 1$s**: Turns **red** (`Drowsy`).
   * **Holding up your phone**: Flags **red** (`PHONE DETECTED`).

---

## 📂 Project Structure

```
campus-monitor/
├── campus-monitor/
│   ├── models/
│   │   └── face_landmarker.task     # MediaPipe 478-point 3D face mesh bundle
│   ├── static/
│   │   ├── dashboard.html           # AttentionAI 2-panel live focus dashboard
│   │   └── camera.html              # Mobile smartphone streaming client
│   ├── config.json                  # Detection thresholds and room settings
│   ├── server.py                    # FastAPI server, WebSockets & vision pipeline
│   ├── yolo11n.pt                   # YOLO11 object detection weights
│   ├── yolo11n-pose.pt              # YOLO11 skeletal pose keypoint weights
│   └── requirements.txt             # Sub-package requirements
├── run.py                           # Root launcher with auto-port allocation
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## 🔌 API & WebSocket Reference

### HTTP Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the main AttentionAI Focus Tracker dashboard |
| `GET` | `/cam/{rid}` | Serves the mobile camera streaming client for camera ID `{rid}` |
| `GET` | `/api/config` | Returns server LAN IP, active port, and room configuration |
| `POST` | `/api/new-camera` | Dynamically provisions a new classroom camera ID for phone pairing |
| `POST` | `/api/reset-session/{rid}`| Resets timer, state accumulators, and donut chart for room `{rid}` |
| `POST` | `/api/rooms/{rid}/rename` | Renames a camera room (e.g., "Hall A", "Desk 1") |

### WebSocket Protocol
* `ws://localhost:5050/ws/camera/{rid}`: Full-duplex binary frame ingestion and real-time telemetry response channel with ping-pong pacing.
* `ws://localhost:5050/ws/dashboard`: Broadcast channel for classroom fleet multi-camera grid and incident alerts.

---

## 🎨 Themes & Customization

The dashboard includes a theme switcher located in the top-right toolbar:
* **Dark Theme (Default)**: Deep obsidian `#0a0c10` with high-contrast emerald, amber, and crimson accents.
* **Warm Beige & Jet Black**: Elegant `#f7f4ec` creamy beige background with pitch-black typography and crisp borders for high-ambient-light environments.

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE). Built with OpenCV, MediaPipe, Ultralytics YOLO11, and FastAPI.
