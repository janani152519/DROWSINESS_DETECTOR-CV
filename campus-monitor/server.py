"""CampusGuard & AttentionAI Focus Tracker Engine.
- Real-time Attention Tracking: MediaPipe Face Landmark Mesh (478 pts), EAR (Eye Aspect Ratio),
  Blink frequency (Blinks/min), Gaze Direction, and 3D Head Pose (solvePnP: Yaw, Pitch, Roll).
- Object & Unusual Behavior Detection: YOLO11 for Mobile Phone violation detection & YOLO11-pose
  for posture, slumping, and head-down tracking.
- Session Analytics: Real-time temporal distribution (Focused %, Distracted %, Drowsy %) with timer.
- Real-time WebSocket streaming: Sub-second latency for live video, overlays, and telemetry.
- Multi-camera classroom support: Ingest from laptop webcam and remote mobile phones via QR code.
"""
import asyncio
import base64
import csv
import datetime
import io
import json
import os
import socket
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path

# Suppress noisy library logs & cache directory warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BASE = Path(__file__).parent
CFG_FILE = BASE / "config.json"
KEY = os.getenv("DASH_KEY", "")
SERVER_PORT = int(os.getenv("PORT", "5050"))

# Default Configuration
cfg = {
    "watch": {
        "cell phone": "Mobile phone"
    },
    "head_down_seconds": 5,
    "eyes_closed_seconds": 1.5,
    "phone_alert_frames": 2,
    "alert_cooldown": 10,
    "rooms": {}
}

if CFG_FILE.exists():
    try:
        cfg.update(json.loads(CFG_FILE.read_text(encoding="utf-8")))
    except Exception as e:
        print(f"Error loading config: {e}")

def save_cfg():
    CFG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

def get_lan_ip():
    """Discover host machine local Wi-Fi / LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# Initialize AI Models
print("[+] Loading YOLO11 models & MediaPipe Face Landmarker...")
pose_model = YOLO(BASE / "yolo11n-pose.pt")
det_model = YOLO(BASE / "yolo11n.pt")
pose_lock = threading.Lock()
det_lock = threading.Lock()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# MediaPipe Face Landmarker
face_task_path = BASE / "models" / "face_landmarker.task"
if not face_task_path.exists():
    # check repo root
    alt_path = BASE.parent / "models" / "face_landmarker.task"
    if alt_path.exists():
        face_task_path = alt_path

base_options = mp_python.BaseOptions(model_asset_path=str(face_task_path))
face_options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1
)
face_detector = mp_vision.FaceLandmarker.create_from_options(face_options)
face_lock = threading.Lock()
print("[✓] AI Models & Face Landmarker loaded successfully.")

# Facial Landmark Constants (478-point mesh)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# 3D Reference Model Points for Head Pose solvePnP
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
    (0.0, 110.0, -30.0),         # Chin (landmark 152)
    (-60.0, -50.0, -40.0),       # Left eye outer corner (landmark 33)
    (60.0, -50.0, -40.0),        # Right eye outer corner (landmark 263)
    (-40.0, 70.0, -35.0),        # Left mouth corner (landmark 61)
    (40.0, 70.0, -35.0)          # Right mouth corner (landmark 291)
], dtype=np.float64)

POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

def calculate_ear(landmarks_2d, eye_indices, w, h):
    """Compute Eye Aspect Ratio (EAR) using Euclidean distance."""
    pts = np.array([[landmarks_2d[idx].x * w, landmarks_2d[idx].y * h] for idx in eye_indices], dtype=np.float64)
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    ear = (A + B) / (2.0 * (C + 1e-6))
    return float(ear), pts

def compute_head_pose(landmarks_2d, w, h):
    """Solve Perspective-n-Point to determine head Yaw, Pitch, Roll in degrees."""
    image_points = np.array([
        (landmarks_2d[idx].x * w, landmarks_2d[idx].y * h)
        for idx in POSE_LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = float(w)
    center = (float(w) / 2.0, float(h) / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    if roll > 90.0:
        roll -= 180.0
    elif roll < -90.0:
        roll += 180.0

    return round(yaw, 1), round(pitch, 1), round(roll, 1)

def compute_gaze(blendshapes, yaw_deg, pitch_deg):
    """Classify gaze direction into: center, down, up, left, right."""
    bs_map = {b.category_name: b.score for b in blendshapes} if blendshapes else {}
    look_down = max(bs_map.get("eyeLookDownLeft", 0.0), bs_map.get("eyeLookDownRight", 0.0))
    look_up = max(bs_map.get("eyeLookUpLeft", 0.0), bs_map.get("eyeLookUpRight", 0.0))
    look_in_l = bs_map.get("eyeLookInLeft", 0.0)
    look_out_l = bs_map.get("eyeLookOutLeft", 0.0)
    look_in_r = bs_map.get("eyeLookInRight", 0.0)
    look_out_r = bs_map.get("eyeLookOutRight", 0.0)

    # Looking to side (head yaw or eye gaze)
    if yaw_deg < -25.0 or look_out_l > 0.52 or look_in_r > 0.52:
        return "left"
    elif yaw_deg > 25.0 or look_in_l > 0.52 or look_out_r > 0.52:
        return "right"
    # Looking up away from camera
    elif pitch_deg > 22.0 or look_up > 0.60:
        return "up"
    # Looking down away from camera
    elif pitch_deg < -24.0 or look_down > 0.75:
        return "down"
    return "center"

class SimpleTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}

    def update(self, detected_boxes):
        now = time.time()
        self.tracks = {tid: tr for tid, tr in self.tracks.items() if now - tr["last_seen"] < 3.0}

        assigned = []
        for box in detected_boxes:
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            best_tid, best_dist = None, 110.0
            for tid, tr in self.tracks.items():
                if tid in assigned:
                    continue
                tcx, tcy = tr["centroid"]
                dist = np.hypot(cx - tcx, cy - tcy)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                self.tracks[best_tid] = {"box": box, "centroid": (cx, cy), "last_seen": now}
                assigned.append(best_tid)

        res_ids = []
        for box in detected_boxes:
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            found_tid = None
            for tid, tr in self.tracks.items():
                if tr["centroid"] == (cx, cy):
                    found_tid = tid
                    break
            if found_tid is None:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {"box": box, "centroid": (cx, cy), "last_seen": now}
                res_ids.append(new_id)
            else:
                res_ids.append(found_tid)

        return res_ids

class Room:
    def __init__(self, rid):
        self.id = rid
        self.name = cfg["rooms"].get(rid, f"Classroom {rid}")
        self.tracker = SimpleTracker()
        self.online = False
        self.last_seen = 0
        self.jpeg = b""

        # AttentionAI Diagnostics & Session tracking
        self.session_start = time.time()
        self.last_state_tick = time.time()
        self.session_states = {"Focused": 0.0, "Distracted": 0.0, "Drowsy": 0.0}
        self.current_state = "Focused"
        self.current_conf = 85
        self.current_reason = ""
        self.ear = 0.280
        self.blinks = deque(maxlen=200)
        self.blinks_per_min = 18
        self.gaze = "center"
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        # Temporal counters
        self.ear_low_frames = 0
        self.is_blink_state = False
        self.eyes_open = True

        self.stats = dict(
            people=0, active=0, head_down=0, sleeping=0, unusual=0, phones=0,
            engagement=100, fps=0.0, eyes_open=True, eyes_closed_sec=0.0,
            drowsy_reason="", phone=False,
            state="Focused", confidence=85, ear=0.28, blinks_per_min=18,
            gaze="center", yaw=0.0, pitch=0.0, roll=0.0,
            session_focused_pct=100.0, session_distracted_pct=0.0,
            session_drowsy_pct=0.0, session_elapsed=0
        )

        self.down_since = {}
        self.eyes_closed_since = {}
        self.nod_since = {}
        self.sleep_alerted = set()
        self.phone_streak = 0
        self.last_alert = {}
        self.last_hist = 0
        self.history = deque(maxlen=120)
        self.frame_count = 0
        self.fps_start = time.time()
        self.fps = 0.0

    def reset_session(self):
        """Reset session metrics and timer."""
        now = time.time()
        self.session_start = now
        self.last_state_tick = now
        self.session_states = {"Focused": 0.0, "Distracted": 0.0, "Drowsy": 0.0}
        self.blinks.clear()
        self.ear_low_frames = 0
        self.current_state = "Focused"
        self.current_conf = 92
        self.current_reason = "Attentive & engaged"
        self.stats["state"] = "Focused"
        self.stats["confidence"] = 92
        self.stats["reason"] = "Attentive & engaged"
        self.stats["session_focused_pct"] = 100.0
        self.stats["session_distracted_pct"] = 0.0
        self.stats["session_drowsy_pct"] = 0.0
        self.stats["session_elapsed"] = 0

rooms = {}
alerts = deque(maxlen=300)
dashboards = set()

app = FastAPI(title="Campus Multi-Camera AI Monitor")

def ok(k):
    return (not KEY) or k == KEY

def raise_alert(room, kind, msg, severity="warning"):
    now = time.time()
    cooldown = cfg.get("alert_cooldown", 10)
    if now - room.last_alert.get(kind, 0) < cooldown:
        return
    room.last_alert[kind] = now
    alert_obj = dict(
        id=int(now * 1000),
        t=now,
        room_id=room.id,
        room=room.name,
        kind=kind,
        msg=msg,
        severity=severity
    )
    alerts.appendleft(alert_obj)

def analyze(room, frame):
    now = time.time()
    room.frame_count += 1
    if now - room.fps_start >= 1.0:
        room.fps = room.frame_count / (now - room.fps_start)
        room.frame_count = 0
        room.fps_start = now

    h, w = frame.shape[:2]

    # 1. YOLO-Pose for Multi-student Body & Unusual Behavior Tracking
    people = []
    try:
            with pose_lock:
                pr = pose_model(frame, conf=0.30, verbose=False)[0]

            if pr.boxes is not None and len(pr.boxes):
                boxes = pr.boxes.xyxy.cpu().numpy()
                kps = pr.keypoints.data.cpu().numpy() if pr.keypoints is not None else None
                track_ids = room.tracker.update(boxes)

                for i, (tid, b) in enumerate(zip(track_ids, boxes)):
                    down = False
                    unusual = False
                    unusual_reason = ""
                    head_nodding = False

                    if kps is not None and i < len(kps):
                        kp = kps[i]
                        nose = kp[0]
                        lear, rear = kp[3], kp[4]
                        ls, rs = kp[5], kp[6]
                        lw, rw = kp[9], kp[10]

                        if ls[2] > 0.3 and rs[2] > 0.3:
                            sw = abs(ls[0] - rs[0]) + 1e-5
                            sm_y = (ls[1] + rs[1]) / 2.0

                            if nose[2] < 0.25 or nose[1] > (sm_y - 0.10 * sw):
                                down = True

                            nose_shoulder_dist = sm_y - nose[1]
                            if nose[2] > 0.3 and nose_shoulder_dist < 0.15 * sw and not down:
                                head_nodding = True

                            if lw[2] > 0.3 and lear[2] > 0.3 and np.hypot(lw[0] - lear[0], lw[1] - lear[1]) < 0.40 * sw:
                                unusual = True
                                unusual_reason = "Hand-to-ear"
                            if rw[2] > 0.3 and rear[2] > 0.3 and np.hypot(rw[0] - rear[0], rw[1] - rear[1]) < 0.40 * sw:
                                unusual = True
                                unusual_reason = "Hand-to-ear"

                            shoulder_tilt = abs(ls[1] - rs[1]) / sw
                            if shoulder_tilt > 0.55:
                                unusual = True
                                unusual_reason = "Slumping"

                    people.append(dict(
                        id=tid,
                        box=b,
                        down=down,
                        unusual=unusual,
                        unusual_reason=unusual_reason,
                        head_nodding=head_nodding,
                        phone=False,
                        sleeping=False
                    ))
    except Exception as pe:
        pass

    # 2. YOLO11 Detection for Mobile Phone Violations
    with det_lock:
        dr = det_model(frame, conf=0.20, verbose=False)[0]

    watched = []
    has_phone_detected = False
    for bx in dr.boxes:
        cname = det_model.names[int(bx.cls)]
        if cname in cfg["watch"] or cname == "cell phone":
            label = cfg["watch"].get(cname, "Mobile phone")
            box_coords = bx.xyxy[0].cpu().numpy()
            conf = float(bx.conf[0])
            watched.append((label, box_coords, conf, cname))
            has_phone_detected = True

    if has_phone_detected:
        room.phone_streak += 1
    else:
        room.phone_streak = max(0, room.phone_streak - 1)

    confirmed_phone = room.phone_streak >= cfg.get("phone_alert_frames", 2)
    if confirmed_phone:
        for p in people:
            p["phone"] = True
        raise_alert(room, "Phone Detected", f"Mobile phone in use in {room.name}", severity="critical")

    # 3. MediaPipe FaceLandmarker for AttentionAI Focus Diagnostics
    # Compute: EAR, Blinks/min, Gaze Direction, Head Pose (Yaw, Pitch, Roll)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    face_box = None
    left_eye_pts = None
    right_eye_pts = None
    face_detected = False

    with face_lock:
        try:
            lm_result = face_detector.detect(mp_img)
            if lm_result.face_landmarks and len(lm_result.face_landmarks) > 0:
                face_detected = True
                landmarks = lm_result.face_landmarks[0]
                blendshapes = lm_result.face_blendshapes[0] if lm_result.face_blendshapes else []

                # A. Eye Aspect Ratio (EAR) & Blink Blendshapes
                ear_l, left_eye_pts = calculate_ear(landmarks, LEFT_EYE_INDICES, w, h)
                ear_r, right_eye_pts = calculate_ear(landmarks, RIGHT_EYE_INDICES, w, h)
                ear_avg = (ear_l + ear_r) / 2.0
                room.ear = round(ear_avg, 3)

                bs_map = {b.category_name: b.score for b in blendshapes} if blendshapes else {}
                blink_score = max(bs_map.get("eyeBlinkLeft", 0.0), bs_map.get("eyeBlinkRight", 0.0))

                # Multi-modal eye closure check (EAR < 0.23 or MediaPipe blink > 0.45)
                eyes_closed_now = (ear_avg < 0.23) or (blink_score > 0.45)
                room.eyes_open = (not eyes_closed_now) and (room.ear_low_frames < 3)

                # B. Blink Tracking & Blinks/Min
                if eyes_closed_now:
                    room.ear_low_frames += 1
                else:
                    if 1 <= room.ear_low_frames <= 5:
                        room.blinks.append(now)
                    # Graceful step-down decay to protect against single-frame camera jitter
                    room.ear_low_frames = max(0, room.ear_low_frames - 2)

                cutoff = now - 60.0
                room.blinks = deque([t for t in room.blinks if t >= cutoff], maxlen=200)
                elapsed_sec = max(1.0, now - room.session_start)
                if elapsed_sec < 55.0:
                    room.blinks_per_min = int(round((len(room.blinks) / elapsed_sec) * 60.0))
                else:
                    room.blinks_per_min = len(room.blinks)

                # C. Head Pose: Yaw, Pitch, Roll (solvePnP)
                yaw, pitch, roll = compute_head_pose(landmarks, w, h)
                room.yaw = yaw
                room.pitch = pitch
                room.roll = roll

                # D. Gaze Direction
                gaze = compute_gaze(blendshapes, yaw, pitch)
                room.gaze = gaze

                # E. Face Bounding Box
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                pad_x = (max(xs) - min(xs)) * 0.12
                pad_y = (max(ys) - min(ys)) * 0.15
                face_box = (
                    max(0, int(min(xs) - pad_x)),
                    max(0, int(min(ys) - pad_y)),
                    min(w, int(max(xs) + pad_x)),
                    min(h, int(max(ys) + pad_y))
                )
        except Exception as e:
            # Fallback if detection encounters transient exception
            pass

    # Fallback face box detection if landmarks were not detected in current frame
    if face_box is None and len(people) > 0 and people[0].get("box") is not None:
        p_box = people[0]["box"]
        pw = p_box[2] - p_box[0]
        ph = p_box[3] - p_box[1]
        head_cx = (p_box[0] + p_box[2]) / 2.0
        head_cy = p_box[1] + ph * 0.16
        half_s = max(pw * 0.28, ph * 0.15)
        face_box = (
            max(0, int(head_cx - half_s)),
            max(0, int(head_cy - half_s * 1.1)),
            min(w, int(head_cx + half_s)),
            min(h, int(head_cy + half_s * 1.1))
        )

    if face_box is None:
        try:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.18, minNeighbors=3, minSize=(60, 60))
            if len(detected_faces) > 0:
                dfx, dfy, dfw, dfh = detected_faces[0]
                face_box = (dfx, dfy, dfx + dfw, dfy + dfh)
        except Exception:
            pass

    # 4. AttentionAI Focus Classification & Confidence Logic
    dt = min(1.0, max(0.001, now - room.last_state_tick))
    room.last_state_tick = now

    state = "Focused"
    conf = 94
    reason = "Looking at camera & attentive"

    is_drowsy = (room.ear_low_frames >= 7) or (room.pitch < -16.0 and room.ear_low_frames >= 4)

    if confirmed_phone or has_phone_detected:
        state = "Distracted"
        conf = 95
        reason = "Mobile Phone Detected!"
    elif is_drowsy:
        state = "Drowsy"
        conf = min(98, 80 + int(room.ear_low_frames * 1.5))
        reason = "Eyes closed / Drowsiness detected"
    elif abs(room.yaw) > 25.0:
        state = "Distracted"
        conf = min(96, 75 + int(abs(room.yaw) * 0.6))
        reason = "Head turned to the side"
    elif room.gaze in ["left", "right"]:
        state = "Distracted"
        conf = 85
        reason = f"Looking to the {room.gaze}"
    elif room.pitch > 22.0:
        state = "Distracted"
        conf = min(92, 70 + int(abs(room.pitch) * 0.6))
        reason = "Looking up away from camera"
    elif room.pitch < -24.0:
        state = "Distracted"
        conf = min(92, 70 + int(abs(room.pitch) * 0.6))
        reason = "Looking down away from camera"
    elif not face_detected and len(people) > 0 and people[0]["down"]:
        state = "Drowsy"
        conf = 85
        reason = "Head slumped down"
    elif not face_detected and len(people) > 0 and people[0]["unusual"]:
        state = "Distracted"
        conf = 76
        reason = people[0]["unusual_reason"]
    else:
        state = "Focused"
        conf = 94
        reason = "Looking at camera & attentive"

    room.current_state = state
    room.current_conf = conf
    room.current_reason = reason

    # Accumulate Session Statistics
    room.session_states[state] = room.session_states.get(state, 0.0) + dt
    total_session = sum(room.session_states.values())
    
    # Graceful warm-up seed (2.0s focused) so fresh or reset sessions don't spike to 100% distracted on a single frame
    warmup = max(0.0, 2.0 - total_session)
    eff_f = room.session_states.get("Focused", 0.0) + warmup
    eff_di = room.session_states.get("Distracted", 0.0)
    eff_dr = room.session_states.get("Drowsy", 0.0)
    eff_tot = eff_f + eff_di + eff_dr
    if eff_tot > 0:
        focused_pct = (eff_f / eff_tot) * 100.0
        distracted_pct = (eff_di / eff_tot) * 100.0
        drowsy_pct = (eff_dr / eff_tot) * 100.0
    else:
        focused_pct, distracted_pct, drowsy_pct = 100.0, 0.0, 0.0

    session_elapsed = int(now - room.session_start)

    # 5. Visual Overlays on Live Feed (Exact AttentionAI Layout)
    # A. Draw Face Bounding Box & Eye Landmarks
    if face_box:
        fx1, fy1, fx2, fy2 = face_box
        # Distinct state colors in BGR:
        # Phone violation = Bright Red (30, 30, 240)
        # Focused = Vibrant Green (50, 205, 50)
        # Distracted = Golden Amber (0, 190, 255)
        # Drowsy = Crimson Red (50, 50, 240)
        if confirmed_phone or has_phone_detected:
            color = (30, 30, 240)
            badge_text = f"PHONE DETECTED {conf}%"
        elif state == "Focused":
            color = (50, 205, 50)
            badge_text = f"Focused {conf}%"
        elif state == "Distracted":
            color = (0, 190, 255)
            badge_text = f"Distracted {conf}%"
        else:
            color = (50, 50, 240)
            badge_text = f"Drowsy {conf}%"

        # Sleek Yellow/Colored Face Box with thick corners
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
        corner_len = min(28, int((fx2 - fx1) * 0.22))
        # Top-left corner
        cv2.line(frame, (fx1, fy1), (fx1 + corner_len, fy1), color, 4)
        cv2.line(frame, (fx1, fy1), (fx1, fy1 + corner_len), color, 4)
        # Top-right corner
        cv2.line(frame, (fx2, fy1), (fx2 - corner_len, fy1), color, 4)
        cv2.line(frame, (fx2, fy1), (fx2, fy1 + corner_len), color, 4)
        # Bottom-left corner
        cv2.line(frame, (fx1, fy2), (fx1 + corner_len, fy2), color, 4)
        cv2.line(frame, (fx1, fy2), (fx1, fy2 - corner_len), color, 4)
        # Bottom-right corner
        cv2.line(frame, (fx2, fy2), (fx2 - corner_len, fy2), color, 4)
        cv2.line(frame, (fx2, fy2), (fx2, fy2 - corner_len), color, 4)

        # Eye Landmark dots (Cyan/Teal dots on eyes)
        if left_eye_pts is not None:
            for pt in left_eye_pts:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255, 220, 0), -1)
        if right_eye_pts is not None:
            for pt in right_eye_pts:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255, 220, 0), -1)

        # State + Confidence Badge directly above head (e.g. "Distracted 73%" or "PHONE DETECTED 95%")
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.65
        thickness = 1
        (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, thickness)

        bw = tw + 20
        bh = th + 14
        bx1 = fx1
        by1 = max(8, fy1 - bh - 6)
        bx2 = bx1 + bw
        by2 = by1 + bh

        # Solid badge background
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
        text_color = (15, 18, 22) if (state == "Distracted" and not has_phone_detected) else (255, 255, 255)
        cv2.putText(frame, badge_text, (bx1 + 10, by2 - 6), font, font_scale, text_color, 2, cv2.LINE_AA)

    # B. Phone Detection Boxes
    for label, coords, conf_val, _ in watched:
        px1, py1, px2, py2 = map(int, coords)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 245), 3)
        p_badge = f"PHONE {int(conf_val*100)}%"
        (ptw, pth), _ = cv2.getTextSize(p_badge, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(frame, (px1, max(0, py1 - pth - 10)), (px1 + ptw + 12, py1), (0, 0, 245), -1)
        cv2.putText(frame, p_badge, (px1 + 6, py1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # C. Left Real-time Diagnostic HUD Overlay (Dark semi-transparent panel matching AttentionAI)
    hud_w = 175
    hud_h = 165
    hud_x = 12
    hud_y = 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (12, 14, 18), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (45, 52, 64), 1)

    hud_lines = [
        (f"FPS: {room.fps:.1f}", (140, 200, 255)),
        (f"EAR: {room.ear:.3f}", (255, 255, 255)),
        (f"Blinks: {room.blinks_per_min}/min", (255, 255, 255)),
        (f"Gaze: {room.gaze}", (255, 255, 255)),
        (f"Yaw: {room.yaw:+.1f} deg", (200, 220, 240)),
        (f"Pitch: {room.pitch:+.1f} deg", (200, 220, 240)),
        (f"Roll: {room.roll:+.1f} deg", (200, 220, 240))
    ]

    ty = hud_y + 20
    for txt, col in hud_lines:
        cv2.putText(frame, txt, (hud_x + 10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        ty += 20

    # Classroom Fleet Aggregate Stats
    total_p = len(people)
    active_p = sum(1 for p in people if not p["down"] and not p["sleeping"])
    head_down_p = sum(1 for p in people if p["down"])
    sleeping_p = sum(1 for p in people if p["sleeping"] or state == "Drowsy")
    unusual_p = sum(1 for p in people if p["unusual"])
    phone_p = sum(1 for p in people if p["phone"]) + (1 if confirmed_phone else 0)

    room.stats = dict(
        people=total_p,
        active=active_p,
        head_down=head_down_p,
        sleeping=sleeping_p,
        unusual=unusual_p,
        phones=phone_p,
        engagement=int(focused_pct),
        fps=round(room.fps, 1),
        eyes_open=getattr(room, "eyes_open", True),
        eyes_closed_sec=round(room.ear_low_frames / max(1.0, room.fps), 1),
        drowsy_reason=reason if state == "Drowsy" else "",
        phone=confirmed_phone,
        state=state,
        confidence=conf,
        reason=reason,
        ear=round(room.ear, 3),
        blinks_per_min=room.blinks_per_min,
        gaze=room.gaze,
        yaw=room.yaw,
        pitch=room.pitch,
        roll=room.roll,
        session_focused_pct=round(focused_pct, 1),
        session_distracted_pct=round(distracted_pct, 1),
        session_drowsy_pct=round(drowsy_pct, 1),
        session_elapsed=session_elapsed
    )

    if now - room.last_hist >= 4.0:
        room.last_hist = now
        room.history.append([
            int(now),
            room.stats["people"],
            room.stats["active"],
            room.stats["phones"],
            room.stats["engagement"]
        ])

    preview_w = 640
    preview_h = int(h * (preview_w / w)) if w > 0 else 480
    small = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
    room.jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()

@app.websocket("/ws/camera/{rid}")
async def camera_ws(ws: WebSocket, rid: str):
    await ws.accept()
    if rid not in rooms:
        rooms[rid] = Room(rid)
    room = rooms[rid]
    room.online = True
    loop = asyncio.get_running_loop()

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("text"):
                try:
                    payload = json.loads(msg["text"])
                    if payload.get("action") == "reset_session":
                        room.reset_session()
                        await ws.send_text(json.dumps({
                            "status": "ok",
                            "action": "session_reset",
                            "state": "Focused",
                            "confidence": 92,
                            "reason": "Attentive & engaged",
                            "session_focused_pct": 100.0,
                            "session_distracted_pct": 0.0,
                            "session_drowsy_pct": 0.0,
                            "session_elapsed": 0,
                            "ear": room.stats["ear"],
                            "blinks_per_min": room.stats["blinks_per_min"],
                            "gaze": room.stats["gaze"],
                            "yaw": room.stats["yaw"],
                            "pitch": room.stats["pitch"],
                            "roll": room.stats["roll"],
                            "eyes_open": True,
                            "eyes_closed_sec": 0.0,
                            "drowsy": False,
                            "phone": False,
                            "people": room.stats["people"],
                            "active": room.stats["active"],
                            "engagement": 100,
                            "fps": room.stats["fps"]
                        }))
                        continue
                    new_name = payload.get("name")
                    if new_name:
                        room.name = new_name[:50]
                        cfg["rooms"][rid] = room.name
                        save_cfg()
                except Exception:
                    pass
                continue

            raw_bytes = msg.get("bytes")
            if not raw_bytes:
                continue

            frame = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            await loop.run_in_executor(None, analyze, room, frame)
            room.last_seen = time.time()

            # Immediate full AttentionAI telemetry response
            await ws.send_text(json.dumps({
                "status": "ok",
                "frame": base64.b64encode(room.jpeg).decode() if room.jpeg else "",
                "state": room.stats["state"],
                "confidence": room.stats["confidence"],
                "reason": room.stats["reason"],
                "ear": room.stats["ear"],
                "blinks_per_min": room.stats["blinks_per_min"],
                "gaze": room.stats["gaze"],
                "yaw": room.stats["yaw"],
                "pitch": room.stats["pitch"],
                "roll": room.stats["roll"],
                "session_focused_pct": room.stats["session_focused_pct"],
                "session_distracted_pct": room.stats["session_distracted_pct"],
                "session_drowsy_pct": room.stats["session_drowsy_pct"],
                "session_elapsed": room.stats["session_elapsed"],
                "eyes_open": room.stats["eyes_open"],
                "eyes_closed_sec": room.stats["eyes_closed_sec"],
                "drowsy": room.stats["state"] == "Drowsy",
                "phone": room.stats["phone"],
                "people": room.stats["people"],
                "active": room.stats["active"],
                "engagement": room.stats["engagement"],
                "fps": room.stats["fps"]
            }))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        room.online = False

@app.websocket("/ws/dashboard")
async def dashboard_ws(ws: WebSocket, k: str = ""):
    if not ok(k):
        await ws.close(code=1008)
        return
    await ws.accept()
    dashboards.add(ws)
    try:
        while True:
            await ws.receive_text()
    except Exception:
        pass
    finally:
        dashboards.discard(ws)

async def broadcaster():
    """Broadcast global room state, camera feeds, and alert logs to all open dashboards."""
    while True:
        await asyncio.sleep(0.4)
        if not dashboards:
            continue

        now = time.time()
        cams = []
        for rid, r in list(rooms.items()):
            is_active = r.online and (now - r.last_seen < 4.0)
            cams.append({
                "id": rid,
                "name": r.name,
                "online": is_active,
                "stats": r.stats,
                "thumb": base64.b64encode(r.jpeg).decode() if (r.jpeg and is_active) else "",
                "last_seen": int(r.last_seen)
            })

        total_people = sum(c["stats"].get("people", 0) for c in cams if c["online"])
        total_active = sum(c["stats"].get("active", 0) for c in cams if c["online"])
        total_sleep = sum(c["stats"].get("sleeping", 0) for c in cams if c["online"])
        total_phones = sum(c["stats"].get("phones", 0) for c in cams if c["online"])

        active_cams = [c for c in cams if c["online"]]
        avg_engagement = int(
            sum(c["stats"].get("engagement", 100) for c in active_cams) / len(active_cams)
        ) if active_cams else 100

        data = {
            "summary": {
                "cameras_online": len(active_cams),
                "total_people": total_people,
                "active_students": total_active,
                "drowsy_students": total_sleep,
                "phone_violations": total_phones,
                "avg_engagement": avg_engagement
            },
            "cameras": cams,
            "alerts": list(alerts)[:25]
        }
        msg = json.dumps(data)
        dead = []
        for ws in list(dashboards):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            dashboards.discard(ws)

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(broadcaster())

# Static & Web Dashboard Endpoints
@app.get("/")
def get_dashboard():
    return FileResponse(BASE / "static" / "dashboard.html")

@app.get("/cam/{rid}")
def get_cam(rid: str):
    return FileResponse(BASE / "static" / "camera.html")

@app.get("/api/config")
def get_config(k: str = ""):
    if not ok(k):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    lan_ip = get_lan_ip()
    return {
        "cfg": cfg,
        "lan_ip": lan_ip,
        "port": SERVER_PORT,
        "server_url": f"http://{lan_ip}:{SERVER_PORT}"
    }

@app.post("/api/reset-session/{rid}")
def post_reset_session(rid: str, k: str = ""):
    if not ok(k):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if rid in rooms:
        rooms[rid].reset_session()
        return {
            "status": "ok",
            "message": f"Session for {rid} reset.",
            "session_focused_pct": 100.0,
            "session_distracted_pct": 0.0,
            "session_drowsy_pct": 0.0,
            "session_elapsed": 0
        }
    return {"status": "ok"}

@app.post("/api/new-camera")
def create_camera(k: str = ""):
    if not ok(k):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    rid = uuid.uuid4().hex[:6]
    room_name = f"Classroom {len(cfg['rooms']) + 1}"
    cfg["rooms"][rid] = room_name
    save_cfg()
    if rid not in rooms:
        rooms[rid] = Room(rid)
    return {"id": rid, "name": room_name}

@app.post("/api/rooms/{rid}/rename")
async def rename_room(rid: str, req: Request, k: str = ""):
    if not ok(k):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    data = await req.json()
    new_name = data.get("name", "").strip()
    if not new_name:
        return JSONResponse({"error": "Name required"}, status_code=400)
    cfg["rooms"][rid] = new_name
    save_cfg()
    if rid in rooms:
        rooms[rid].name = new_name
    return {"status": "ok", "name": new_name}

@app.delete("/api/rooms/{rid}")
def delete_room(rid: str, k: str = ""):
    if not ok(k):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if rid in cfg["rooms"]:
        del cfg["rooms"][rid]
        save_cfg()
    if rid in rooms:
        del rooms[rid]
    return {"status": "ok"}

@app.get("/api/reports/incidents.csv")
def export_incidents(k: str = ""):
    if not ok(k):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "Date Time", "Room ID", "Room Name", "Incident Type", "Severity", "Details"])
    for a in alerts:
        dt_str = datetime.datetime.fromtimestamp(a["t"]).strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([a["t"], dt_str, a.get("room_id", ""), a["room"], a["kind"], a["severity"], a["msg"]])
    output.seek(0)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=campus_incidents.csv"}
    )

if __name__ == "__main__":
    host_ip = "0.0.0.0"
    print(f"\n=======================================================")
    print(f"  CampusGuard AI & AttentionAI Focus Tracker")
    print(f"  Local Sentinel Dashboard: http://localhost:{SERVER_PORT}")
    print(f"  Local Network IP:          http://{get_lan_ip()}:{SERVER_PORT}")
    print(f"=======================================================\n")
    uvicorn.run(app, host=host_ip, port=SERVER_PORT, log_level="info")
