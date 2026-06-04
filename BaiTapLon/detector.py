import os
import cv2
import numpy as np
from datetime import datetime
import time

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

MODEL_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "yolov8m.pt"),
    os.path.join(os.path.dirname(__file__), "yolov8s.pt"),
    os.path.join(os.path.dirname(__file__), "yolov8n.pt")
]
MODEL_PATH = next((path for path in MODEL_CANDIDATES if os.path.exists(path)), os.path.join(os.path.dirname(__file__), "yolov8n.pt"))

DEFAULT_IMG_SIZE = 640

COLORS = {
    "chicken": (0, 185, 255),
    "duck": (255, 77, 77),
    "pig": (201, 76, 255)
}

# Controls how many frames are skipped between full detection passes.
# A value of 2 means full detection runs every other frame, while intermediate frames still update tracks.
DETECTION_INTERVAL = 2

# COCO Class IDs (YOLOv8 COCO dataset)
COCO_BIRD = 14  # bird -> map to duck/chicken
COCO_SHEEP = 18  # sheep -> pig
COCO_COW = 19    # cow -> pig
COCO_DOG = 16    # dog -> pig/chicken
COCO_CAT = 15    # cat -> chicken
COCO_HORSE = 17  # horse -> pig
COCO_ELEPHANT = 20  # elephant -> pig

DEFAULT_CONFIDENCE = 0.22
DEFAULT_IOU = 0.45
MIN_BBOX_AREA = 1700
MAX_BBOX_AREA = 450000

model = None
ORIGINAL_CLASSES = None
if ULTRALYTICS_AVAILABLE:
    try:
        model = YOLO(MODEL_PATH)
        if hasattr(model, "names"):
            names = model.names
            if isinstance(names, dict):
                ORIGINAL_CLASSES = {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, (list, tuple)):
                ORIGINAL_CLASSES = {idx: str(name) for idx, name in enumerate(names)}
    except Exception as e:
        print(f"Warning: could not load YOLO model from {MODEL_PATH}: {e}")
        model = None

if model is not None:
    print(f"YOLO model loaded from {MODEL_PATH}")
    print(f"Model classes: {ORIGINAL_CLASSES}")
elif not ULTRALYTICS_AVAILABLE:
    print("Ultralytics package is not available. Install ultralytics to run detection.")
else:
    print(f"YOLO model unavailable. Please verify the model file at {MODEL_PATH}.")

TRACKS = {}
NEXT_TRACK_ID = 0
TOTALS = {"chicken": 0, "duck": 0, "pig": 0}
MAX_TRACK_AGE = 28
ASSIGNMENT_DISTANCE = 240
TRACK_CONFIDENCE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.12
LABEL_CHANGE_MIN_VOTES = 3
LABEL_CHANGE_RATIO = 0.60


def _to_int_coords(xyxy):
    if hasattr(xyxy, 'cpu'):
        xyxy = xyxy.cpu().numpy()
    xyxy = np.array(xyxy).reshape(-1)
    return [int(float(v)) for v in xyxy[:4]]


def _to_float(value):
    if hasattr(value, 'cpu'):
        value = value.cpu().numpy()
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError("Cannot convert empty value to float")
    return float(arr.flatten()[0])


def _euclidean(a, b):
    return np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32))


def _calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calculate union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def _predict_position(track):
    """Predict next position based on velocity"""
    if 'velocity' not in track:
        return track['centroid']

    vx, vy = track['velocity']
    x, y = track['centroid']
    return (x + vx, y + vy)


def _match_tracks(detections):
    global TRACKS, NEXT_TRACK_ID, TOTALS

    assigned = set()
    updated_tracks = {}

    # Sort detections by confidence for better matching
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    for det in detections:
        best_id = None
        best_score = 0.0

        for track_id, track in TRACKS.items():
            if track_id in assigned:
                continue

            # Avoid aggressive cross-label matching unless object alignment is very strong
            iou_score = _calculate_iou(track['bbox'], det['bbox'])
            if track['label'] != det['label'] and iou_score < 0.55:
                continue

            # Calculate centroid distance score (normalized)
            centroid_dist = _euclidean(track['centroid'], det['centroid'])
            dist_score = max(0, 1.0 - centroid_dist / ASSIGNMENT_DISTANCE)

            # Predict position if velocity available
            predicted_pos = _predict_position(track)
            pred_dist = _euclidean(predicted_pos, det['centroid'])
            pred_score = max(0, 1.0 - pred_dist / (ASSIGNMENT_DISTANCE * 1.5))

            # Size consistency check (bbox area similarity)
            track_area = (track['bbox'][2] - track['bbox'][0]) * (track['bbox'][3] - track['bbox'][1])
            det_area = (det['bbox'][2] - det['bbox'][0]) * (det['bbox'][3] - det['bbox'][1])
            size_similarity = 1.0 - min(abs(track_area - det_area) / max(track_area, det_area), 1.0)

            # Combined score: prioritize same-label detections, IoU, then prediction, then distance and size
            label_penalty = 0.35 if track['label'] != det['label'] else 0.0
            combined_score = (iou_score * 0.5) + (pred_score * 0.25) + (dist_score * 0.15) + (size_similarity * 0.1) - label_penalty

            if combined_score > best_score and (iou_score > IOU_THRESHOLD or combined_score > TRACK_CONFIDENCE_THRESHOLD):
                best_score = combined_score
                best_id = track_id

        if best_id is not None:
            # Update track with new detection
            track = TRACKS[best_id]
            old_centroid = track['centroid']

            # Calculate velocity (smooth it with previous velocity)
            vx = det['centroid'][0] - old_centroid[0]
            vy = det['centroid'][1] - old_centroid[1]

            # Smooth velocity with previous velocity if available
            if 'velocity' in track:
                prev_vx, prev_vy = track['velocity']
                vx = 0.7 * vx + 0.3 * prev_vx
                vy = 0.7 * vy + 0.3 * prev_vy

            votes = dict(track.get('label_votes', {track['label']: 1}))
            votes[det['label']] = votes.get(det['label'], 0) + 1
            stable_label = max(votes, key=votes.get)

            if track['label'] == stable_label:
                track_confidence = min(1.0, track.get('track_confidence', 0.5) + 0.18)
            else:
                if votes[stable_label] >= LABEL_CHANGE_MIN_VOTES and votes[stable_label] >= LABEL_CHANGE_RATIO * sum(votes.values()):
                    track_confidence = max(0.30, min(1.0, track.get('track_confidence', 0.5) + 0.10))
                else:
                    stable_label = track['label']
                    track_confidence = max(0.22, track.get('track_confidence', 0.5) - 0.08)

            updated_tracks[best_id] = {
                'label': stable_label,
                'label_votes': votes,
                'centroid': det['centroid'],
                'bbox': det['bbox'],
                'velocity': (vx, vy),
                'age': 0,
                'confidence': det['confidence'],
                'track_confidence': track_confidence,
                'last_iou': iou_score
            }
            assigned.add(best_id)
        else:
            # Create new track if confidence is sufficient and not overlapping with existing tracks
            overlap = False
            for existing_track in TRACKS.values():
                if _calculate_iou(existing_track['bbox'], det['bbox']) > 0.5:
                    overlap = True
                    break
            if not overlap and det['confidence'] >= 0.28:
                TOTALS[det['label']] += 1
                updated_tracks[NEXT_TRACK_ID] = {
                    'label': det['label'],
                    'label_votes': {det['label']: 1},
                    'centroid': det['centroid'],
                    'bbox': det['bbox'],
                    'velocity': (0, 0),
                    'age': 0,
                    'confidence': det['confidence'],
                    'track_confidence': 0.55,
                    'last_iou': 0.0
                }
                NEXT_TRACK_ID += 1

    # Handle unassigned tracks with improved logic
    for track_id, track in TRACKS.items():
        if track_id not in assigned:
            track['age'] += 1
            track['track_confidence'] = max(0.1, track.get('track_confidence', 0.5) - 0.05)

            max_age = MAX_TRACK_AGE
            if track.get('last_iou', 0) > 0.5:
                max_age += 5

            if track['age'] < max_age and track.get('track_confidence', 0.22) > 0.18:
                predicted_centroid = _predict_position(track)
                vx, vy = track.get('velocity', (0, 0))
                damped_velocity = (vx * 0.88, vy * 0.88)

                x1, y1, x2, y2 = track['bbox']
                predicted_bbox = (
                    int(x1 + damped_velocity[0] * 0.5),
                    int(y1 + damped_velocity[1] * 0.5),
                    int(x2 + damped_velocity[0] * 0.5),
                    int(y2 + damped_velocity[1] * 0.5)
                )

                updated_tracks[track_id] = {
                    **track,
                    'centroid': predicted_centroid,
                    'bbox': predicted_bbox,
                    'velocity': damped_velocity
                }

    TRACKS = updated_tracks


def _normalize_class_name(cls):
    if not ORIGINAL_CLASSES:
        return ''
    return ORIGINAL_CLASSES.get(cls, '').lower()


def _update_tracks_without_detection():
    """Update existing tracks when a frame is processed without a full detection pass."""
    _match_tracks([])


def _map_class_to_animal(cls, conf, bbox_area, aspect_ratio):
    normalized = _normalize_class_name(cls)
    if normalized:
        if any(keyword in normalized for keyword in ['chicken', 'hen', 'rooster', 'cock']):
            return 'chicken'
        if any(keyword in normalized for keyword in ['duck', 'goose', 'swan', 'turkey', 'waterfowl']):
            return 'duck'
        if any(keyword in normalized for keyword in ['pig', 'boar', 'hog', 'sow', 'piglet']):
            return 'pig'
        if any(keyword in normalized for keyword in ['cow', 'sheep', 'bull', 'calf', 'horse', 'elephant', 'cattle']):
            return 'pig'
        if 'bird' in normalized:
            if bbox_area < 9000:
                return 'chicken'
            if bbox_area < 22000:
                return 'duck' if aspect_ratio >= 0.95 else 'chicken'
            return 'duck'
        if any(keyword in normalized for keyword in ['dog', 'cat']):
            if conf < 0.35:
                return None
            if bbox_area > 35000 or (aspect_ratio > 1.3 and bbox_area > 20000):
                return 'pig'
            if bbox_area > 20000:
                return 'duck'
            return 'chicken'

    if cls == COCO_BIRD:
        if bbox_area < 9000:
            return 'chicken'
        if bbox_area < 22000:
            return 'duck' if aspect_ratio >= 1.0 else 'chicken'
        return 'duck'
    if cls in (COCO_SHEEP, COCO_COW, COCO_HORSE, COCO_ELEPHANT):
        return 'pig'
    if cls in (COCO_DOG, COCO_CAT):
        if conf < 0.35:
            return None
        if bbox_area > 35000 or (aspect_ratio > 1.3 and bbox_area > 20000):
            return 'pig'
        if bbox_area > 20000:
            return 'duck'
        return 'chicken'
    return None


def process_frame(frame, detect=True):
    counts = {"chicken": 0, "duck": 0, "pig": 0}

    if model is None:
        cv2.putText(frame,
                    "Model unavailable",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)
        return frame, counts, TOTALS.copy()

    detections = []
    detection_counts = {"chicken": 0, "duck": 0, "pig": 0}

    if detect:
        results = model(frame, conf=DEFAULT_CONFIDENCE, iou=DEFAULT_IOU, imgsz=DEFAULT_IMG_SIZE, verbose=False)

        for r in results:
            if not hasattr(r, 'boxes'):
                continue
            for box in r.boxes:
                try:
                    cls = int(_to_float(box.cls))
                    conf = _to_float(box.conf)
                    x1, y1, x2, y2 = _to_int_coords(box.xyxy)
                except Exception:
                    continue

                bbox_width = max(1, x2 - x1)
                bbox_height = max(1, y2 - y1)
                bbox_area = bbox_width * bbox_height
                aspect_ratio = bbox_width / bbox_height

                if bbox_area < MIN_BBOX_AREA or bbox_area > MAX_BBOX_AREA:
                    continue

                label = _map_class_to_animal(cls, conf, bbox_area, aspect_ratio)
                if label is None:
                    continue

                if conf < 0.15:
                    continue

                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                detections.append({
                    'label': label,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'centroid': centroid,
                    'coco_class': ORIGINAL_CLASSES.get(cls, "unknown") if ORIGINAL_CLASSES else "unknown"
                })
                detection_counts[label] += 1

        _match_tracks(detections)
    else:
        # Use track prediction to fill in intermediate frames and keep counts stable
        _update_tracks_without_detection()

    # Build stable current counts from active tracks, which helps when a detection drops temporarily.
    track_counts = {"chicken": 0, "duck": 0, "pig": 0}
    for track_id, track in TRACKS.items():
        if track.get('track_confidence', 0.0) >= 0.18 and track.get('age', 0) <= 4:
            track_counts[track['label']] += 1

    for animal in counts.keys():
        counts[animal] = max(detection_counts[animal], track_counts[animal])

    # Draw detections for objects recognized this frame
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        conf = det['confidence']
        color = COLORS.get(label, (0, 255, 0))

        track_id = None
        for tid, track in TRACKS.items():
            if (track['centroid'] == det['centroid'] and
                track['label'] == det['label'] and
                track['age'] == 0):
                track_id = tid
                break

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        track_info = f"ID:{track_id} " if track_id is not None else ""
        origin = det.get('coco_class', 'unknown')
        label_text = f"{label} ({origin}) {track_info}{conf:.2f}"

        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        text_w, text_h = text_size
        text_x = max(x1, 0)
        text_y = y1 - 10 if y1 - 10 > text_h + 4 else y1 + text_h + 14
        cv2.rectangle(frame,
                      (text_x, text_y - text_h - 6),
                      (text_x + text_w + 12, text_y + 2),
                      (0, 0, 0),
                      -1)
        cv2.putText(frame,
                    label_text,
                    (text_x + 6, text_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    1,
                    cv2.LINE_AA)

    # Draw predicted bounding boxes for lost tracks to improve continuity.
    for track_id, track in TRACKS.items():
        if track['age'] > 0 and track['age'] < MAX_TRACK_AGE and track.get('track_confidence', 0.25) >= 0.30:
            x1, y1, x2, y2 = track['bbox']
            color = COLORS.get(track['label'], (255, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, f"Pred ID:{track_id}", (x1, max(y1-8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cx, cy = track['centroid']
            cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)

    return frame, counts, TOTALS.copy()


def reset_counts():
    """Reset tracking and total counts."""
    global TRACKS, NEXT_TRACK_ID, TOTALS
    TRACKS = {}
    NEXT_TRACK_ID = 0
    TOTALS = {"chicken": 0, "duck": 0, "pig": 0}
    print("Counts reset to zero")


def export_counts_to_csv(filename="livestock_counts.csv"):
    try:
        import csv
        filepath = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Animal Type', 'Total Count'])
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for animal, count in TOTALS.items():
                writer.writerow([timestamp, animal, count])
        return filepath
    except Exception as e:
        print(f"Error exporting CSV: {e}")
        return None