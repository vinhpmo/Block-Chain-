import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from detector import process_frame, export_counts_to_csv, reset_counts
import csv
from datetime import datetime
from io import StringIO, BytesIO
import time
import sqlite3
import hashlib

import pandas as pd

app = Flask(__name__)

counts_global = {
    "current": {"chicken": 0, "duck": 0, "pig": 0},
    "total": {"chicken": 0, "duck": 0, "pig": 0}
}

video_fps = 10  # Default FPS control: slow down video for clearer bounding boxes and detection
DETECTION_INTERVAL = 2  # Run a full detection pass every 2 frames for better speed
current_video_source = "file"  # "file" or "camera"
current_camera_url = None
default_camera_url = "http://192.100.171:4747"  # Default camera URL for DROIcam
uploaded_video_path = None

DB_PATH = os.path.join(app.root_path, 'data', 'livestock_ledger.db')
SAVE_SNAPSHOT_INTERVAL = 60.0
last_snapshot_time = 0


def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL;')
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS daily_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day TEXT NOT NULL,
            created_at TEXT NOT NULL,
            source TEXT,
            chicken INTEGER DEFAULT 0,
            duck INTEGER DEFAULT 0,
            pig INTEGER DEFAULT 0,
            total INTEGER DEFAULT 0,
            record_hash TEXT,
            previous_hash TEXT,
            tx_hash TEXT,
            chain_id TEXT
        )
    ''')
    conn.commit()
    conn.close()


def compute_record_hash(day, chicken, duck, pig, total, source, previous_hash):
    payload = f"{day}|{chicken}|{duck}|{pig}|{total}|{source}|{previous_hash}"
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def get_last_hash():
    conn = get_db_connection()
    row = conn.execute('SELECT record_hash FROM daily_snapshots ORDER BY created_at DESC LIMIT 1').fetchone()
    conn.close()
    return row['record_hash'] if row else ''


def open_camera_capture(url):
    """Try opening camera stream with several common URL forms."""
    if not url:
        return None, None

    url = url.strip()
    candidates = []
    if url.isdigit():
        candidates.extend([int(url), 0, 1, 2, 3])
    else:
        candidates.append(url)
        if not url.startswith(('http://', 'https://', 'rtsp://')):
            candidates.append(f'http://{url}')
            candidates.append(f'rtsp://{url}')

        normalized = url.rstrip('/')
        if normalized == url:
            candidates.append(url + '/')

        if normalized.startswith('http://') or normalized.startswith('https://'):
            if not any(path in normalized for path in ['/video', '/shot.jpg', '/mjpeg', '/h264', '/stream', 'action=stream', 'video_feed']):
                candidates.extend([
                    normalized + '/video',
                    normalized + '/video?x-mjpeg',
                    normalized + '/video_feed',
                    normalized + '/?action=stream',
                    normalized + '/shot.jpg',
                    normalized + '/mjpeg',
                    normalized + '/stream',
                    normalized + '/h264',
                    normalized + '/cgi-bin/mjpg/video.cgi?stream=1'
                    , normalized + '/video.mjpg',
                    normalized + '/mjpeg/video.mjpg',
                    normalized + '/mjpg/video.mjpg',
                    normalized + '/cam.mjpg',
                    normalized + '/axis-cgi/mjpg/video.cgi?resolution=640x480',
                    normalized + '/live.m3u8',
                    normalized + '/live/0',
                    normalized + '/live/1'
                ])

            ip_port = normalized.replace('http://', '').replace('https://', '').split('/')[0]
            rtsp_base = f'rtsp://{ip_port}'
            candidates.extend([
                f'{rtsp_base}/live/ch0',
                f'{rtsp_base}/live/ch1',
                f'{rtsp_base}/live',
                f'{rtsp_base}/stream',
                f'{rtsp_base}/h264',
                f'{rtsp_base}/mjpeg',
                f'{rtsp_base}/0',
                f'{rtsp_base}/1'
            ])

        elif normalized.startswith('rtsp://'):
            base = normalized.split('://', 1)[1].split('/')[0]
            rtsp_base = f'rtsp://{base}'
            if normalized == url:
                candidates.extend([
                    rtsp_base + '/live/ch0',
                    rtsp_base + '/live',
                    rtsp_base + '/stream',
                    rtsp_base + '/0'
                ])

    tried = set()
    for candidate in candidates:
        if candidate in tried or candidate == '':
            continue
        tried.add(candidate)
        app.logger.debug(f'Trying camera candidate: {candidate}')
        cap = cv2.VideoCapture(candidate)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            continue

        # For network streams, verify at least one frame can be read.
        if isinstance(candidate, str) and candidate.startswith(('http://', 'https://', 'rtsp://')):
            success = False
            for _ in range(10):
                ok, frame = cap.read()
                if ok and frame is not None:
                    success = True
                    break
                time.sleep(0.2)
            if success:
                return cap, candidate
            try:
                cap.release()
            except Exception:
                pass
            continue

        return cap, candidate

    return None, None


def validate_camera_url(url):
    """Validate the camera URL and return a working endpoint if found."""
    cap, used_url = open_camera_capture(url)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
        return used_url
    return None


def save_daily_snapshot(current_counts, totals, source):
    global last_snapshot_time
    now_ts = time.time()
    if now_ts - last_snapshot_time < SAVE_SNAPSHOT_INTERVAL:
        return None

    last_snapshot_time = now_ts
    day = datetime.now().date().isoformat()
    total_count = totals.get('chicken', 0) + totals.get('duck', 0) + totals.get('pig', 0)
    previous_hash = get_last_hash()
    record_hash = compute_record_hash(
        day,
        current_counts.get('chicken', 0),
        current_counts.get('duck', 0),
        current_counts.get('pig', 0),
        total_count,
        source,
        previous_hash
    )

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO daily_snapshots (day, created_at, source, chicken, duck, pig, total, record_hash, previous_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            day,
            datetime.now().isoformat(),
            source,
            current_counts.get('chicken', 0),
            current_counts.get('duck', 0),
            current_counts.get('pig', 0),
            total_count,
            record_hash,
            previous_hash
        )
    )
    conn.commit()
    snapshot_id = cursor.lastrowid
    conn.close()

    return {
        'id': snapshot_id,
        'day': day,
        'source': source,
        'chicken': current_counts.get('chicken', 0),
        'duck': current_counts.get('duck', 0),
        'pig': current_counts.get('pig', 0),
        'total': total_count,
        'record_hash': record_hash,
        'previous_hash': previous_hash
    }


init_db()

def generate_frames():
    global counts_global, current_video_source, current_camera_url, uploaded_video_path

    cap = None

    if current_video_source == "camera":
        camera_url = current_camera_url if current_camera_url else default_camera_url
        cap, used_source = open_camera_capture(camera_url)
        if cap is not None:
            current_camera_url = used_source
        else:
            cap = None
    else:
        # Use uploaded video or default video
        video_path = uploaded_video_path if uploaded_video_path else os.path.join(app.root_path, "data", "farm.mp4")
        cap = cv2.VideoCapture(video_path)

    if cap is None or not cap.isOpened():
        error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        if current_video_source == "camera":
            source_name = current_camera_url if current_camera_url else "camera"
        else:
            source_name = "video file"
        cv2.putText(error_frame,
                    f"Loi: khong mo duoc {source_name}",
                    (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return

    frame_delay = 1.0 / video_fps if video_fps > 0 else 0.033  # Default ~30 FPS

    frame_index = 0
    while True:
        start_time = time.time()

        success, frame = cap.read()
        if not success:
            if current_video_source == "camera":
                # For camera, just continue trying
                continue
            else:
                # For video file, loop back to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        detect_frame = (frame_index % DETECTION_INTERVAL == 0)
        frame, counts, totals = process_frame(frame, detect=detect_frame)
        frame_index += 1
        counts_global = {"current": counts, "total": totals}

        panel_x1, panel_y1 = 10, 10
        panel_x2, panel_y2 = 260, 140
        cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (48, 209, 88), 2)

        title = "Đếm hiện tại / Tổng"
        cv2.putText(frame, title, (panel_x1 + 10, panel_y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2)

        y = panel_y1 + 50
        for k in counts.keys():
            cv2.putText(frame, f"{k}: {counts[k]} / {totals[k]}",
                        (panel_x1 + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (180, 255, 180),
                        1,
                        cv2.LINE_AA)
            y += 25

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Reduced quality for faster encoding
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Control FPS by adding delay
        elapsed = time.time() - start_time
        frame_delay = 1.0 / video_fps if video_fps > 0 else 0.033
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    direct_passthrough=True)

def compute_trending_summary():
    current = counts_global['current'].copy()
    total = counts_global['total'].copy()
    total_sum = sum(total.values()) or 1
    ranking = sorted(total.items(), key=lambda item: item[1], reverse=True)
    trend = [
        {
            'animal': animal,
            'count': count,
            'percent': round((count / total_sum) * 100, 1)
        }
        for animal, count in ranking
    ]
    top_animal = ranking[0][0] if ranking and ranking[0][1] > 0 else None
    top_animal_count = ranking[0][1] if ranking else 0
    top_animal_percent = round((top_animal_count / total_sum) * 100, 1)

    conn = get_db_connection()
    latest_row = conn.execute(
        'SELECT id, day, source, chicken, duck, pig, total, record_hash, tx_hash, chain_id, created_at '
        'FROM daily_snapshots ORDER BY created_at DESC LIMIT 1'
    ).fetchone()
    conn.close()
    latest_snapshot = dict(latest_row) if latest_row else None

    return {
        'current': current,
        'total': total,
        'top_animal': top_animal,
        'top_animal_count': top_animal_count,
        'top_animal_percent': top_animal_percent,
        'trend': trend,
        'latest_snapshot': latest_snapshot
    }

@app.route('/api/snapshot_history')
def snapshot_history():
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT id, day, created_at, source, chicken, duck, pig, total, record_hash, tx_hash, chain_id '
        'FROM daily_snapshots ORDER BY created_at DESC LIMIT 24'
    ).fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])

@app.route('/api/snapshot_diff')
def snapshot_diff():
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT id, created_at, source, chicken, duck, pig, total, record_hash, tx_hash, chain_id '
        'FROM daily_snapshots ORDER BY created_at DESC LIMIT 2'
    ).fetchall()
    conn.close()

    if not rows:
        return jsonify({'latest': None, 'previous': None, 'diff': {}})

    latest = dict(rows[0])
    previous = dict(rows[1]) if len(rows) > 1 else None
    diff = {}
    if previous:
        diff = {
            'chicken': latest['chicken'] - previous['chicken'],
            'duck': latest['duck'] - previous['duck'],
            'pig': latest['pig'] - previous['pig'],
            'total': latest['total'] - previous['total']
        }

    return jsonify({'latest': latest, 'previous': previous, 'diff': diff})

@app.route('/counts')
def counts():
    try:
        save_daily_snapshot(counts_global['current'], counts_global['total'], current_video_source)
    except Exception as e:
        print('Could not save snapshot:', e)
    return jsonify(counts_global)

@app.route('/api/trending')
def api_trending():
    try:
        return jsonify(compute_trending_summary())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export/excel')
def export_excel():
    try:
        # Create data for Excel export
        data = []
        timestamp = datetime.now()
        date_str = timestamp.strftime('%Y-%m-%d')
        time_str = timestamp.strftime('%H:%M:%S')

        for animal in ['chicken', 'duck', 'pig']:
            data.append({
                'Ngày': date_str,
                'Thời gian': time_str,
                'Loại vật nuôi': animal,
                'Số lượng hiện tại': counts_global['current'].get(animal, 0),
                'Tổng số lượng': counts_global['total'].get(animal, 0),
                'Nguồn video': 'Camera' if current_video_source == 'camera' else 'Video file'
            })

        # Create DataFrame
        df = pd.DataFrame(data)

        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Đếm vật nuôi', index=False)

            # Auto-adjust column widths
            worksheet = writer.sheets['Đếm vật nuôi']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column_letter].width = adjusted_width

        output.seek(0)

        filename = f'dem_vat_nuoi_{date_str}_{timestamp.strftime("%H%M%S")}.xlsx'
        return Response(
            output.getvalue(),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={
                'Content-Disposition': f'attachment; filename={filename}'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fps', methods=['GET', 'POST'])
def fps_control():
    global video_fps
    if request.method == 'POST':
        data = request.get_json()
        fps = data.get('fps', 60)
        if 1 <= fps <= 60:
            video_fps = fps
            return jsonify({'fps': video_fps, 'status': 'updated'})
        return jsonify({'error': 'FPS must be between 1 and 60'}), 400
    return jsonify({'fps': video_fps})

@app.route('/api/reset', methods=['POST'])
def reset_counts_endpoint():
    global counts_global
    reset_counts()  # This resets the internal TOTALS in detector.py
    counts_global["total"] = {"chicken": 0, "duck": 0, "pig": 0}
    return jsonify({'status': 'reset_totals', 'counts': counts_global, 'message': 'Đã reset tổng số lượng, tiếp tục đếm từ đầu'})

@app.route('/api/save_snapshot', methods=['POST'])
def save_snapshot_endpoint():
    try:
        snapshot = save_daily_snapshot(counts_global['current'], counts_global['total'], current_video_source)
        if snapshot is None:
            return jsonify({'status': 'skipped', 'message': 'Snapshot đã được lưu gần đây, vui lòng thử lại sau một lát.'})
        return jsonify({'status': 'success', 'snapshot': snapshot})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/daily_records')
def daily_records():
    conn = get_db_connection()
    query = '''
        SELECT ds.id, ds.day, ds.created_at, ds.source, ds.chicken, ds.duck, ds.pig, ds.total,
               ds.record_hash, ds.previous_hash, ds.tx_hash, ds.chain_id
        FROM daily_snapshots ds
        JOIN (
            SELECT day, MAX(created_at) AS max_created_at
            FROM daily_snapshots
            GROUP BY day
        ) latest ON ds.day = latest.day AND ds.created_at = latest.max_created_at
        ORDER BY ds.day DESC
    '''
    rows = conn.execute(query).fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])

@app.route('/api/store_tx', methods=['POST'])
def store_tx():
    data = request.get_json() or {}
    record_id = data.get('record_id')
    tx_hash = data.get('tx_hash')
    chain_id = data.get('chain_id')

    if not record_id or not tx_hash:
        return jsonify({'status': 'error', 'message': 'record_id và tx_hash là bắt buộc.'}), 400

    conn = get_db_connection()
    conn.execute(
        'UPDATE daily_snapshots SET tx_hash = ?, chain_id = ? WHERE id = ?',
        (tx_hash, chain_id, record_id)
    )
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'message': 'Đã lưu giao dịch blockchain.'})

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    global current_video_source, current_camera_url, uploaded_video_path

    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No video file selected'}), 400

    if file and file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Save uploaded video
        upload_dir = os.path.join(app.root_path, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        filename = f"uploaded_{int(time.time())}_{file.filename}"
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        # Switch to uploaded video
        current_video_source = "file"
        uploaded_video_path = filepath
        current_camera_url = None

        return jsonify({'status': 'success', 'message': 'Video uploaded successfully', 'filename': filename})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid video file format'}), 400

@app.route('/api/set_camera_url', methods=['POST'])
def set_camera_url():
    global current_video_source, current_camera_url, uploaded_video_path

    data = request.get_json() or {}
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'status': 'error', 'message': 'URL camera không hợp lệ.'}), 400
    valid_url = validate_camera_url(url)
    if not valid_url:
        return jsonify({'status': 'error', 'message': 'Không thể kết nối camera URL. Vui lòng kiểm tra lại.'}), 400

    current_video_source = "camera"
    current_camera_url = valid_url
    uploaded_video_path = None

    return jsonify({'status': 'success', 'message': 'Kết nối camera URL thành công', 'camera_url': valid_url})

@app.route('/api/switch_camera', methods=['POST'])
def switch_camera():
    global current_video_source, current_camera_url, uploaded_video_path

    current_video_source = "camera"
    current_camera_url = None
    uploaded_video_path = None

    return jsonify({'status': 'success', 'message': 'Switched to camera mode'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)