"""
Flash-based liveness check (single-file).

Usage:
 - Run the script.
 - Position face in webcam view.
 - Press 'f' to run a randomized flash liveness prompt.
 - Press 'q' to quit.

Dependencies:
 pip install opencv-python mediapipe numpy
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import random

# -------- CONFIG (tweak these) --------
FLASH_DURATION_MS = 160         # flash on-screen time in milliseconds (0.08-0.18 recommended)
AFTER_CAPTURE_COUNT = 5         # number of frames to capture after flash and average
BEFORE_CAPTURE_COUNT = 5        # frames to sample before flash
MIN_MEAN_DELTA = 0.03           # normalized mean delta threshold (0-1) - tunable
MAX_NONUNIFORMITY = 0.85        # std(delta)/mean(delta) threshold - tunable
MAX_MEAN = 0.4                  # Anomalies have very high mean_delta
FLASH_WINDOW_PADDING = 40       # pixels to expand bounding box to include cheek reflections
USE_RANDOM_COLOR = False        # if True, flash will pick a random bright color; else white
WARN_BEFORE_FLASH = True        # show a warning message briefly before flash (for safety)
# --------------------------------------

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def get_face_bbox(frame, detection):
    ih, iw = frame.shape[:2]
    bb = detection.location_data.relative_bounding_box
    x1 = max(0, int(bb.xmin * iw))
    y1 = max(0, int(bb.ymin * ih))
    x2 = min(iw, int((bb.xmin + bb.width) * iw))
    y2 = min(ih, int((bb.ymin + bb.height) * ih))
    return x1, y1, x2, y2

def expand_bbox(x1, y1, x2, y2, frame_shape, padding):
    ih, iw = frame_shape[:2]
    x1p = max(0, x1 - padding)
    y1p = max(0, y1 - padding)
    x2p = min(iw, x2 + padding)
    y2p = min(ih, y2 + padding)
    return x1p, y1p, x2p, y2p

def sample_frames(cap, count=3, wait_ms=30):
    """Capture 'count' frames from cap quickly and return list."""
    frames = []
    for _ in range(count):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f.copy())
        # small wait to let camera expose
        if wait_ms:
            cv2.waitKey(wait_ms)
    return frames

def compute_flash_metrics(before_imgs, after_imgs, bbox):
    """
    before_imgs / after_imgs: lists of BGR frames
    bbox: (x1,y1,x2,y2)
    Returns: mean_delta_norm, nonuniformity
    """
    x1, y1, x2, y2 = bbox
    # Convert to V channel (brightness) and average multiple frames
    def mean_v(frames):
        vs = []
        for f in frames:
            if f is None: continue
            crop = f[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2].astype(np.float32) / 255.0
            vs.append(v)
        if not vs:
            return None
        return np.mean(np.stack(vs, axis=0), axis=0)  # shape (frames, h, w) -> mean across frames -> (h,w)

    v_before = mean_v(before_imgs)
    v_after = mean_v(after_imgs)
    if v_before is None or v_after is None:
        return None, None

    # delta per-pixel
    delta = v_after - v_before  # could be negative if darkened
    # focus on positive reflectance increase
    delta_pos = np.clip(delta, 0, None)
    mean_delta = float(np.mean(delta_pos))  # normalized 0..1
    # non-uniformity: std / (mean + eps)
    eps = 1e-8
    nonuniformity = float(np.std(delta_pos) / (mean_delta + eps)) if mean_delta > 0 else 0.0

    return mean_delta, nonuniformity

def fullscreen_flash(color=(255,255,255), duration_ms=120):
    """Show fullscreen colored flash for duration_ms. Returns after flash."""
    # create full screen window
    try:
        winname = "FLASH_WINDOW"
        cv2.namedWindow(winname, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        blank = np.zeros((100,100,3), dtype=np.uint8)
        blank[:] = color
        cv2.imshow(winname, blank)
        cv2.waitKey(duration_ms)
        cv2.destroyWindow(winname)
    except Exception as e:
        # fallback: show in normal window (less effective)
        fallback = "FLASH_FALLBACK"
        blank = np.zeros((600,800,3), dtype=np.uint8)
        blank[:] = color
        cv2.imshow(fallback, blank)
        cv2.waitKey(duration_ms)
        cv2.destroyWindow(fallback)

def pick_bright_color():
    # pick random bright RGB color ensuring high V in HSV
    h = random.randint(0, 179)
    s = random.randint(100, 255)
    v = random.randint(200, 255)
    # convert HSV to BGR
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0].tolist()
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def run_flash_liveness(cap, face_bbox):
    x1,y1,x2,y2 = face_bbox
    x1p,y1p,x2p,y2p = expand_bbox(x1,y1,x2,y2, (cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)), FLASH_WINDOW_PADDING)
    # capture before frames
    before = sample_frames(cap, BEFORE_CAPTURE_COUNT, wait_ms=30)
    # short warning to user
    if WARN_BEFORE_FLASH:
        # small overlay in main window (handled by main loop) - we sleep briefly here
        time.sleep(0.2)
    # pick color
    if USE_RANDOM_COLOR:
        color = pick_bright_color()
    else:
        color = (255,255,255)
    # execute flash (blocks for duration)
    fullscreen_flash(color=color, duration_ms=FLASH_DURATION_MS)
    # capture after frames
    after = sample_frames(cap, AFTER_CAPTURE_COUNT, wait_ms=20)
    # compute metrics
    mean_delta, nonuniformity = compute_flash_metrics(before, after, (x1,y1,x2,y2))
    return mean_delta, nonuniformity, color

def main_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Flash liveness demo")
    print(" - Press 'f' to run flash check (requires explicit press).")
    print(" - Press 'q' to quit.")
    print("Safety: Flash can be bright — do not use on/by sensitive individuals without consent.")

    last_detection = None
    result_text = ""
    debug_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)

        ih, iw = frame.shape[:2]
        if results.detections:
            det = results.detections[0]
            x1,y1,x2,y2 = get_face_bbox(frame, det)
            # draw bbox
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            last_detection = (x1,y1,x2,y2)
            cv2.putText(frame, "Face detected", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            cv2.putText(frame, "No face", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # show last result text
        if result_text:
            cv2.putText(frame, result_text, (10, ih-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        if debug_text:
            cv2.putText(frame, debug_text, (10, ih-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

        cv2.imshow("Liveness Flash Demo", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break
        if k == ord('f'):
            if last_detection is None:
                result_text = "No face to test!"
                continue
            # warning
            if WARN_BEFORE_FLASH:
                result_text = "Warning: flash in 0.5s. Press 'f' again to cancel."
                cv2.imshow("Liveness Flash Demo", frame)
                # brief wait to allow user to cancel
                key = cv2.waitKey(500) & 0xFF
                if key == ord('f'):
                    result_text = "Flash cancelled."
                    continue
            result_text = "Running flash..."
            cv2.imshow("Liveness Flash Demo", frame)
            cv2.waitKey(100)

            # run test
            mean_delta, nonuniformity, color = run_flash_liveness(cap, last_detection)
            if mean_delta is None:
                result_text = "Flash test failed: couldn't capture ROI frames."
                debug_text = ""
                continue

            # Decide live/spoof using thresholds (tunable)
            is_live =  (nonuniformity <= MAX_NONUNIFORMITY) and (mean_delta <= MAX_MEAN)
            result_text = f"Result: {'LIVE' if is_live else 'SPOOF'}  meanΔ={mean_delta:.3f} nonuni={nonuniformity:.3f}"
            debug_text = f"Color: {color}  meanΔ={mean_delta:.3f} nonuni={nonuniformity:.3f}"
            # small beep or feedback (optional)
            # print(result_text)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
