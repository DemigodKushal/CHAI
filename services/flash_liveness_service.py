# services/flash_liveness_service.py
import cv2
import numpy as np
import mediapipe as mp
import time
import random

# --- Tunable Parameters ---
FLASH_DURATION_MS = 160
AFTER_CAPTURE_COUNT = 5
BEFORE_CAPTURE_COUNT = 5
MIN_MEAN_DELTA = 0.03
MAX_NONUNIFORMITY = 0.85
MAX_MEAN = 0.4
WARN_BEFORE_FLASH = True
USE_RANDOM_COLOR = False

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)


class FlashLivenessService:
    def __init__(self):
        pass

    def _get_face_bbox(self, frame, detection):
        ih, iw = frame.shape[:2]
        bb = detection.location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * iw))
        y1 = max(0, int(bb.ymin * ih))
        x2 = min(iw, int((bb.xmin + bb.width) * iw))
        y2 = min(ih, int((bb.ymin + bb.height) * ih))
        return x1, y1, x2, y2

    def _sample_frames(self, cap, count=3, wait_ms=30):
        frames = []
        for _ in range(count):
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f.copy())
            if wait_ms:
                cv2.waitKey(wait_ms)
        return frames

    def _compute_flash_metrics(self, before_imgs, after_imgs, bbox):
        x1, y1, x2, y2 = bbox

        def mean_v(frames):
            vs = []
            for f in frames:
                if f is None:
                    continue
                crop = f[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                v = hsv[:, :, 2].astype(np.float32) / 255.0
                vs.append(v)
            if not vs:
                return None
            return np.mean(np.stack(vs, axis=0), axis=0)

        v_before = mean_v(before_imgs)
        v_after = mean_v(after_imgs)
        if v_before is None or v_after is None:
            return None, None

        delta = v_after - v_before
        delta_pos = np.clip(delta, 0, None)
        mean_delta = float(np.mean(delta_pos))
        eps = 1e-8
        nonuniformity = float(np.std(delta_pos) / (mean_delta + eps)) if mean_delta > 0 else 0.0

        return mean_delta, nonuniformity

    def _fullscreen_flash(self, color=(255, 255, 255), duration_ms=120):
        """Displays a full-screen flash for the given duration."""
        try:
            winname = "FLASH"
            cv2.namedWindow(winname, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            blank = np.zeros((100, 100, 3), dtype=np.uint8)
            blank[:] = color
            cv2.imshow(winname, blank)
            cv2.waitKey(duration_ms)
            cv2.destroyWindow(winname)
        except Exception:
            fallback = np.zeros((600, 800, 3), dtype=np.uint8)
            fallback[:] = color
            cv2.imshow("FLASH_FALLBACK", fallback)
            cv2.waitKey(duration_ms)
            cv2.destroyWindow("FLASH_FALLBACK")

    def _pick_bright_color(self):
        h = random.randint(0, 179)
        s = random.randint(100, 255)
        v = random.randint(200, 255)
        hsv = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def verify_liveness(self, cap):
        """Runs the flash-based liveness test. Returns True if LIVE, else False."""
        ret, frame = cap.read()
        if not ret:
            print("❌ Unable to read frame for liveness check.")
            return False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)
        if not results.detections:
            print("❌ No face detected for liveness test.")
            return False

        det = results.detections[0]
        bbox = self._get_face_bbox(frame, det)

        # Capture frames before and after flash
        before = self._sample_frames(cap, BEFORE_CAPTURE_COUNT, 30)
        if WARN_BEFORE_FLASH:
            time.sleep(0.2)

        color = self._pick_bright_color() if USE_RANDOM_COLOR else (255, 255, 255)
        self._fullscreen_flash(color=color, duration_ms=FLASH_DURATION_MS)
        after = self._sample_frames(cap, AFTER_CAPTURE_COUNT, 20)

        mean_delta, nonuniformity = self._compute_flash_metrics(before, after, bbox)
        if mean_delta is None:
            print("⚠️ Could not compute liveness metrics.")
            return False

        is_live = (mean_delta >= MIN_MEAN_DELTA) and \
                  (nonuniformity <= MAX_NONUNIFORMITY) and \
                  (mean_delta <= MAX_MEAN)

        print(f"Liveness → meanΔ={mean_delta:.3f}, nonuni={nonuniformity:.3f}, result={'LIVE' if is_live else 'SPOOF'}")
        return is_live
