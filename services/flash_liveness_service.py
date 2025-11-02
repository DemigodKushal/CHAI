# services/flash_liveness_service.py
import cv2
import numpy as np
import mediapipe as mp
import time
import random

# -------- CONFIG (tweak these) --------
FLASH_DURATION_MS = 160
AFTER_CAPTURE_COUNT = 5
BEFORE_CAPTURE_COUNT = 5
MIN_MEAN_DELTA = 0.03
MAX_NONUNIFORMITY = 3.0    # ✅ CHANGED from 1.2 to 3.0 (more lenient)
MAX_MEAN = 0.5             # ✅ CHANGED from 0.2 to 0.5 (more lenient)
FLASH_WINDOW_PADDING = 40
USE_RANDOM_COLOR = False
WARN_BEFORE_FLASH = True

mp_face = mp.solutions.face_detection


class FlashLivenessService:
    def __init__(self, min_mean_delta=MIN_MEAN_DELTA,
                 max_nonuniformity=MAX_NONUNIFORMITY,
                 max_mean=MAX_MEAN,
                 flash_duration_ms=FLASH_DURATION_MS,
                 before_count=BEFORE_CAPTURE_COUNT,
                 after_count=AFTER_CAPTURE_COUNT,
                 padding=FLASH_WINDOW_PADDING,
                 use_random_color=USE_RANDOM_COLOR,
                 warn_before_flash=WARN_BEFORE_FLASH):
        self.min_mean_delta = min_mean_delta
        self.max_nonuniformity = max_nonuniformity
        self.max_mean = max_mean
        self.flash_duration_ms = flash_duration_ms
        self.before_count = before_count
        self.after_count = after_count
        self.padding = padding
        self.use_random_color = use_random_color
        self.warn_before_flash = warn_before_flash

        self.detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    def _get_face_bbox(self, frame, detection):
        ih, iw = frame.shape[:2]
        bb = detection.location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * iw))
        y1 = max(0, int(bb.ymin * ih))
        x2 = min(iw, int((bb.xmin + bb.width) * iw))
        y2 = min(ih, int((bb.ymin + bb.height) * ih))
        return x1, y1, x2, y2

    def _expand_bbox(self, x1, y1, x2, y2, frame_shape, padding):
        # frame_shape can be (h, w) or floats from cap.get; make ints
        if isinstance(frame_shape, tuple) and len(frame_shape) >= 2:
            ih, iw = int(frame_shape[0]), int(frame_shape[1])
        else:
            # fallback
            ih, iw = frame_shape, frame_shape
        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(iw, x2 + padding)
        y2p = min(ih, y2 + padding)
        return x1p, y1p, x2p, y2p

    def _sample_frames(self, cap, count=3, wait_ms=30):
        frames = []
        for _ in range(count):
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f.copy())
            if wait_ms:
                # small wait to let camera auto-expose settle if needed
                cv2.waitKey(wait_ms)
        return frames

    def _compute_flash_metrics(self, before_imgs, after_imgs, bbox):
        x1, y1, x2, y2 = bbox

        def mean_v(frames):
            vs = []
            for f in frames:
                if f is None:
                    continue
                # guard cropping indices
                h, w = f.shape[:2]
                xa = max(0, min(w, x1))
                xb = max(0, min(w, x2))
                ya = max(0, min(h, y1))
                yb = max(0, min(h, y2))
                if xb <= xa or yb <= ya:
                    continue
                crop = f[ya:yb, xa:xb]
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

        # delta per-pixel
        delta = v_after - v_before  # could be negative if darkened
        # focus on positive reflectance increase
        delta_pos = np.clip(delta, 0, None)
        mean_delta = float(np.mean(delta_pos))  # normalized 0..1
        # non-uniformity: std / (mean + eps)
        eps = 1e-8
        nonuniformity = float(np.std(delta_pos) / (mean_delta + eps)) if mean_delta > 0 else 0.0

        return mean_delta, nonuniformity

    def _fullscreen_flash(self, color=(255, 255, 255), duration_ms=120):
        """
        Open a separate fullscreen OpenCV window for the flash.
        This blocks for duration_ms and then closes the window.
        """
        try:
            winname = "FLASH_WINDOW"
            cv2.namedWindow(winname, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # create a full-screen sized blank using the primary monitor resolution if possible
            # but OpenCV uses the window system; drawing a 100x100 filled rectangle works for fullscreen window
            blank = np.zeros((100, 100, 3), dtype=np.uint8)
            blank[:] = color
            cv2.imshow(winname, blank)
            cv2.waitKey(duration_ms)
            cv2.destroyWindow(winname)
        except Exception:
            # fallback: regular window
            fallback = "FLASH_FALLBACK"
            blank = np.zeros((600, 800, 3), dtype=np.uint8)
            blank[:] = color
            cv2.imshow(fallback, blank)
            cv2.waitKey(duration_ms)
            cv2.destroyWindow(fallback)

    def _pick_bright_color(self):
        # pick random bright RGB color ensuring high V in HSV
        h = random.randint(0, 179)
        s = random.randint(100, 255)
        v = random.randint(200, 255)
        hsv = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def run_flash_liveness(self, cap, face_bbox):
        """
        Run the flash liveness sequence using the provided camera and face bbox.
        Returns (mean_delta, nonuniformity, color) or (None, None, None) on failure.
        """
        x1, y1, x2, y2 = face_bbox

        # determine frame dims from cap if possible (cap.get returns floats)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.get(cv2.CAP_PROP_FRAME_HEIGHT) else None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.get(cv2.CAP_PROP_FRAME_WIDTH) else None
        frame_shape = (h, w) if h and w else None

        # expand bbox to include cheeks etc (if frame dims available)
        if frame_shape:
            x1p, y1p, x2p, y2p = self._expand_bbox(x1, y1, x2, y2, frame_shape, self.padding)
        else:
            x1p, y1p, x2p, y2p = x1, y1, x2, y2

        # capture before frames
        before = self._sample_frames(cap, self.before_count, wait_ms=30)
        if self.warn_before_flash:
            time.sleep(0.2)

        # pick color
        color = self._pick_bright_color() if self.use_random_color else (255, 255, 255)

        # execute flash (blocks for duration)
        self._fullscreen_flash(color=color, duration_ms=self.flash_duration_ms)

        # capture after frames
        after = self._sample_frames(cap, self.after_count, wait_ms=20)

        # compute metrics (use expanded bbox coordinates)
        mean_delta, nonuniformity = self._compute_flash_metrics(before, after, (x1p, y1p, x2p, y2p))
        return mean_delta, nonuniformity, color

    def verify_liveness(self, cap):
        """
        High-level helper: detect face in current frame, run flash-liveness, decide LIVE/SPOOF.
        Returns boolean (True for LIVE, False for SPOOF or failure).
        """
        # read a frame for face detection
        ret, frame = cap.read()
        if not ret:
            print("❌ Unable to read frame for liveness check.")
            return False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        if not results.detections:
            print("❌ No face detected for liveness test.")
            return False

        det = results.detections[0]
        bbox = self._get_face_bbox(frame, det)

        # run flash-liveness sequence
        metrics = self.run_flash_liveness(cap, bbox)
        if metrics is None:
            print("⚠️ Could not compute liveness metrics.")
            return False

        mean_delta, nonuniformity, color = metrics

        if mean_delta is None:
            print("⚠️ Could not compute liveness metrics (None).")
            return False

        # Decision per supplied logic: (nonuniformity <= MAX_NONUNIFORMITY) and (mean_delta <= MAX_MEAN)
        is_live = (nonuniformity <= self.max_nonuniformity) and (mean_delta <= self.max_mean)

        print(f"Liveness → meanΔ={mean_delta:.3f}, nonuni={nonuniformity:.3f}, color={color}, result={'LIVE' if is_live else 'SPOOF'}")
        return is_live
