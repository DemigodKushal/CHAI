# services/liveness_detector.py - UPDATED FINAL VERSION
"""
Multi-layer liveness detection with STRICT EDGE DENSITY CHECK
Prevents screen spoofs from passing
"""

import cv2
import numpy as np


class LivenessDetector:
    """Enhanced liveness detection with hard failure modes"""
    
    def __init__(self):
        # STRICT THRESHOLDS
        self.MIN_BRIGHTNESS_CHANGE = 2.5
        self.MAX_BRIGHTNESS_CHANGE = 12.0
        
        self.MIN_COLOR_VARIANCE = 2400.0
        self.MAX_EDGE_DENSITY = 0.060          # STRICTER: 0.065 → 0.060
        
        self.MIN_UNIFORMITY = 52.0
        self.MAX_NONUNIFORMITY = 0.40          # STRICTER: 0.45 → 0.40
        
        self.MIN_MEAN_DELTA = 0.8
        self.MAX_MEAN_DELTA = 2.5
        
        self.MIN_TOTAL_SCORE = 4.2             # RAISED: 4.0 → 4.2
        
        # HARD FAIL thresholds (automatic fail regardless of score)
        self.HARD_FAIL_EDGE_DENSITY = 0.065    # Screens always > 0.06
        self.HARD_FAIL_LOW_VARIANCE = 2200.0   # Too far or screen
    
    def get_brightness(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        return center_region.mean()
    
    def get_color_variance(self, frame):
        b, g, r = cv2.split(frame)
        h, w = b.shape
        center_b = b[h//4:3*h//4, w//4:3*w//4]
        center_g = g[h//4:3*h//4, w//4:3*w//4]
        center_r = r[h//4:3*h//4, w//4:3*w//4]
        return np.mean([np.var(center_b), np.var(center_g), np.var(center_r)])
    
    def get_edge_density(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        edges = cv2.Canny(center_region, 50, 150)
        return np.sum(edges > 0) / edges.size
    
    def get_brightness_uniformity(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        return np.std(center_region)
    
    def get_nonuniformity(self, before_frames, after_frames):
        before_brightness_list = [self.get_brightness(f) for f in before_frames]
        after_brightness_list = [self.get_brightness(f) for f in after_frames]
        brightness_changes = [after - before for before, after in zip(before_brightness_list, after_brightness_list)]
        return np.std(brightness_changes)
    
    def get_mean_delta(self, before_frames, after_frames):
        deltas = []
        for i in range(len(before_frames) - 1):
            gray1 = cv2.cvtColor(before_frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(before_frames[i+1], cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            deltas.append(diff.mean())
        return np.mean(deltas) if deltas else 0
    
    def _get_distance_feedback(self, metrics):
        """Provide helpful feedback based on distance indicators"""
        variance = metrics['color_variance']
        brightness = metrics['brightness_change_percent']
        
        if variance < 2200 and brightness < 3.0:
            return "⬅️ Please move CLOSER to the camera"
        
        if brightness > 13.0:
            return "➡️ Please move BACK from the camera"
        
        if variance < 2600:
            return "⬅️ Move slightly CLOSER for better detection"
        
        if 2800 <= variance <= 4000 and 3.0 <= brightness <= 10.0:
            return "✅ Distance is perfect!"
        
        return "⚠️ Position not optimal - try adjusting distance"
    
    def analyze_frames(self, before_frames, after_frames):
        before_brightness = np.mean([self.get_brightness(f) for f in before_frames])
        after_brightness = np.mean([self.get_brightness(f) for f in after_frames])
        brightness_change = after_brightness - before_brightness
        
        before_variance = np.mean([self.get_color_variance(f) for f in before_frames])
        before_edges = np.mean([self.get_edge_density(f) for f in before_frames])
        before_uniformity = np.mean([self.get_brightness_uniformity(f) for f in before_frames])
        
        nonuniformity = self.get_nonuniformity(before_frames, after_frames)
        mean_delta = self.get_mean_delta(before_frames, after_frames)
        
        brightness_change_percent = (brightness_change / before_brightness * 100) if before_brightness > 0 else 0
        
        metrics = {
            'before_brightness': before_brightness,
            'after_brightness': after_brightness,
            'brightness_change_percent': brightness_change_percent,
            'color_variance': before_variance,
            'edge_density': before_edges,
            'uniformity': before_uniformity,
            'nonuniformity': nonuniformity,
            'mean_delta': mean_delta
        }
        
        distance_feedback = self._get_distance_feedback(metrics)
        metrics['distance_feedback'] = distance_feedback
        
        is_live, fail_reason, scores = self._check_liveness_with_scoring(metrics)
        metrics['scores'] = scores
        
        return is_live, metrics, fail_reason
    
    def _check_liveness_with_scoring(self, m):
        """Strict scoring with HARD FAIL conditions"""
        scores = {}
        reasons = []
        
        # === HARD FAIL CONDITIONS (immediate rejection) ===
        
        # Hard fail 1: Edge density too high (screen pixels)
        if m['edge_density'] > self.HARD_FAIL_EDGE_DENSITY:
            return False, f"🚫 Screen detected (edge density: {m['edge_density']:.4f})", {}
        
        # Hard fail 2: Variance too low (too far or flat surface)
        if m['color_variance'] < self.HARD_FAIL_LOW_VARIANCE:
            return False, f"📍 Too far from camera or flat surface (variance: {m['color_variance']:.0f})", {}
        
        # Hard fail 3: Nonuniformity too high (screen refresh pattern)
        if m['nonuniformity'] > 0.6:
            return False, f"🚫 Screen refresh pattern detected (nonuniformity: {m['nonuniformity']:.2f})", {}
        
        # === SCORING (if not hard failed) ===
        
        # Check 1: Brightness (0-1)
        if self.MIN_BRIGHTNESS_CHANGE <= m['brightness_change_percent'] <= self.MAX_BRIGHTNESS_CHANGE:
            mid = 6.0
            distance = abs(m['brightness_change_percent'] - mid)
            scores['brightness'] = max(0.5, 1.0 - (distance / 8.0))
        else:
            scores['brightness'] = 0.2
            if m['brightness_change_percent'] < self.MIN_BRIGHTNESS_CHANGE:
                reasons.append(f"Too far from camera")
            else:
                reasons.append(f"Too close to camera")
        
        # Check 2: Color variance (0-1)
        if m['color_variance'] >= self.MIN_COLOR_VARIANCE:
            if m['color_variance'] >= 2800:
                scores['variance'] = min(1.0, 0.8 + (m['color_variance'] - 2800) / 3000)
            else:
                scores['variance'] = 0.5 + (m['color_variance'] - 2400) / 1000
        else:
            scores['variance'] = 0.1
            reasons.append(f"Low detail")
        
        # Check 3: Edge density (0-1) - STRICT
        if m['edge_density'] <= self.MAX_EDGE_DENSITY:
            # More points for lower edge density
            scores['edges'] = 0.6 + (0.4 * (1 - m['edge_density'] / self.MAX_EDGE_DENSITY))
        else:
            scores['edges'] = 0.2  # Partial credit
            reasons.append(f"High edges")
        
        # Check 4: Uniformity (0-1)
        if m['uniformity'] >= self.MIN_UNIFORMITY:
            excess = m['uniformity'] - self.MIN_UNIFORMITY
            scores['uniformity'] = min(1.0, 0.7 + (excess / 30))
        elif m['uniformity'] >= 45:
            scores['uniformity'] = 0.5
        else:
            scores['uniformity'] = 0.2
            reasons.append(f"Flat surface")
        
        # Check 5: Nonuniformity (0-1) - STRICT
        if m['nonuniformity'] <= self.MAX_NONUNIFORMITY:
            scores['nonuniformity'] = 0.8 + (0.2 * (1 - m['nonuniformity'] / self.MAX_NONUNIFORMITY))
        else:
            scores['nonuniformity'] = 0.3
            reasons.append(f"Inconsistent")
        
        # Check 6: Mean delta (0-1)
        if self.MIN_MEAN_DELTA <= m['mean_delta'] <= self.MAX_MEAN_DELTA:
            mid = 1.5
            distance = abs(m['mean_delta'] - mid)
            scores['mean_delta'] = max(0.6, 1.0 - (distance / 1.5))
        elif 0.6 <= m['mean_delta'] <= 2.8:
            scores['mean_delta'] = 0.4
        else:
            scores['mean_delta'] = 0.1
            reasons.append(f"Motion issue")
        
        total_score = sum(scores.values())
        is_live = total_score >= self.MIN_TOTAL_SCORE
        
        if not is_live:
            if m['color_variance'] < 2400 or m['brightness_change_percent'] < 3.0:
                fail_reason = f"📍 {m['distance_feedback']}"
            elif m['brightness_change_percent'] > 13.0:
                fail_reason = f"📍 {m['distance_feedback']}"
            else:
                fail_reason = f"Low confidence ({total_score:.2f}/6.0) - {', '.join(reasons[:2])}"
        else:
            fail_reason = ""
        
        return is_live, fail_reason, scores
    
    def print_analysis(self, metrics):
        m = metrics
        print(f"📊 Liveness Analysis:")
        print(f"   Brightness: {m['before_brightness']:.2f} → {m['after_brightness']:.2f} ({m['brightness_change_percent']:.2f}%)")
        print(f"   Color Variance: {m['color_variance']:.2f}")
        print(f"   Edge Density: {m['edge_density']:.4f}")
        print(f"   Brightness Uniformity: {m['uniformity']:.2f}")
        print(f"   Nonuniformity: {m['nonuniformity']:.2f}")
        print(f"   Mean Delta: {m['mean_delta']:.2f}")
        if 'scores' in m and m['scores']:
            total = sum(m['scores'].values())
            print(f"   📊 Score: {total:.2f}/6.0 (threshold: {self.MIN_TOTAL_SCORE})")
        if 'distance_feedback' in m:
            print(f"   {m['distance_feedback']}")
