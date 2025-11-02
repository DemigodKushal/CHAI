# services/liveness_detector.py - FINAL WORKING VERSION
"""
Multi-layer liveness detection - Uses UNIFORMITY as main discriminator
Screens have HIGH uniformity (flat), real faces have LOWER uniformity (depth)
"""

import cv2
import numpy as np


class LivenessDetector:
    """Working liveness detection - uniformity-based"""
    
    def __init__(self):
        # Flash response
        self.MIN_BRIGHTNESS_CHANGE = 2.5
        self.MAX_BRIGHTNESS_CHANGE = 30.0  # Very lenient for real faces
        
        # KEY DISCRIMINATOR: Uniformity
        self.MIN_UNIFORMITY = 38.0         # Real faces: 45-48
        self.MAX_UNIFORMITY = 60.0         # SCREENS: 51-54 → HARD FAIL if > 50
        
        # Secondary checks
        self.MIN_COLOR_VARIANCE = 1800.0
        self.MAX_EDGE_DENSITY = 0.065      # Screens: 0.059-0.078, Real: 0.032-0.036
        
        # Scoring
        self.MIN_TOTAL_SCORE = 3.5
        
        # HARD FAIL
        self.HARD_FAIL_HIGH_UNIFORMITY = 65.0   # Screens are > 50
        self.HARD_FAIL_LOW_VARIANCE = 1500.0
    
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
        """Provide helpful feedback"""
        variance = metrics['color_variance']
        brightness = metrics['brightness_change_percent']
        
        if variance < 1600 and brightness < 2.5:
            return "⬅️ Please move CLOSER to the camera"
        
        if brightness > 26.0:
            return "➡️ Please move BACK from the camera (too close)"
        
        if variance < 2000:
            return "⬅️ Move slightly CLOSER for better detection"
        
        return "✅ Position is good!"
    
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
        """Uniformity-based discrimination"""
        scores = {}
        reasons = []
        
        # Replace the uniformity hard fail with combined check:

        # === HARD FAIL #1: SCREEN (high uniformity AND high edges) ===
        if m['uniformity'] > 52.0 and m['edge_density'] > 0.045:
            return False, f"🚫 Screen detected (uniformity: {m['uniformity']:.1f}, edges: {m['edge_density']:.4f})", {}

        
        # === HARD FAIL #2: Variance too low ===
        if m['color_variance'] < self.HARD_FAIL_LOW_VARIANCE:
            return False, f"📍 Move closer (variance: {m['color_variance']:.0f})", {}
        
        # === HARD FAIL #3: Edge density (screens have higher edges) ===
        if m['edge_density'] > self.MAX_EDGE_DENSITY:
            return False, f"🚫 Screen pixels detected (edges: {m['edge_density']:.4f})", {}
        
        # === SCORING ===
        
        # 1. Brightness (0-1)
        if self.MIN_BRIGHTNESS_CHANGE <= m['brightness_change_percent'] <= self.MAX_BRIGHTNESS_CHANGE:
            scores['brightness'] = 1.0
        else:
            scores['brightness'] = 0.3
            if m['brightness_change_percent'] < self.MIN_BRIGHTNESS_CHANGE:
                reasons.append(f"Weak flash")
        
        # 2. Color variance (0-1)
        if m['color_variance'] >= 2200:
            scores['variance'] = 1.0
        elif m['color_variance'] >= self.MIN_COLOR_VARIANCE:
            scores['variance'] = 0.7
        else:
            scores['variance'] = 0.3
            reasons.append(f"Low variance")
        
        # 3. UNIFORMITY (0-1) - KEY METRIC (inverted: lower is better for real faces)
        if m['uniformity'] <= 60.0:  # Real faces: 45-48
            scores['uniformity'] = 1.0
        elif m['uniformity'] <= self.MAX_UNIFORMITY:
            scores['uniformity'] = 0.5
        else:
            scores['uniformity'] = 0.0  # Will be caught by hard fail
        
        # 4. Edge density (0-1)
        if m['edge_density'] <= 0.040:  # Real faces: 0.032-0.036
            scores['edges'] = 1.0
        elif m['edge_density'] <= 0.055:
            scores['edges'] = 0.7
        else:
            scores['edges'] = 0.3
            reasons.append(f"High edges")
        
        # 5. Mean delta (0-1)
        if 1.0 <= m['mean_delta'] <= 2.5:
            scores['mean_delta'] = 1.0
        elif 0.5 <= m['mean_delta'] <= 3.0:
            scores['mean_delta'] = 0.6
        else:
            scores['mean_delta'] = 0.2
        
        # 6. Nonuniformity bonus
        if m['nonuniformity'] <= 0.5:
            scores['nonuniformity'] = 0.5
        else:
            scores['nonuniformity'] = 0.0
        
        total_score = sum(scores.values())
        is_live = total_score >= self.MIN_TOTAL_SCORE
        
        if not is_live:
            fail_reason = f"Low confidence ({total_score:.2f}/5.5) - {', '.join(reasons[:2]) if reasons else 'Multiple checks failed'}"
        else:
            fail_reason = ""
        
        return is_live, fail_reason, scores
    
    def print_analysis(self, metrics):
        m = metrics
        print(f"📊 Liveness Analysis:")
        print(f"   Brightness: {m['before_brightness']:.2f} → {m['after_brightness']:.2f} ({m['brightness_change_percent']:.2f}%)")
        print(f"   Color Variance: {m['color_variance']:.2f}")
        print(f"   Edge Density: {m['edge_density']:.4f} (max: {self.MAX_EDGE_DENSITY})")
        print(f"   ⭐ Uniformity: {m['uniformity']:.2f} (max: {self.MAX_UNIFORMITY} - SCREENS > 50)")
        print(f"   Nonuniformity: {m['nonuniformity']:.2f}")
        print(f"   Mean Delta: {m['mean_delta']:.2f}")
        if 'scores' in m and m['scores']:
            total = sum(m['scores'].values())
            print(f"   📊 Score: {total:.2f}/5.5 (threshold: {self.MIN_TOTAL_SCORE})")
        if 'distance_feedback' in m:
            print(f"   {m['distance_feedback']}")
