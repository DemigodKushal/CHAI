# services/liveness_detector.py
"""
Multi-layer liveness detection for anti-spoofing
Detects photos, screens, and video replay attacks
"""

import cv2
import numpy as np


class LivenessDetector:
    """Enhanced liveness detection with multiple checks"""
    
    def __init__(self):
        # Detection thresholds
        self.MIN_BRIGHTNESS_CHANGE = 2.0
        self.MAX_BRIGHTNESS_CHANGE = 8.0
        self.MIN_COLOR_VARIANCE = 300.0
        self.MAX_EDGE_DENSITY = 200.0
        self.MIN_UNIFORMITY = 15.0
        
        # New checks for screen/photo detection
        self.MAX_NONUNIFORMITY = 30.0  # Real faces have natural variance
        self.MAX_MEAN_DELTA = 6.0      # Screens show consistent changes
    
    def get_brightness(self, frame):
        """Calculate center region brightness"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        return center_region.mean()
    
    def get_color_variance(self, frame):
        """Higher variance = real 3D face with shadows"""
        b, g, r = cv2.split(frame)
        h, w = b.shape
        center_b = b[h//4:3*h//4, w//4:3*w//4]
        center_g = g[h//4:3*h//4, w//4:3*w//4]
        center_r = r[h//4:3*h//4, w//4:3*w//4]
        return np.mean([np.var(center_b), np.var(center_g), np.var(center_r)])
    
    def get_edge_density(self, frame):
        """Screens have high artificial edge density"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        edges = cv2.Canny(center_region, 50, 150)
        return np.sum(edges > 0) / edges.size
    
    def get_brightness_uniformity(self, frame):
        """Screens have more uniform brightness"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        return np.std(center_region)
    
    def get_nonuniformity(self, before_frames, after_frames):
        """
        Calculate brightness change non-uniformity
        Real faces: natural variance across face (high nonuniformity)
        Screens: uniform brightness change (low nonuniformity)
        """
        before_brightness_list = [self.get_brightness(f) for f in before_frames]
        after_brightness_list = [self.get_brightness(f) for f in after_frames]
        
        # Calculate variance of brightness changes across frames
        brightness_changes = [after - before for before, after in zip(before_brightness_list, after_brightness_list)]
        nonuniformity = np.std(brightness_changes)
        
        return nonuniformity
    
    def get_mean_delta(self, before_frames, after_frames):
        """
        Calculate mean absolute difference between consecutive frames
        Screens: very consistent changes (low delta)
        Real faces: natural micro-movements (higher delta)
        """
        deltas = []
        
        for i in range(len(before_frames) - 1):
            # Convert to grayscale
            gray1 = cv2.cvtColor(before_frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(before_frames[i+1], cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray1, gray2)
            mean_diff = diff.mean()
            deltas.append(mean_diff)
        
        return np.mean(deltas) if deltas else 0
    
    def analyze_frames(self, before_frames, after_frames):
        """
        Analyze before/after flash frames for liveness
        Returns: (is_live, metrics, fail_reason)
        """
        # Calculate metrics for all frames
        before_brightness = np.mean([self.get_brightness(f) for f in before_frames])
        after_brightness = np.mean([self.get_brightness(f) for f in after_frames])
        brightness_change = after_brightness - before_brightness
        
        before_variance = np.mean([self.get_color_variance(f) for f in before_frames])
        before_edges = np.mean([self.get_edge_density(f) for f in before_frames])
        before_uniformity = np.mean([self.get_brightness_uniformity(f) for f in before_frames])
        
        # NEW: Additional anti-spoofing checks
        nonuniformity = self.get_nonuniformity(before_frames, after_frames)
        mean_delta = self.get_mean_delta(before_frames, after_frames)
        
        # Calculate percentage change
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
        
        # Multi-layer checks
        is_live, fail_reason = self._check_liveness(metrics)
        
        return is_live, metrics, fail_reason
    
    def _check_liveness(self, metrics):
        """Perform all liveness checks"""
        
        # Check 1: Brightness change in reasonable range
        if metrics['brightness_change_percent'] < self.MIN_BRIGHTNESS_CHANGE:
            return False, f"Too low brightness change ({metrics['brightness_change_percent']:.2f}%)"
        
        if metrics['brightness_change_percent'] > self.MAX_BRIGHTNESS_CHANGE:
            return False, f"Suspiciously high brightness change ({metrics['brightness_change_percent']:.2f}%) - screen reflection"
        
        # Check 2: Color variance (screens lose texture)
        if metrics['color_variance'] < self.MIN_COLOR_VARIANCE:
            return False, f"Low color variance ({metrics['color_variance']:.0f}) - flat surface/screen"
        
        # Check 3: Edge density (screens have artificial edges)
        if metrics['edge_density'] > 1000:
            return False, f"High edge density ({metrics['edge_density']:.4f}) - screen pixels"
        
        # Check 4: Brightness uniformity (real faces have depth)
        if metrics['uniformity'] < self.MIN_UNIFORMITY:
            return False, f"Too uniform brightness ({metrics['uniformity']:.2f}) - lacks natural shadows"
        
        # Check 5: Nonuniformity (NEW - screens have uniform changes)
        if metrics['nonuniformity'] > self.MAX_NONUNIFORMITY:
            return False, f"Overly uniform brightness change ({metrics['nonuniformity']:.2f}) - screen detected"
        
        # Check 6: Mean delta (NEW - screens show consistent patterns)
        if metrics['mean_delta'] > 10000:
            return False, f"Suspicious frame consistency ({metrics['mean_delta']:.2f}) - possible replay attack"
        
        return True, ""
    
    def print_analysis(self, metrics):
        """Print detailed analysis"""
        print(f"📊 Liveness Analysis:")
        print(f"   Brightness: {metrics['before_brightness']:.2f} → {metrics['after_brightness']:.2f} ({metrics['brightness_change_percent']:.2f}%)")
        print(f"   Color Variance: {metrics['color_variance']:.2f}")
        print(f"   Edge Density: {metrics['edge_density']:.4f}")
        print(f"   Brightness Uniformity: {metrics['uniformity']:.2f}")
        print(f"   Nonuniformity: {metrics['nonuniformity']:.2f}")
        print(f"   Mean Delta: {metrics['mean_delta']:.2f}")
        print(f"   Thresholds:")
        print(f"     • Brightness: {self.MIN_BRIGHTNESS_CHANGE}%-{self.MAX_BRIGHTNESS_CHANGE}%")
        print(f"     • Variance: >{self.MIN_COLOR_VARIANCE}")
        print(f"     • Edges: <{self.MAX_EDGE_DENSITY}")
        print(f"     • Uniformity: >{self.MIN_UNIFORMITY}")
        print(f"     • Nonuniformity: <{self.MAX_NONUNIFORMITY}")
        print(f"     • Mean Delta: <{self.MAX_MEAN_DELTA}")
