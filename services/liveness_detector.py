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
        # PRIMARY CHECK: Brightness change from flash
        self.MIN_BRIGHTNESS_CHANGE = 3.0   # Real faces: 3-15%
        self.MAX_BRIGHTNESS_CHANGE = 20.0  # Screens reflect more light
        
        # TEXTURE CHECKS: Real faces have natural texture
        self.MIN_COLOR_VARIANCE = 200.0    # Relaxed for real faces with shadows
        self.MIN_TEXTURE_SCORE = 100.0     # LBP texture score
        
        # SCREEN DETECTION: Screens have artificial patterns
        self.MAX_SCREEN_PATTERN_SCORE = 0.7  # Moiré patterns, pixel grid
        self.MAX_EDGE_SHARPNESS = 0.15       # Screens have sharp digital edges
        
        # DEPTH CHECKS: Real faces are 3D
        self.MIN_BRIGHTNESS_STD = 12.0     # Real faces have depth variation
    
    def get_brightness(self, frame):
        """Calculate center region brightness"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        return center_region.mean()
    
    def get_brightness_std(self, frame):
        """Calculate brightness standard deviation (depth indicator)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        return np.std(center_region)
    
    def get_color_variance(self, frame):
        """Higher variance = real 3D face with shadows"""
        b, g, r = cv2.split(frame)
        h, w = b.shape
        center_b = b[h//4:3*h//4, w//4:3*w//4]
        center_g = g[h//4:3*h//4, w//4:3*w//4]
        center_r = r[h//4:3*h//4, w//4:3*w//4]
        return np.mean([np.var(center_b), np.var(center_g), np.var(center_r)])
    
    def get_texture_score(self, frame):
        """
        Calculate Local Binary Pattern (LBP) texture score
        Real skin has natural texture, screens/photos have artificial patterns
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        
        # Simple LBP approximation using gradient variance
        grad_x = cv2.Sobel(center_region, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(center_region, cv2.CV_64F, 0, 1, ksize=3)
        
        # Texture score = variance of gradient magnitudes
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.var(magnitude)
    
    def detect_screen_patterns(self, frame):
        """
        Detect moiré patterns and pixel grid artifacts from screens
        Returns score: 0 = no pattern, 1 = strong screen pattern
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        
        # Apply FFT to detect periodic patterns
        f_transform = np.fft.fft2(center_region)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Remove DC component (center)
        center_h, center_w = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        magnitude_spectrum[center_h-5:center_h+5, center_w-5:center_w+5] = 0
        
        # High frequency peaks indicate screen pixel patterns
        max_magnitude = np.max(magnitude_spectrum)
        mean_magnitude = np.mean(magnitude_spectrum)
        
        # Normalize: strong peaks = screen, weak peaks = natural texture
        pattern_score = (max_magnitude / (mean_magnitude + 1e-6)) / 1000.0
        return min(pattern_score, 1.0)
    
    def get_edge_sharpness(self, frame):
        """
        Measure edge sharpness - screens have artificially sharp edges
        Real faces have softer, natural edges
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        
        # Laplacian variance = sharpness metric
        laplacian = cv2.Laplacian(center_region, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 scale
        normalized_sharpness = min(variance / 1000.0, 1.0)
        return normalized_sharpness
    
    def analyze_frames(self, before_frames, after_frames):
        """
        Analyze before/after flash frames for liveness
        Returns: (is_live, metrics, fail_reason)
        """
        # Calculate brightness change (PRIMARY CHECK)
        before_brightness = np.mean([self.get_brightness(f) for f in before_frames])
        after_brightness = np.mean([self.get_brightness(f) for f in after_frames])
        brightness_change = after_brightness - before_brightness
        brightness_change_percent = (brightness_change / before_brightness * 100) if before_brightness > 0 else 0
        
        # Texture and depth checks
        brightness_std = np.mean([self.get_brightness_std(f) for f in before_frames])
        color_variance = np.mean([self.get_color_variance(f) for f in before_frames])
        texture_score = np.mean([self.get_texture_score(f) for f in before_frames])
        
        # Screen detection checks
        screen_pattern = np.mean([self.detect_screen_patterns(f) for f in before_frames])
        edge_sharpness = np.mean([self.get_edge_sharpness(f) for f in before_frames])
        
        metrics = {
            'brightness_change_percent': brightness_change_percent,
            'brightness_std': brightness_std,
            'color_variance': color_variance,
            'texture_score': texture_score,
            'screen_pattern_score': screen_pattern,
            'edge_sharpness': edge_sharpness
        }
        
        # Multi-layer checks
        is_live, fail_reason = self._check_liveness(metrics)
        
        return is_live, metrics, fail_reason
    
    def _check_liveness(self, metrics):
        """Perform all liveness checks with weighted scoring"""
        
        # CRITICAL CHECK: Brightness change from flash
        if metrics['brightness_change_percent'] < self.MIN_BRIGHTNESS_CHANGE:
            return False, f"No flash response ({metrics['brightness_change_percent']:.1f}% < {self.MIN_BRIGHTNESS_CHANGE}%)"
        
        if metrics['brightness_change_percent'] > self.MAX_BRIGHTNESS_CHANGE:
            return False, f"Screen reflection detected ({metrics['brightness_change_percent']:.1f}% > {self.MAX_BRIGHTNESS_CHANGE}%)"
        
        # SCREEN PATTERN CHECK (MOST RELIABLE FOR DETECTING PHONES)
        if metrics['screen_pattern_score'] > self.MAX_SCREEN_PATTERN_SCORE:
            return False, f"Screen pixel pattern detected (score: {metrics['screen_pattern_score']:.2f})"
        
        # EDGE SHARPNESS CHECK
        if metrics['edge_sharpness'] > self.MAX_EDGE_SHARPNESS:
            return False, f"Artificial edges detected (sharpness: {metrics['edge_sharpness']:.2f})"
        
        # DEPTH CHECK: Real faces have shadow variations
        if metrics['brightness_std'] < self.MIN_BRIGHTNESS_STD:
            return False, f"Flat surface detected (std: {metrics['brightness_std']:.1f})"
        
        # TEXTURE CHECK: Real skin has natural texture
        if metrics['texture_score'] < self.MIN_TEXTURE_SCORE:
            return False, f"Artificial texture (score: {metrics['texture_score']:.1f})"
        
        # COLOR VARIANCE CHECK
        if metrics['color_variance'] < self.MIN_COLOR_VARIANCE:
            return False, f"Low color depth (variance: {metrics['color_variance']:.0f})"
        
        # ALL CHECKS PASSED
        return True, "✅ Live person detected"
    
    def print_analysis(self, metrics):
        """Print detailed analysis"""
        print(f"\n📊 Liveness Analysis:")
        print(f"   ✓ Flash Response: {metrics['brightness_change_percent']:.1f}%")
        print(f"   ✓ Depth Variation: {metrics['brightness_std']:.1f}")
        print(f"   ✓ Color Variance: {metrics['color_variance']:.0f}")
        print(f"   ✓ Skin Texture: {metrics['texture_score']:.1f}")
        print(f"   ✓ Screen Pattern: {metrics['screen_pattern_score']:.2f}")
        print(f"   ✓ Edge Sharpness: {metrics['edge_sharpness']:.2f}")
        print(f"\n   Thresholds:")
        print(f"     • Flash: {self.MIN_BRIGHTNESS_CHANGE}-{self.MAX_BRIGHTNESS_CHANGE}%")
        print(f"     • Screen Pattern: <{self.MAX_SCREEN_PATTERN_SCORE}")
        print(f"     • Edge Sharpness: <{self.MAX_EDGE_SHARPNESS}")
        print(f"     • Depth STD: >{self.MIN_BRIGHTNESS_STD}")
        print(f"     • Texture: >{self.MIN_TEXTURE_SCORE}")
        print(f"     • Color Variance: >{self.MIN_COLOR_VARIANCE}")
