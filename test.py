# test_liveness_calibration.py
"""
Liveness Detection Calibration Tool
Run multiple tests to analyze metrics and optimize thresholds
"""

import cv2
import numpy as np
import json
from datetime import datetime
from services.liveness_detector import LivenessDetector
from services.frame_processor import FrameProcessor

class LivenessCalibration:
    def __init__(self):
        self.detector = LivenessDetector()
        self.frame_processor = FrameProcessor()
        self.results = []
        
    def capture_test_frames(self, label, num_before=5, num_after=5):
        """
        Capture frames for testing
        Args:
            label: Test label (e.g., "real_face", "phone_screen", "photo")
        """
        print(f"\n{'='*60}")
        print(f"📸 TEST: {label.upper()}")
        print(f"{'='*60}")
        print("⏳ Preparing camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return None
        
        print("✅ Camera ready!")
        print(f"\n🎬 Instructions:")
        print(f"   1. Position the {label} in frame")
        print(f"   2. Press SPACE to start capture")
        print(f"   3. Look directly at camera")
        print(f"   4. Screen will flash WHITE")
        print(f"   5. Keep still until complete")
        
        # Preview window
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.putText(frame, f"Test: {label}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to start", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Liveness Calibration', frame)
            
            key = cv2.waitKey(1)
            if key == 32:  # SPACE
                break
            elif key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return None
        
        # Capture BEFORE frames
        print(f"\n📸 Capturing {num_before} BEFORE frames...")
        before_frames = []
        for i in range(num_before):
            ret, frame = cap.read()
            if ret:
                before_frames.append(frame.copy())
                cv2.putText(frame, f"BEFORE {i+1}/{num_before}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Liveness Calibration', frame)
                cv2.waitKey(50)
        
        # FLASH
        print("⚡ FLASHING...")
        flash_frame = np.ones_like(frame) * 255
        cv2.imshow('Liveness Calibration', flash_frame)
        cv2.waitKey(200)
        
        # Capture AFTER frames
        print(f"📸 Capturing {num_after} AFTER frames...")
        after_frames = []
        for i in range(num_after):
            ret, frame = cap.read()
            if ret:
                after_frames.append(frame.copy())
                cv2.putText(frame, f"AFTER {i+1}/{num_after}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow('Liveness Calibration', frame)
                cv2.waitKey(50)
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("✅ Capture complete!")
        return before_frames, after_frames
    
    def analyze_and_log(self, label, before_frames, after_frames):
        """Analyze frames and log results"""
        print(f"\n🔍 Analyzing {label}...")
        
        is_live, metrics, fail_reason = self.detector.analyze_frames(before_frames, after_frames)
        
        # Log result
        result = {
            'timestamp': datetime.now().isoformat(),
            'label': label,
            'is_live': is_live,
            'fail_reason': fail_reason,
            'metrics': {
                'before_brightness': float(metrics['before_brightness']),
                'after_brightness': float(metrics['after_brightness']),
                'brightness_change_percent': float(metrics['brightness_change_percent']),
                'color_variance': float(metrics['color_variance']),
                'edge_density': float(metrics['edge_density']),
                'uniformity': float(metrics['uniformity']),
                'nonuniformity': float(metrics['nonuniformity']),
                'mean_delta': float(metrics['mean_delta'])
            }
        }
        
        self.results.append(result)
        
        # Print analysis
        self.detector.print_analysis(metrics)
        
        if is_live:
            print(f"\n✅ Result: LIVE")
        else:
            print(f"\n❌ Result: SPOOF")
            print(f"   Reason: {fail_reason}")
        
        return result
    
    def run_test(self, label):
        """Run a single test"""
        frames = self.capture_test_frames(label)
        if frames:
            before_frames, after_frames = frames
            return self.analyze_and_log(label, before_frames, after_frames)
        return None
    
    def generate_report(self, filename='liveness_calibration_results.json'):
        """Generate comprehensive calibration report"""
        print(f"\n{'='*60}")
        print("📊 CALIBRATION REPORT")
        print(f"{'='*60}")
        
        if not self.results:
            print("❌ No results to report")
            return
        
        # Group by label
        grouped = {}
        for result in self.results:
            label = result['label']
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(result)
        
        # Print summary table
        print(f"\n{'Label':<20} {'Count':<8} {'Live':<8} {'Spoof':<8}")
        print("-" * 60)
        
        for label, results in grouped.items():
            live_count = sum(1 for r in results if r['is_live'])
            spoof_count = len(results) - live_count
            print(f"{label:<20} {len(results):<8} {live_count:<8} {spoof_count:<8}")
        
        # Print metric ranges
        print(f"\n📊 METRIC RANGES:")
        print("-" * 60)
        
        metrics_summary = {}
        for metric in ['brightness_change_percent', 'color_variance', 'edge_density', 
                      'uniformity', 'nonuniformity', 'mean_delta']:
            values = [r['metrics'][metric] for r in self.results]
            metrics_summary[metric] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
            
            print(f"\n{metric}:")
            print(f"  Min:  {metrics_summary[metric]['min']:.2f}")
            print(f"  Max:  {metrics_summary[metric]['max']:.2f}")
            print(f"  Mean: {metrics_summary[metric]['mean']:.2f}")
            print(f"  Std:  {metrics_summary[metric]['std']:.2f}")
        
        # Recommended thresholds
        print(f"\n💡 RECOMMENDED THRESHOLDS:")
        print("-" * 60)
        
        # Group results by type
        real_faces = [r for r in self.results if 'real' in r['label'].lower()]
        spoofs = [r for r in self.results if 'real' not in r['label'].lower()]
        
        if real_faces and spoofs:
            print("\nBased on your test data:")
            
            # Brightness change
            real_brightness = [r['metrics']['brightness_change_percent'] for r in real_faces]
            spoof_brightness = [r['metrics']['brightness_change_percent'] for r in spoofs]
            print(f"\nBrightness Change %:")
            print(f"  Real faces: {min(real_brightness):.2f} - {max(real_brightness):.2f}")
            print(f"  Spoofs:     {min(spoof_brightness):.2f} - {max(spoof_brightness):.2f}")
            print(f"  → MIN_BRIGHTNESS_CHANGE = {max(min(real_brightness) - 0.5, 1.5):.1f}")
            print(f"  → MAX_BRIGHTNESS_CHANGE = {min(max(real_brightness) + 1.0, 10.0):.1f}")
            
            # Color variance
            real_variance = [r['metrics']['color_variance'] for r in real_faces]
            spoof_variance = [r['metrics']['color_variance'] for r in spoofs]
            print(f"\nColor Variance:")
            print(f"  Real faces: {min(real_variance):.0f} - {max(real_variance):.0f}")
            print(f"  Spoofs:     {min(spoof_variance):.0f} - {max(spoof_variance):.0f}")
            print(f"  → MIN_COLOR_VARIANCE = {max(min(real_variance) - 50, 150):.0f}")
            
            # Similar for other metrics...
        
        # Save to JSON
        with open(filename, 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': len(self.results),
                    'by_label': {label: len(results) for label, results in grouped.items()}
                },
                'metrics_summary': metrics_summary,
                'detailed_results': self.results
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        print(f"{'='*60}\n")


def main():
    """Interactive calibration session"""
    calibrator = LivenessCalibration()
    
    print(f"\n{'='*60}")
    print("🔬 LIVENESS DETECTION CALIBRATION TOOL")
    print(f"{'='*60}")
    print("\nThis tool helps you calibrate liveness detection thresholds")
    print("by testing different scenarios and analyzing the metrics.")
    print("\nRecommended test sequence:")
    print("  1. real_face_good_lighting")
    print("  2. real_face_poor_lighting")
    print("  3. phone_screen")
    print("  4. laptop_screen")
    print("  5. printed_photo")
    print("  6. tablet_screen")
    
    while True:
        print(f"\n{'='*60}")
        print("OPTIONS:")
        print("  1. Run new test")
        print("  2. Generate report")
        print("  3. Exit")
        print(f"{'='*60}")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            label = input("\nEnter test label (e.g., 'real_face', 'phone_screen'): ").strip()
            if label:
                calibrator.run_test(label)
        
        elif choice == '2':
            filename = input("\nEnter filename (default: liveness_calibration_results.json): ").strip()
            if not filename:
                filename = 'liveness_calibration_results.json'
            calibrator.generate_report(filename)
        
        elif choice == '3':
            print("\n👋 Exiting calibration tool")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == '__main__':
    main()
