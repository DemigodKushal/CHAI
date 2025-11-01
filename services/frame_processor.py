# services/frame_processor.py
"""Frame processing utilities for base64 decoding"""

import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image


class FrameProcessor:
    """Process base64 encoded frames from browser"""
    
    @staticmethod
    def decode_frame(b64_str):
        """Decode base64 string to OpenCV frame"""
        img_data = b64_str.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def decode_frames_batch(b64_strings):
        """Decode multiple base64 frames"""
        return [FrameProcessor.decode_frame(f) for f in b64_strings]
