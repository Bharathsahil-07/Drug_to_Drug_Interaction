import easyocr
import os
import torch

# Initialize EasyOCR reader with GPU detection
# Enabling GPU (CUDA) if available for significant speedup
_use_gpu = torch.cuda.is_available()
print(f"[*] EasyOCR initializing. GPU: {_use_gpu}")

reader = easyocr.Reader(['en'], gpu=_use_gpu)

import cv2
import numpy as np

def extract_text(image_path):
    """
    Extract text from image using EasyOCR.
    Handles low-resolution images by upscaling and enhancement.
    """
    if not os.path.exists(image_path):
        return ""
        
    # Read image with OpenCV for potential pre-processing
    img = cv2.imread(image_path)
    if img is None:
        return ""

    # Upscale if too small (EasyOCR struggles with small text)
    height, width = img.shape[:2]
    if width < 800:
        scale = 800 / width
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale for OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Read text
    results = reader.readtext(gray, detail=0)
    
    # Fallback: Try inverting if NO text found (often helps with white-on-dark labels)
    if not results:
        inverted = cv2.bitwise_not(gray)
        results = reader.readtext(inverted, detail=0)

    # Combine results into a single string with newlines to preserve structure
    raw_text = "\n".join(results)
    
    return raw_text.strip()
