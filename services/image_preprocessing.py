import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess image for better OCR results.
    1. Convert to grayscale
    2. Resize
    3. Increase contrast
    4. Apply adaptive threshold
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to fixed width while maintaining aspect ratio
    width = 1000
    height = int(gray.shape[0] * (width / gray.shape[1]))
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LANCZOS4)
    
    # Increase contrast (Alpha for contrast [1.0-3.0], Beta for brightness [0-100])
    alpha = 1.5
    beta = 10
    adjusted = cv2.convertScaleAbs(resized, alpha=alpha, beta=beta)
    
    # Sharpen image to enhance text edges
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(adjusted, -1, kernel)
    
    # Apply adaptive thresholding with larger block size for better character preservation
    threshold = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 21, 5
    )
    
    # Save processed image to a temporary file
    processed_path = image_path.replace('.', '_processed.')
    cv2.imwrite(processed_path, threshold)
    
    return processed_path
