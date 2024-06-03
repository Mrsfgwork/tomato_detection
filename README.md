# Tomato Detection using OpenCV

## Introduction

This project aims to detect red tomatoes in a series of images using computer vision techniques with OpenCV. The code reads an image, converts it to the HSV color space, applies color masking to identify red regions, processes these regions to remove noise, and finally detects and highlights the tomatoes in the image by drawing bounding boxes around them. The results are displayed in a resizable window with a fixed size of 1080x720 pixels.

## Prerequisites

To run this project, you need to have the following libraries installed:

- OpenCV: `pip install opencv-python`
- NumPy: `pip install numpy`

Ensure you have Python 3.x installed.

## Code Explanation

### Libraries

```python
import cv2
import numpy as np
```

We import OpenCV and NumPy libraries for image processing and numerical operations.

### Function: `detect_red_tomatoes`

```python
def detect_red_tomatoes(image_path):
    frame = cv2.imread(image_path)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    blurred = cv2.medianBlur(red_mask, 15)

    kernel = np.ones((5,5),np.uint8)
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 500:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'tomato', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.namedWindow(f"Tomato Detection {image_path}", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"Tomato Detection {image_path}", 1080, 720)
    cv2.imshow(f"Tomato Detection {image_path}", frame)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

#### Steps:

1. **Reading the Image**:
   - The image is read from the specified path using `cv2.imread(image_path)`.

2. **Color Space Conversion**:
   - The image is converted from BGR to HSV color space using `cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)`. HSV color space is more suitable for color-based segmentation.

3. **Color Masking**:
   - We define the lower and upper bounds for the red color in HSV space and create a mask using `cv2.inRange(hsv_frame, lower_red, upper_red)`. This mask isolates the red regions in the image.

4. **Noise Reduction**:
   - A median blur is applied to the mask using `cv2.medianBlur(red_mask, 15)` to reduce noise and smooth the image.
   - Morphological opening is performed using `cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)` with a 5x5 kernel to remove small objects from the foreground.

5. **Contour Detection**:
   - Contours are detected in the processed mask using `cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`.

6. **Bounding Boxes**:
   - For each contour, a bounding rectangle is computed using `cv2.boundingRect(contour)`.
   - If the bounding box area is greater than 500 pixels, a rectangle is drawn around the detected tomato using `cv2.rectangle`, and a label is added using `cv2.putText`.

7. **Displaying the Result**:
   - A window is created and resized to 1080x720 pixels using `cv2.namedWindow` and `cv2.resizeWindow`.
   - The processed image is displayed using `cv2.imshow`, and the window waits for a key press to close using `cv2.waitKey(0)` and `cv2.destroyAllWindows()`.

### Main Execution

```python
for x in range(1,11):
    detect_red_tomatoes(F'image{x}.jpg')
```

This loop iterates over image filenames from `image1.jpg` to `image10.jpg` and applies the `detect_red_tomatoes` function to each image.

## Usage

1. Place your images in the same directory as the script and name them `image1.jpg`, `image2.jpg`, ..., `image10.jpg`.
2. Run the script using Python:

   ```bash
   python tomato_detection.py
   ```

3. The script will open a window for each image, displaying the detected tomatoes with bounding boxes and labels.

## Conclusion

This project provides a basic yet effective method for detecting red tomatoes in images using color segmentation and contour detection techniques with OpenCV. For more accurate and robust results, especially in varying lighting conditions and backgrounds, advanced methods like deep learning-based object detection models (e.g., YOLO, SSD, Faster R-CNN) can be employed.

## Future Work

- Implement advanced object detection algorithms for better accuracy.
- Add support for detecting tomatoes of different ripeness stages (e.g., green, yellow).
- Optimize the code for real-time video processing.
- Integrate with a graphical user interface (GUI) for easier interaction and visualization.

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)

---

This `README.md` provides a comprehensive overview of the project, detailed steps for understanding the code, and instructions for usage.
