import cv2
import numpy as np

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
            cv2.putText(frame, 'Tomato', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow(f"Tomato Detection {image_path}", frame)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_red_tomatoes("image10.jpg")