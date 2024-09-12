import cv2
import numpy as np

# The actual diameter of the hole (in inches)
actual_hole_diameter = 4.25

# The actual diameter of the ball (in inches)
actual_ball_diameter = 1.68

# Focal length in pixels of an iPhone 13 (you might need to adjust this based on actual tests)
focal_length = 4000


def detect_green(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return mask


def analyze_slope(mask):
    sobelx = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    gradient_direction = np.arctan2(sobely, sobelx) * (180 / np.pi)
    return gradient_magnitude, gradient_direction


def display_slope(frame, gradient_magnitude, gradient_direction):
    magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_normalized = np.uint8(magnitude_normalized)
    direction_normalized = cv2.normalize(gradient_direction, None, 0, 255, cv2.NORM_MINMAX)
    direction_normalized = np.uint8(direction_normalized)

    magnitude_colormap = cv2.applyColorMap(magnitude_normalized, cv2.COLORMAP_JET)
    direction_colormap = cv2.applyColorMap(direction_normalized, cv2.COLORMAP_HSV)

    combined = cv2.addWeighted(frame, 0.7, magnitude_colormap, 0.3, 0)

    return combined, magnitude_colormap, direction_colormap


def detect_golf_ball(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    rect_area = w * h
    extent = float(area) / rect_area
    if perimeter == 0:
        circularity = 0
    else:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

    aspect_ratio_threshold = 1.2
    extent_threshold = 0.5
    circularity_threshold = 0.75

    if area > 300 and aspect_ratio < aspect_ratio_threshold and extent > extent_threshold and circularity > circularity_threshold:
        equivalent_diameter = np.sqrt(4 * area / np.pi)
        return True, area, (x, y, w, h), equivalent_diameter
    else:
        return False, None, None, None


def is_contour_a_hole(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    rect_area = w * h
    extent = float(area) / rect_area
    if perimeter == 0:
        circularity = 0
    else:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

    aspect_ratio_threshold = 1.2
    extent_threshold = 0.5
    circularity_threshold = 0.7

    if area > 450 and aspect_ratio < aspect_ratio_threshold and extent > extent_threshold and circularity > circularity_threshold:
        equivalent_diameter = np.sqrt(4 * area / np.pi)
        return True, area, (x, y, w, h), equivalent_diameter
    else:
        return False, None, None, None


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    global hole_contour, ball_contour
    hole_contour = None
    ball_contour = None

    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    _, white_thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    _, dark_thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    white_contours, _ = cv2.findContours(white_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dark_contours, _ = cv2.findContours(dark_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    is_ball, ball_area, ball_rect, equivalent_ball_diameter = False, None, (0, 0, 0, 0), 0
    is_hole, area, hole_rect, equivalent_diameter = False, None, (0, 0, 0, 0), 0

    for contour in white_contours:
        is_ball, ball_area, ball_rect, equivalent_ball_diameter = detect_golf_ball(contour)
        if is_ball:
            ball_contour = contour
            break

    for contour in dark_contours:
        is_hole, area, hole_rect, equivalent_diameter = is_contour_a_hole(contour)
        if is_hole:
            hole_contour = contour
            break

    if is_hole:
        x, y, w, h = hole_rect
        cv2.drawContours(frame, [hole_contour], -1, (0, 255, 0), 2)
        perceived_diameter = equivalent_diameter
        distance_to_hole = (actual_hole_diameter * focal_length) / perceived_diameter

        if is_ball:
            ball_x, ball_y, ball_w, ball_h = ball_rect
            distance_to_ball = (actual_ball_diameter * focal_length) / equivalent_ball_diameter
            distance_to_hole -= distance_to_ball

        cv2.putText(frame, f"Dist to Hole: {distance_to_hole:.2f} units", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    if is_ball:
        ball_x, ball_y, ball_w, ball_h = ball_rect
        cv2.drawContours(frame, [ball_contour], -1, (255, 0, 0), 2)
        perceived_ball_diameter = equivalent_ball_diameter
        distance_to_ball = (actual_ball_diameter * focal_length) / perceived_ball_diameter
        cv2.putText(frame, f"Dist to Ball: {distance_to_ball:.2f} units", (ball_x, ball_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow('Hole and Ball Detection', frame)
    mask = detect_green(frame)
    gradient_magnitude, gradient_direction = analyze_slope(mask)

    combined, magnitude_colormap, direction_colormap = display_slope(frame, gradient_magnitude,
                                                                             gradient_direction)

    # Display the results
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Green Mask', mask)
    cv2.imshow('Slope Magnitude', magnitude_colormap)
    cv2.imshow('Slope Direction', direction_colormap)
    cv2.imshow('Combined', combined)

    cv2.imshow('Hole and Ball Detection', frame)
    cv2.waitKey(0)  # Wait for any key to be pressed
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
