import cv2
import numpy as np

# --- Physical measurements ---
actual_hole_diameter = 4.25
actual_ball_diameter = 1.68
focal_length = 4000  # calibrate this for accuracy!

# --- Slope function (can customize for physics later) ---
def slope_func(x, y):
    return y / (x + 1e-5)

def draw_slope_field(img, roi, step=30, length=15):
    h, w, _ = img.shape
    overlay = img.copy()

    x0, y0, x1, y1 = roi  # ROI rectangle

    for y in range(y0, y1, step):
        for x in range(x0, x1, step):
            X = (x - w // 2) / 50
            Y = (h // 2 - y) / 50
            m = slope_func(X, Y)
            m = np.clip(m, -10, 10)

            dx, dy = 1, m
            norm = np.sqrt(dx**2 + dy**2)
            dx, dy = dx / norm, dy / norm

            x1l = int(x - length * dx)
            y1l = int(y + length * dy)
            x2l = int(x + length * dx)
            y2l = int(y - length * dy)

            color_val = int(255 * (np.clip(abs(m), 0, 5) / 5))
            color = (color_val, 255 - color_val, 0)

            cv2.line(overlay, (x1l, y1l), (x2l, y2l), color, 1)

            # Tiny arrowheads
            angle = np.arctan2(-dy, dx)
            tip_len = 5
            for a in (angle + np.pi * 0.75, angle - np.pi * 0.75):
                xt = int(x2l - tip_len * np.cos(a))
                yt = int(y2l + tip_len * np.sin(a))
                cv2.line(overlay, (x2l, y2l), (xt, yt), color, 1)

    return overlay

def detect_golf_ball(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    rect_area = w * h
    extent = float(area) / rect_area
    circularity = 0 if perimeter == 0 else 4 * np.pi * (area / (perimeter * perimeter))

    if area > 300 and aspect_ratio < 1.2 and extent > 0.5 and circularity > 0.75:
        equivalent_diameter = np.sqrt(4 * area / np.pi)
        return True, (x, y, w, h), equivalent_diameter
    return False, None, None

def is_contour_a_hole(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    rect_area = w * h
    extent = float(area) / rect_area
    circularity = 0 if perimeter == 0 else 4 * np.pi * (area / (perimeter * perimeter))

    if area > 450 and aspect_ratio < 1.2 and extent > 0.5 and circularity > 0.7:
        equivalent_diameter = np.sqrt(4 * area / np.pi)
        return True, (x, y, w, h), equivalent_diameter
    return False, None, None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, white_thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        _, dark_thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

        white_contours, _ = cv2.findContours(white_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dark_contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ball_found, hole_found = False, False
        ball_rect, hole_rect = None, None

        for c in white_contours:
            ball_found, ball_rect, ball_diam = detect_golf_ball(c)
            if ball_found:
                cv2.drawContours(frame, [c], -1, (255, 0, 0), 2)
                bx, by, bw, bh = ball_rect
                cv2.putText(frame, "Ball", (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                break

        for c in dark_contours:
            hole_found, hole_rect, hole_diam = is_contour_a_hole(c)
            if hole_found:
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                hx, hy, hw, hh = hole_rect
                cv2.putText(frame, "Hole", (hx, hy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                break

        # Auto-select ROI if both found
        if ball_found and hole_found:
            bx, by, bw, bh = ball_rect
            hx, hy, hw, hh = hole_rect
            x0 = min(bx, hx) - 20
            y0 = min(by, hy) - 20
            x1 = max(bx + bw, hx + hw) + 20
            y1 = max(by + bh, hy + hh) + 20

            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(frame.shape[1], x1), min(frame.shape[0], y1)

            slope_overlay = draw_slope_field(frame, (x0, y0, x1, y1))
            cv2.imshow("Slope Field", slope_overlay)
        else:
            cv2.imshow("Slope Field", frame)

        cv2.imshow("Ball/Hole Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
