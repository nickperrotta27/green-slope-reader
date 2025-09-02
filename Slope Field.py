import cv2
import numpy as np

# Example slope function
def slope_func(x, y):
    return y / (x + 1e-5)

def draw_slope_field(img, roi, step=30, length=15):
    h, w, _ = img.shape
    overlay = img.copy()

    x0, y0, x1, y1 = roi  # ROI rectangle

    for y in range(y0, y1, step):
        for x in range(x0, x1, step):
            # Convert to math coords
            X = (x - w // 2) / 50
            Y = (h // 2 - y) / 50

            m = slope_func(X, Y)

            dx, dy = 1, m
            norm = np.sqrt(dx**2 + dy**2)
            dx, dy = dx / norm, dy / norm

            x1l = int(x - length * dx)
            y1l = int(y + length * dy)
            x2l = int(x + length * dx)
            y2l = int(y - length * dy)

            cv2.line(overlay, (x1l, y1l), (x2l, y2l), (0, 255, 0), 1)

    return overlay

# --- Mouse ROI selection ---
roi_start = None
roi_end = None
drawing = False
snapshot = None
preview = None

def mouse_callback(event, x, y, flags, param):
    global roi_start, roi_end, drawing, snapshot, preview

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        roi_end = (x, y)
        drawing = True
        preview = snapshot.copy()

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi_end = (x, y)
        preview = snapshot.copy()
        cv2.rectangle(preview, roi_start, roi_end, (0, 0, 255), 2)
        cv2.imshow("Snapshot - Select Ground", preview)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (x, y)
        drawing = False
        x0, y0 = roi_start
        x1, y1 = roi_end
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)

        slope_img = draw_slope_field(snapshot, (x0, y0, x1, y1))
        cv2.imshow("Slope Field", slope_img)

def main():
    global snapshot
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # take snapshot
            snapshot = frame.copy()
            cv2.imshow("Snapshot - Select Ground", snapshot)
            cv2.setMouseCallback("Snapshot - Select Ground", mouse_callback)

        elif key == ord('q'):  # quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
