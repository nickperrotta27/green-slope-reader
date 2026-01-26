import os
import cv2
import numpy as np
import onnxruntime as ort

# ------------------------
# CONFIG
# ------------------------
MIDAS_MODEL_PATH = "midas_v21_small_256.onnx"  # path to your MiDaS ONNX model
CALIBRATION_FILE = "calibration_points.npy"
# Homography: will be interactively set (TL, TR, BR, BL)
TOPDOWN_SIZE = (1000, 600)  # restored to your original size (width, height)
SLOPE_SCALE = 1.0

# Putting speed settings (pixels per frame in simulation)
# These represent different putting speeds: soft, medium, firm
PUTTING_SPEEDS = {
    'soft': 2.0,  # Slow putt - breaks more
    'medium': 4.0,  # Normal putt
    'firm': 7.0,  # Fast putt - breaks less
    'lag': 3.0  # Lag putt (trying to get close)
}
DEFAULT_SPEED = 'medium'

# Physics parameters
FRICTION_COEFF = 0.12  # Deceleration factor (higher = more friction)
GRAVITY = 9.81  # meters/sec^2
PIXELS_PER_METER = 100  # Calibration: pixels to real-world distance


# ------------------------
# MiDaS ONNX helpers
# ------------------------
def load_midas_onnx(path):
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    inp_name = sess.get_inputs()[0].name
    return sess, inp_name


def remove_shadows(img_bgr):
    """
    Reduce shadow effects using illumination normalization.
    Combines multiple techniques for robust shadow mitigation:
    1. LAB color space CLAHE for illumination normalization
    2. Bilateral filtering to preserve edges while smoothing
    3. Optional: Intrinsic image decomposition approach
    """
    # Convert to LAB color space (separates lightness from color)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    # This normalizes local illumination variations (shadows/highlights)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_equalized = clahe.apply(l)

    # Merge channels back
    lab_equalized = cv2.merge([l_equalized, a, b])

    # Convert back to BGR
    result = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)

    # Bilateral filter: smooths while preserving edges
    # This reduces noise from equalization without blurring actual slope features
    result = cv2.bilateralFilter(result, 9, 75, 75)

    return result


def predict_depth_onnx(sess, inp_name, img_bgr):
    # small MiDaS expects 256x256 for many versions; keep consistent with your model file
    target_w, target_h = 256, 256

    # SHADOW REMOVAL: preprocess image to reduce shadow effects
    img_no_shadows = remove_shadows(img_bgr)

    img = cv2.cvtColor(img_no_shadows, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
    pred = sess.run(None, {inp_name: img})[0]
    pred = np.squeeze(pred)
    pred = cv2.resize(pred, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return pred


# ------------------------
# Calibration helpers
# ------------------------
def select_points(img, n=4, win="Select 4 corners (TL,TR,BR,BL)"):
    pts = []
    display = img.copy()

    def click_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(display, str(len(pts)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(win, display)

    cv2.imshow(win, display)
    cv2.setMouseCallback(win, click_cb)
    print("Click the 4 corners in order: top-left, top-right, bottom-right, bottom-left.")
    while len(pts) < n:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(win)
    return np.array(pts, dtype=np.float32)


def calibrate_with_prompt(frame):
    # If file exists, allow user to press 'r' to recalibrate
    if os.path.exists(CALIBRATION_FILE):
        pts = np.load(CALIBRATION_FILE)
        print(f"Loaded calibration from {CALIBRATION_FILE}.")
        cv2.imshow("Calibration loaded - press 'r' to recalibrate or any other key to continue", frame)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key != ord('r'):
            return pts
        print("Recalibration chosen.")
    # interactive selection
    pts_new = select_points(frame)
    if pts_new.shape[0] != 4:
        raise RuntimeError("Calibration aborted or incomplete (4 points required).")
    np.save(CALIBRATION_FILE, pts_new)
    print(f"Saved calibration points to {CALIBRATION_FILE}")
    return pts_new


# ------------------------
# Warp / gradient / visualization
# ------------------------
def warp_to_topdown(img_or_map, H, size=TOPDOWN_SIZE):
    # works with 1- or 3-channel arrays
    return cv2.warpPerspective(img_or_map, H, size, flags=cv2.INTER_CUBIC)


def compute_gradient_field(depth_topdown):
    # depth_topdown: 2D float array
    # Apply Gaussian blur to reduce noise in gradient computation
    depth_smooth = cv2.GaussianBlur(depth_topdown, (5, 5), 1.0)

    dzdx = cv2.Sobel(depth_smooth, cv2.CV_64F, 1, 0, ksize=3)
    dzdy = cv2.Sobel(depth_smooth, cv2.CV_64F, 0, 1, ksize=3)
    vx = -SLOPE_SCALE * dzdx
    vy = -SLOPE_SCALE * dzdy
    mag = np.sqrt(vx * vx + vy * vy) + 1e-12
    nx = vx / mag
    ny = vy / mag
    return nx, ny, mag


def draw_vector_field(topdown_rgb, nx, ny, mag, step=25, length=15):
    h, w = topdown_rgb.shape[:2]
    vis = topdown_rgb.copy()
    # percentile for coloring
    clip_val = np.percentile(mag, 95)
    if clip_val <= 0:
        clip_val = 1.0
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx = nx[y, x]
            dy = ny[y, x]
            m = mag[y, x]
            x1 = int(x - length * dx)
            y1 = int(y - length * dy)
            x2 = int(x + length * dx)
            y2 = int(y + length * dy)
            cval = int(255 * np.clip(m / clip_val, 0, 1))
            color = (cval, 255 - cval, 0)
            cv2.line(vis, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            # small arrow head
            angle = np.arctan2(-dy, dx)
            tip = 4
            for a in (angle + np.pi * 0.75, angle - np.pi * 0.75):
                xt = int(x2 - tip * np.cos(a))
                yt = int(y2 + tip * np.sin(a))
                cv2.line(vis, (x2, y2), (xt, yt), color, 1, cv2.LINE_AA)
    return vis


# ------------------------
# Physics-based ball simulation
# ------------------------
def simulate_putt(ball_pos, hole_pos, initial_speed, nx, ny, mag, dt=0.01, max_steps=5000):
    """
    Simulate the ball's path accounting for:
    - Initial velocity (putting speed)
    - Friction/deceleration
    - Slope-induced deflection

    Returns: trajectory points, final position, whether it reached hole
    """
    pos = np.array(ball_pos, dtype=np.float64)
    target = np.array(hole_pos, dtype=np.float64)

    # Initial velocity vector toward hole
    to_hole = target - pos
    direction = to_hole / (np.linalg.norm(to_hole) + 1e-12)
    velocity = direction * initial_speed

    trajectory = [pos.copy()]
    hole_radius = 8.0  # pixels

    for step in range(max_steps):
        # Current position (with bounds checking)
        x, y = int(np.clip(pos[0], 0, TOPDOWN_SIZE[0] - 1)), int(np.clip(pos[1], 0, TOPDOWN_SIZE[1] - 1))

        # Get slope at current position
        slope_x = nx[y, x]
        slope_y = ny[y, x]
        slope_magnitude = mag[y, x]

        # Slope force (perpendicular to velocity gets full effect)
        slope_force = np.array([slope_x, slope_y]) * slope_magnitude * SLOPE_SCALE

        # Friction force (opposes motion)
        speed = np.linalg.norm(velocity)
        if speed < 0.01:  # Ball stopped
            break

        friction = -velocity / speed * FRICTION_COEFF * speed

        # Update velocity
        velocity += (slope_force + friction) * dt

        # Update position
        pos += velocity * dt
        trajectory.append(pos.copy())

        # Check if reached hole
        dist_to_hole = np.linalg.norm(pos - target)
        if dist_to_hole < hole_radius:
            return trajectory, pos, True

        # Check if out of bounds
        if pos[0] < 0 or pos[0] >= TOPDOWN_SIZE[0] or pos[1] < 0 or pos[1] >= TOPDOWN_SIZE[1]:
            break

    return trajectory, pos, False


def find_optimal_aim_point(ball_pos, hole_pos, putting_speed, nx, ny, mag, num_trials=36):
    """
    Try different aim points in a circle around the hole to find the best one.
    The best aim point is the one where the simulated putt gets closest to the hole.
    """
    best_aim = None
    best_final_dist = float('inf')
    best_trajectory = None

    # Try aiming at points around the hole
    search_radius = 50  # pixels
    for i in range(num_trials):
        angle = 2 * np.pi * i / num_trials
        aim_offset = np.array([np.cos(angle), np.sin(angle)]) * search_radius
        aim_point = hole_pos + aim_offset

        # Simulate putting toward this aim point
        to_aim = aim_point - ball_pos
        direction = to_aim / (np.linalg.norm(to_aim) + 1e-12)

        # Simulate with this direction
        pos = np.array(ball_pos, dtype=np.float64)
        velocity = direction * putting_speed

        # Quick simulation
        for _ in range(2000):
            x, y = int(np.clip(pos[0], 0, TOPDOWN_SIZE[0] - 1)), int(np.clip(pos[1], 0, TOPDOWN_SIZE[1] - 1))
            slope_x = nx[y, x]
            slope_y = ny[y, x]
            slope_magnitude = mag[y, x]

            slope_force = np.array([slope_x, slope_y]) * slope_magnitude * SLOPE_SCALE
            speed = np.linalg.norm(velocity)
            if speed < 0.01:
                break

            friction = -velocity / speed * FRICTION_COEFF * speed
            velocity += (slope_force + friction) * 0.01
            pos += velocity * 0.01

            if pos[0] < 0 or pos[0] >= TOPDOWN_SIZE[0] or pos[1] < 0 or pos[1] >= TOPDOWN_SIZE[1]:
                break

        # How close did we get?
        final_dist = np.linalg.norm(pos - hole_pos)
        if final_dist < best_final_dist:
            best_final_dist = final_dist
            best_aim = aim_point

    # Now do full simulation with best aim point
    to_best_aim = best_aim - ball_pos
    direction = to_best_aim / (np.linalg.norm(to_best_aim) + 1e-12)

    pos = np.array(ball_pos, dtype=np.float64)
    velocity = direction * putting_speed
    trajectory = [pos.copy()]

    for _ in range(3000):
        x, y = int(np.clip(pos[0], 0, TOPDOWN_SIZE[0] - 1)), int(np.clip(pos[1], 0, TOPDOWN_SIZE[1] - 1))
        slope_x = nx[y, x]
        slope_y = ny[y, x]
        slope_magnitude = mag[y, x]

        slope_force = np.array([slope_x, slope_y]) * slope_magnitude * SLOPE_SCALE
        speed = np.linalg.norm(velocity)
        if speed < 0.01:
            break

        friction = -velocity / speed * FRICTION_COEFF * speed
        velocity += (slope_force + friction) * 0.01
        pos += velocity * 0.01
        trajectory.append(pos.copy())

        if pos[0] < 0 or pos[0] >= TOPDOWN_SIZE[0] or pos[1] < 0 or pos[1] >= TOPDOWN_SIZE[1]:
            break

    return best_aim, trajectory, best_final_dist


# ------------------------
# Main prediction logic
# ------------------------
def predict_break_from_frame(frame_bgr, ball_img_xy, hole_img_xy, midas_sess, midas_inpname, H,
                             putting_speed_name='medium'):
    # 1) depth
    depth = predict_depth_onnx(midas_sess, midas_inpname, frame_bgr)
    # 2) warp rgb and depth
    top_rgb = warp_to_topdown(frame_bgr, H, TOPDOWN_SIZE)
    depth_top = warp_to_topdown(depth.astype(np.float32), H, TOPDOWN_SIZE)
    # 3) gradient field
    nx, ny, mag = compute_gradient_field(depth_top)
    # 4) map ball/hole to top-down
    pts = np.array([[ball_img_xy], [hole_img_xy]], dtype=np.float32)  # (2,1,2)
    pts_top = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    ball_top = pts_top[0]
    hole_top = pts_top[1]

    # Get putting speed
    putting_speed = PUTTING_SPEEDS.get(putting_speed_name, PUTTING_SPEEDS['medium'])

    # 5) Find optimal aim point using physics simulation
    aim_point, trajectory, final_dist = find_optimal_aim_point(
        ball_top, hole_top, putting_speed, nx, ny, mag
    )

    # 6) Visualize
    vis = draw_vector_field(top_rgb, nx, ny, mag, step=20, length=12)

    # Draw trajectory
    traj_array = np.array(trajectory, dtype=np.int32)
    for i in range(len(traj_array) - 1):
        cv2.line(vis, tuple(traj_array[i]), tuple(traj_array[i + 1]), (255, 0, 255), 2, cv2.LINE_AA)

    # Draw markers
    cv2.circle(vis, (int(ball_top[0]), int(ball_top[1])), 6, (255, 0, 0), -1)
    cv2.circle(vis, (int(hole_top[0]), int(hole_top[1])), 8, (0, 255, 0), 2)
    cv2.circle(vis, (int(aim_point[0]), int(aim_point[1])), 6, (0, 255, 255), -1)

    # Draw aim line
    cv2.arrowedLine(vis, (int(ball_top[0]), int(ball_top[1])),
                    (int(aim_point[0]), int(aim_point[1])),
                    (255, 255, 0), 2, tipLength=0.1)

    # Add text annotations
    dist = np.linalg.norm(hole_top - ball_top)
    cv2.putText(vis, f"Speed: {putting_speed_name} ({putting_speed:.1f} px/frame)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Distance: {dist / PIXELS_PER_METER:.2f}m",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Final miss distance: {final_dist:.1f} px",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return {
        "vis": vis,
        "top_ball": tuple(ball_top),
        "top_hole": tuple(hole_top),
        "aim_point_top": tuple(aim_point),
        "trajectory": trajectory,
        "distance_px": dist,
        "putting_speed": putting_speed,
        "final_dist": final_dist,
    }


# ------------------------
# Program entry
# ------------------------
if __name__ == "__main__":
    # 1) load model
    sess, inp_name = load_midas_onnx(MIDAS_MODEL_PATH)

    # 2) load image - allow command line argument or prompt user
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try default location first
        default_image = "pictures/Dixon-Spirit-9-Iron-Rackham-1st-Hole.jpg"
        if os.path.exists(default_image):
            print(f"Using default image: {default_image}")
            image_path = default_image
        else:
            print("No image path provided as command line argument.")
            print("Please enter the path to your golf green image:")
            image_path = input("Image path: ").strip()
            # Remove quotes if user pasted path with quotes
            image_path = image_path.strip('"').strip("'")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"\nError: Could not load image from: {image_path}")
        print(f"Current working directory: {os.getcwd()}")
        print("\nTip: You can either:")
        print("  1. Provide full path: python GreenReader.py /full/path/to/image.jpg")
        print("  2. Use relative path: python GreenReader.py ./pictures/image.jpg")
        print("  3. Put image in current directory and use: python GreenReader.py image.jpg")
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # 3) calibration (interactive, saved, supports 'r' to force recalibration)
    pts_img = calibrate_with_prompt(frame)

    # 4) compute homography H (image -> topdown)
    pts_world = np.array([
        [0, 0],
        [TOPDOWN_SIZE[0], 0],
        [TOPDOWN_SIZE[0], TOPDOWN_SIZE[1]],
        [0, TOPDOWN_SIZE[1]]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(pts_img, pts_world)


    # 5) click ball and hole (with reset 'r' and quit 'q')
    class ClickState:
        def __init__(self):
            self.points = []
            self.display = frame.copy()


    state = ClickState()
    window_name = "Select Ball (1st click) then Hole (2nd click) - press r to reset, q to quit"


    def click_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(state.points) < 2:
            state.points.append((x, y))
            label = "Ball" if len(state.points) == 1 else "Hole"
            cv2.circle(state.display, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(state.display, label, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(window_name, state.display)


    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, state.display)
    cv2.setMouseCallback(window_name, click_cb)
    print("Click once for the BALL, then click for the HOLE. Press 'r' to reset, 'q' to quit.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        # reset selection
        if key == ord('r'):
            state.points = []
            state.display = frame.copy()
            cv2.imshow(window_name, state.display)
            print("Reset points. Click again.")
            continue
        if key == ord('q'):
            print("Quitting without processing.")
            cv2.destroyAllWindows()
            raise SystemExit(0)
        # proceed if two points selected
        if len(state.points) >= 2:
            break
        else:
            print("Need two clicks (ball and hole). Press r to reset, q to quit, or click on the image.")

    cv2.destroyAllWindows()
    (bx, by), (hx, hy) = state.points[:2]
    print(f"Ball coords: {bx, by} | Hole coords: {hx, hy}")

    # 6) Ask user for putting speed
    print("\nSelect putting speed:")
    print("1 - Soft (slow, breaks more)")
    print("2 - Medium (normal)")
    print("3 - Firm (fast, breaks less)")
    print("4 - Lag (trying to get close)")
    speed_choice = input("Enter choice (1-4) or press Enter for medium: ").strip()

    speed_map = {'1': 'soft', '2': 'medium', '3': 'firm', '4': 'lag', '': 'medium'}
    speed_name = speed_map.get(speed_choice, 'medium')

    print(f"\nCalculating trajectory with '{speed_name}' speed...")

    # 7) run prediction
    output = predict_break_from_frame(
        frame,
        ball_img_xy=(bx, by),
        hole_img_xy=(hx, hy),
        midas_sess=sess,
        midas_inpname=inp_name,
        H=H,
        putting_speed_name=speed_name
    )

    # 8) show result
    print(f"\nResults:")
    print(f"  Distance to hole: {output['distance_px'] / PIXELS_PER_METER:.2f} meters")
    print(f"  Putting speed: {speed_name} ({output['putting_speed']:.1f} px/frame)")
    print(f"  Predicted miss distance: {output['final_dist']:.1f} pixels")
    print(f"  Aim point: {output['aim_point_top']}")

    cv2.imshow("Slope Visualization with Trajectory (Purple = Ball Path)", output["vis"])
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
