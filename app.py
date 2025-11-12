import cv2
import mediapipe as mp
import numpy as np

# Load the anatomy image
image = cv2.imread('sample.jpeg')
if image is None:
    raise Exception("Image not found! Please place 'sample.jpeg' in the folder.")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Fingertips index
finger_tips = [4, 8, 12, 16, 20]

# State trackers
prev_thumb_index_dist = None
prev_hand_angle = None
zoom_level = 1.0
rotation_x = 0
rotation_y = 0
rotation_z = 0

# Count fingers
def count_fingers(hand_landmarks):
    fingers = []
    if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for tip in finger_tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# Draw fingertip circles
def draw_fingertip_circles(frame, hand_landmarks):
    h, w, _ = frame.shape
    for tip in finger_tips:
        cx = int(hand_landmarks.landmark[tip].x * w)
        cy = int(hand_landmarks.landmark[tip].y * h)
        cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)
def simulate_3d_rotation(img, angle_x, angle_y):
    h, w = img.shape[:2]
    f = w  # focal length for perspective effect

    # Convert degrees to radians
    ax = np.radians(angle_x)
    ay = np.radians(angle_y)

    # Perspective warp matrix simulating X and Y axis
    dx = np.tan(ay) * f
    dy = np.tan(ax) * f

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [0 + dx, 0 - dy],
        [w - dx, 0 - dy],
        [w - dx, h + dy],
        [0 + dx, h + dy]
    ])

    mat = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, mat, (w, h))
    return warped

# Transform image (still only Z-axis for now)
def transform_image(img, zoom, angle_x, angle_y, angle_z):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_z, zoom)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

# Main loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    action = "Waiting..."

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        draw_fingertip_circles(frame, hand_landmarks)

        fingers = count_fingers(hand_landmarks)
        total_fingers = sum(fingers)

        # More stable rotation angle from wrist to middle finger base
        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]

        x1, y1 = int(wrist.x * w), int(wrist.y * h)
        x2, y2 = int(middle_mcp.x * w), int(middle_mcp.y * h)

        dx = x2 - x1
        dy = y2 - y1
        hand_angle = np.arctan2(dy, dx) * 180 / np.pi

        # Zoom (thumb + index)
        if fingers == [1, 1, 0, 0, 0]:
            action = "Zooming"
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            thumb_pt = np.array([int(thumb_tip.x * w), int(thumb_tip.y * h)])
            index_pt = np.array([int(index_tip.x * w), int(index_tip.y * h)])
            distance = np.linalg.norm(thumb_pt - index_pt)
            if prev_thumb_index_dist is not None:
                diff = distance - prev_thumb_index_dist
                if diff > 2 and zoom_level < 2.5:
                    zoom_level += 0.02
                elif diff < -2 and zoom_level > 0.5:
                    zoom_level -= 0.02

            prev_thumb_index_dist = distance

        # Rotation X (3 fingers)
        elif fingers == [1, 1, 1, 0, 0]:
            action = "Rotate X"
            if prev_hand_angle is not None:
                delta = hand_angle - prev_hand_angle
                if abs(delta) > 1:
                    rotation_x += delta * 0.4
            prev_hand_angle = hand_angle

        # Rotation Y (4 fingers)
        elif fingers == [1, 1, 1, 1, 0]:
            action = "Rotate Y"
            if prev_hand_angle is not None:
                delta = hand_angle - prev_hand_angle
                if abs(delta) > 1:
                    rotation_y += delta * 0.4
            prev_hand_angle = hand_angle

        # Rotation Z (5 fingers)
        elif fingers == [1, 1, 1, 1, 1]:
            action = "Rotate Z"
            if prev_hand_angle is not None:
                delta = hand_angle - prev_hand_angle
                if abs(delta) > 1:
                    rotation_z += delta * 0.5
            prev_hand_angle = hand_angle

        elif total_fingers == 0:
            action = "Reset"
            zoom_level = 1.0
            rotation_x = rotation_y = rotation_z = 0
            prev_hand_angle = None
            prev_thumb_index_dist = None

            

    # Transform image (still only Z used)
    rotated_img = simulate_3d_rotation(image, rotation_x, rotation_y)
    transformed_img = transform_image(rotated_img, zoom_level, 0, 0, rotation_z)
    resized_img = cv2.resize(transformed_img, (320, 320))
    frame[10:330, w - 330:w - 10] = resized_img

    # Display action text
    cv2.putText(frame, f"Action: {action}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.imshow("HoloGesture - Smoother Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
