import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import os
import csv
from datetime import datetime

# --------- Config (tweak these if needed) ----------
CENTER_BOX     = (0.30, 0.70)   
CENTER_BOX_V   = (0.25, 0.75)
MAX_YAW_DEG    = 25
MAX_PITCH_DEG  = 25
SMOOTH_FRAMES  = 20
MIN_EYE_OPEN   = 0.25
LOG_FILE       = "eye_log.csv"
LOG_INTERVAL   = 1.0 
# ---------------------------------------------------

L_EYE_LEFT  = 33
L_EYE_RIGHT = 133
L_EYE_TOP   = 159
L_EYE_BOT   = 145

R_EYE_LEFT  = 362
R_EYE_RIGHT = 263
R_EYE_TOP   = 386
R_EYE_BOT   = 374

# Iris centers (approx): left 468–471, right 473–476
L_IRIS = [468, 469, 470, 471]
R_IRIS = [473, 474, 475, 476]

mp_face = mp.solutions.face_mesh

def iris_center(landmarks, ids, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in ids], dtype=np.float32)
    return pts.mean(axis=0)

def norm_ratio(val, a, b):
    denom = max(1e-6, (b - a))
    return float(np.clip((val - a) / denom, 0.0, 1.0))

def eye_openness(top, bot, left, right):
    # approximate normalized eye openness: vertical span / horizontal span
    h = abs(top[1] - bot[1])
    w = abs(right[0] - left[0])
    return float(h / max(w, 1e-6))

def head_pose_angles(landmarks, w, h):
    model_points = np.array([
        (0.0,   0.0,   0.0),
        (0.0,  -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3,  32.7, -26.0),
        (-28.9,-28.9, -24.1),
        (28.9, -28.9, -24.1),
    ], dtype=np.float64)

    idxs = [1, 152, 33, 263, 61, 291]
    image_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idxs], dtype=np.float64)

    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4,1))

    ok, rvec, tvec = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0

    rot_mat, _ = cv.Rodrigues(rvec)
    sy = np.sqrt(rot_mat[0,0]**2 + rot_mat[1,0]**2)
    pitch = np.degrees(np.arctan2(-rot_mat[2,0], sy))
    yaw   = np.degrees(np.arctan2(rot_mat[1,0], rot_mat[0,0]))
    return float(yaw), float(pitch)

def decide_on_screen(left_ratios, right_ratios, yaw_deg, pitch_deg, use_vertical=True):
    h_ratio = (left_ratios[0] + right_ratios[0]) / 2.0
    v_ratio = (left_ratios[1] + right_ratios[1]) / 2.0

    centered_h = (CENTER_BOX[0] <= h_ratio <= CENTER_BOX[1])
    centered_v = (CENTER_BOX_V[0] <= v_ratio <= CENTER_BOX_V[1]) if use_vertical else True
    head_ok    = (abs(yaw_deg) <= MAX_YAW_DEG) and (abs(pitch_deg) <= MAX_PITCH_DEG)

    return (centered_h and centered_v and head_ok), h_ratio, v_ratio
def log_status_to_csv(filename, status, last):
    timestamp = datetime.now().strftime("%H%M%S")
    if timestamp == last:
        return last
    print(timestamp, status)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, status])

def main():
    cap = cv.VideoCapture(0)
    status_hist = []
    last_status = "INIT"
    font = cv.FONT_HERSHEY_SIMPLEX
    last = ""
    with mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            on_screen = False
            dbg = ""

            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark

                # Iris centers
                lc = iris_center(lms, L_IRIS, w, h)
                rc = iris_center(lms, R_IRIS, w, h)

                # Eye edges
                l_left  = np.array([lms[L_EYE_LEFT ].x * w, lms[L_EYE_LEFT ].y * h])
                l_right = np.array([lms[L_EYE_RIGHT].x * w, lms[L_EYE_RIGHT].y * h])
                l_top   = np.array([lms[L_EYE_TOP  ].x * w, lms[L_EYE_TOP  ].y * h])
                l_bot   = np.array([lms[L_EYE_BOT  ].x * w, lms[L_EYE_BOT  ].y * h])

                r_left  = np.array([lms[R_EYE_LEFT ].x * w, lms[R_EYE_LEFT ].y * h])
                r_right = np.array([lms[R_EYE_RIGHT].x * w, lms[R_EYE_RIGHT].y * h])
                r_top   = np.array([lms[R_EYE_TOP  ].x * w, lms[R_EYE_TOP  ].y * h])
                r_bot   = np.array([lms[R_EYE_BOT  ].x * w, lms[R_EYE_BOT  ].y * h])

                # Normalized iris ratios per eye (0=left/top, 1=right/bottom)
                l_h = norm_ratio(lc[0], l_left[0], l_right[0])
                l_v = norm_ratio(lc[1], l_top[1],  l_bot[1])
                r_h = norm_ratio(rc[0], r_left[0], r_right[0])
                r_v = norm_ratio(rc[1], r_top[1],  r_bot[1])

                # Head pose
                yaw_deg, pitch_deg = head_pose_angles(lms, w, h)

                # Gate vertical check by eye openness
                l_open = eye_openness(l_top, l_bot, l_left, l_right)
                r_open = eye_openness(r_top, r_bot, r_left, r_right)
                use_vertical = ((l_open + r_open) / 2.0) >= MIN_EYE_OPEN

                on_screen, h_ratio, v_ratio = decide_on_screen(
                    (l_h, l_v), (r_h, r_v), yaw_deg, pitch_deg, use_vertical=use_vertical
                )
                

                # Draw overlay
                if on_screen:
                    color = (0,200,0)
                else:
                    color = (0,0,255)

                for p in [lc, rc]:
                    cv.circle(frame, (int(p[0]), int(p[1])), 2, color, -1)
                cv.putText(frame, f"yaw={yaw_deg:+.0f} pitch={pitch_deg:+.0f}", (20,80), font, 0.6, (0,0,0), 1, cv.LINE_AA)
                cv.putText(frame, f"open={((l_open+r_open)/2):.2f} v_used={int(use_vertical)}", (20,110), font, 0.5, (0,0,0), 1, cv.LINE_AA)

            status_hist.append(on_screen)
            if len(status_hist) > SMOOTH_FRAMES:
                status_hist.pop(0)
            majority = status_hist.count(True) >= (len(status_hist)/2.0)
            last_status = "ON" if majority else "OFF"

            now = datetime.now().strftime("%H%M%S")
            
            last = log_status_to_csv("track_log.csv", last_status, last)
            last = now


            cv.putText(frame, f"Status: {last_status}", (20,45), font, 0.8,
                       (0,200,0) if majority else (0,0,255), 2, cv.LINE_AA)

            cv.imshow("", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
