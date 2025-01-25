import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Prepare MediaPipe classes
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# For drawing connections between landmarks (hand skeleton).
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),       # thumb
    (5,6), (6,7), (7,8),             # index finger
    (9,10), (10,11), (11,12),        # middle finger
    (13,14), (14,15), (15,16),       # ring finger
    (17,18), (18,19), (19,20),       # pinky
    (0,5), (5,9), (9,13), (13,17),   # palm connections
]

def detect_hands_in_image(
    image_rgb: np.ndarray,
    landmarker,
    orig_width: int,
    orig_height: int
):
    """
    Runs the MediaPipe hand landmarker on the given RGB image.

    Args:
        image_rgb (np.ndarray): (H,W,3) RGB image.
        landmarker: A HandLandmarker instance configured for IMAGE mode.
        orig_width (int): Original width in pixels (image_rgb.shape[1]).
        orig_height (int): Original height in pixels (image_rgb.shape[0]).

    Returns:
        detection_result: a mediapipe.tasks.python.vision.HandLandmarkerResult object,
                          which has:
                            - hand_landmarks: list of 21-keypoint lists
                            - handedness: list of classification results (left/right)
                            - etc.
    """
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )
    detection_result = landmarker.detect(mp_image)
    return detection_result

def get_bounding_box_coordinates(detection_result, image_width, image_height):
    """
    Returns the bounding box coordinates for each detected hand in pixel coordinates.

    Args:
        detection_result: The result from MediaPipe hand landmarker.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.

    Returns:
        List of tuples, each containing (x_min, y_min, x_max, y_max) for a hand.
    """
    bboxes = []
    hand_landmarks_all = detection_result.hand_landmarks
    if not hand_landmarks_all:
        return bboxes
    
    for hand_landmarks in hand_landmarks_all:
        xs = [lm.x * image_width for lm in hand_landmarks]
        ys = [lm.y * image_height for lm in hand_landmarks]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        bboxes.append( (x_min, y_min, x_max, y_max) )
    return bboxes

def get_landmark_coordinates(detection_result, image_width, image_height):
    """
    Returns the landmark coordinates for each detected hand in pixel coordinates (x, y)
    and the original z (relative depth).

    Args:
        detection_result: The result from MediaPipe hand landmarker.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.

    Returns:
        List of lists, where each inner list contains tuples (x, y, z) for each landmark.
    """
    landmarks_list = []
    hand_landmarks_all = detection_result.hand_landmarks
    if not hand_landmarks_all:
        return landmarks_list
    
    for hand_landmarks in hand_landmarks_all:
        landmarks = []
        for lm in hand_landmarks:
            x = lm.x * image_width
            y = lm.y * image_height
            z = lm.z
            landmarks.append( (x, y, z) )
        landmarks_list.append( landmarks )
    return landmarks_list

def draw_hand_annotations(frame_bgr: np.ndarray, detection_result):
    """
    Draws bounding boxes and landmarks on the BGR frame in-place.

    Args:
        frame_bgr (np.ndarray): The original frame in BGR format.
        detection_result: The result from MediaPipe hand landmarker,
                          containing hand landmarks + handedness info.
    """

    height, width, _ = frame_bgr.shape
    hand_landmarks_all = detection_result.hand_landmarks
    hand_handedness_all = detection_result.handedness

    if not hand_landmarks_all:
        return  # No hands detected; do nothing

    for hand_idx, hand_landmarks in enumerate(hand_landmarks_all):
        # Convert normalized landmark coordinates to pixel coordinates.
        px = [int(lm.x * width) for lm in hand_landmarks]
        py = [int(lm.y * height) for lm in hand_landmarks]

        # Draw the 21 landmarks.
        for i in range(21):
            cv2.circle(frame_bgr, (px[i], py[i]), 5, (0, 255, 0), -1)

        # connect them to form a skeleton.
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            cv2.line(frame_bgr,
                     (px[start_idx], py[start_idx]),
                     (px[end_idx],   py[end_idx]),
                     (0, 255, 0), 2)

        # Draw bounding box (use min/max x,y from landmarks).
        x_min, x_max = min(px), max(px)
        y_min, y_max = min(py), max(py)
        cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max),
                      (0, 0, 255), 2)

        # Label with "Left" or "Right" if available.
        if hand_idx < len(hand_handedness_all):
            handedness_label = hand_handedness_all[hand_idx][0].category_name
            confidence_score = hand_handedness_all[hand_idx][0].score
            text = f"{handedness_label} ({confidence_score:.2f})"
            cv2.putText(frame_bgr, text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def main():
    # 1. Paths
    input_video_path = "../data/test.mp4"
    output_frames_dir = "../output"
    os.makedirs(output_frames_dir, exist_ok=True)

    # 2. Create a HandLandmarker instance in IMAGE mode.
    model_path = "/gpfs/data/shenlab/wz1492/HandTrack-SAM2/models/hand_landmarker.task"  # Update if needed
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,  # Detect up to 2 hands, you can set more if needed
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with HandLandmarker.create_from_options(options) as landmarker:
        
        # 3. Open the video.
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_video_path}")
            return

        frame_index = 0

        while True:
            success, frame_bgr = cap.read()
            if not success:
                break  # End of video

            # Convert BGR -> RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 4. Run detection
            detection_result = detect_hands_in_image(
                image_rgb=frame_rgb,
                landmarker=landmarker,
                orig_width=frame_bgr.shape[1],
                orig_height=frame_bgr.shape[0]
            )

            # 5. Draw the landmarks and bounding boxes on the frame
            draw_hand_annotations(frame_bgr, detection_result)

            # 6. Get coordinates and print them
            image_width = frame_bgr.shape[1]
            image_height = frame_bgr.shape[0]
            
            bboxes = get_bounding_box_coordinates(detection_result, image_width, image_height)
            landmarks = get_landmark_coordinates(detection_result, image_width, image_height)

            # Print bounding boxes
            for hand_idx, bbox in enumerate(bboxes):
                print(f"Frame {frame_index}, Hand {hand_idx}: BBox (x_min={bbox[0]:.2f}, y_min={bbox[1]:.2f}, x_max={bbox[2]:.2f}, y_max={bbox[3]:.2f})")

            # Print landmarks
            for hand_idx, hand_landmarks in enumerate(landmarks):
                print(f"Frame {frame_index}, Hand {hand_idx}: Landmarks:")
                for idx, (x, y, z) in enumerate(hand_landmarks):
                    print(f"  {idx}: (x={x:.2f}, y={y:.2f}, z={z:.2f})")

            # 7. Save the annotated frame to disk
            out_path = os.path.join(output_frames_dir, f"frame_{frame_index:05d}.jpg")
            cv2.imwrite(out_path, frame_bgr)
            frame_index += 1

        cap.release()
        print(f"Done! Processed {frame_index} frames.")
        print(f"Annotated frames saved in: {output_frames_dir}")


if __name__ == "__main__":
    main()