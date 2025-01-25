# sam2_integration.py
import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import matplotlib.pyplot as plt

# For SAM 2 imports:
from sam2.build_sam import build_sam2_video_predictor

# If you want to see plots inline in a notebook:
# %matplotlib inline
#########################
# Mediapipe Utils
#########################

mp_hands = mp.solutions.hands

# If you're using the new tasks API:
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

def detect_hands_in_image(image_rgb, landmarker):
    """Runs the MediaPipe hand landmarker on the given RGB image."""
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )
    detection_result = landmarker.detect(mp_image)
    return detection_result

def get_bounding_box_coordinates(detection_result, image_width, image_height):
    """
    Returns bounding boxes [x_min, y_min, x_max, y_max] in pixel coords
    for each detected hand.
    """
    bboxes = []
    hand_landmarks_all = detection_result.hand_landmarks
    if not hand_landmarks_all:
        return bboxes

    for hand_landmarks in hand_landmarks_all:
        xs = [lm.x * image_width for lm in hand_landmarks]
        ys = [lm.y * image_height for lm in hand_landmarks]
        x_min = int(min(xs))
        x_max = int(max(xs))
        y_min = int(min(ys))
        y_max = int(max(ys))
        bboxes.append([x_min, y_min, x_max, y_max])
    return bboxes

#########################
# SAM 2 Utils
#########################

def overlay_sam_mask_on_frame(frame_bgr, mask, color=(0,255,0), alpha=0.5):
    """
    Overlays a single binary mask on a BGR frame in-place.
    `mask` is assumed to be (H,W) boolean or 0/1.
    `color` is the BGR color for the mask overlay (default green).
    `alpha` is blend factor.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # Create 3-channel mask in the chosen color
    colored_mask = np.zeros_like(frame_bgr, dtype=np.uint8)
    colored_mask[mask == 1] = color

    # Blend it onto the frame
    cv2.addWeighted(colored_mask, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)

def overlay_multi_masks_on_frame(frame_bgr, masks_dict):
    """
    Overlays multiple masks on the frame.
    `masks_dict` is a dict of {obj_id: binary_mask}, so we can color each object differently.
    For simplicity, we’ll just pick random colors or use a deterministic palette.
    """
    palette = [
        (0, 255, 0),   # green
        (0, 0, 255),   # red
        (255, 0, 0),   # blue
        (0, 255, 255), # yellow
        (255, 0, 255), # magenta
        (255, 255, 0), # cyan
        # ... add more if you expect more objects
    ]
    for i, (obj_id, mask) in enumerate(masks_dict.items()):
        color = palette[i % len(palette)]
        overlay_sam_mask_on_frame(frame_bgr, mask, color=color, alpha=0.5)


import shutil
from tqdm import tqdm

def segment_hands_with_sam2(
    input_video_path,
    output_video_path,
    sam2_checkpoint,        # e.g. "../checkpoints/sam2.1_hiera_large.pt"
    sam2_config,            # e.g. "configs/sam2.1/sam2.1_hiera_l.yaml"
    tmp_dir="./tmp_frames",
    max_frames=None,
    mediapipe_model_path = '../models/hand_landmarker.task',   # Path to your .task model or set None if using default
):
    """
    1) Extract all frames from input_video_path into `tmp_dir` as JPEGs.
    2) Use a MediaPipe hand landmarker to detect bounding boxes per frame.
    3) Initialize SAM2 on that folder of frames.
    4) Add bounding box prompts for every hand in every frame (unique object ID).
    5) Propagate to get spatio-temporal masks for all objects.
    6) Render each frame with the resulting masks into a new video at output_video_path.
    """

    ##############################
    # 0. Prep temporary directory
    ##############################
    os.makedirs(tmp_dir, exist_ok=True)
    # Clear out old files if any
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))

    ##############################
    # 1. Extract frames with OpenCV
    ##############################
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    frame_idx = 0
    all_frame_paths = []
    while True:
        success, frame_bgr = cap.read()
        if not success:
            break
        if max_frames and frame_idx >= max_frames:
            break
        out_path = os.path.join(tmp_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(out_path, frame_bgr)
        all_frame_paths.append(out_path)
        frame_idx += 1
    cap.release()
    total_frames = len(all_frame_paths)
    print(f"Extracted {total_frames} frames to {tmp_dir}")

    if total_frames == 0:
        print("No frames extracted. Exiting.")
        return

    ##############################
    # 2. Run MediaPipe on each frame to get hand bounding boxes
    ##############################
    # Build a mediapipe HandLandmarker in "image" mode
    if mediapipe_model_path is None:
        raise ValueError("Please provide a valid .task model path for MediaPipe or adjust to a default usage.")
    mp_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mediapipe_model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5
    )
    bounding_boxes_per_frame = [[] for _ in range(total_frames)]

    with HandLandmarker.create_from_options(mp_options) as landmarker:
        for i in tqdm(range(total_frames), desc="Running MediaPipe Hand Detection"):
            frame_bgr = cv2.imread(all_frame_paths[i])
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_bgr.shape

            result = detect_hands_in_image(
                image_rgb=frame_rgb,
                landmarker=landmarker,
            )
            bboxes = get_bounding_box_coordinates(result, w, h)
            bounding_boxes_per_frame[i] = bboxes

    ##############################
    # 3. Initialize SAM2 on these frames
    ##############################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=tmp_dir)  # loads the frames in memory
    predictor.reset_state(inference_state)

    ##############################
    # 4. Add bounding boxes for each hand in each frame
    ##############################
    # We'll assign each bounding box a unique object id across the entire video.
    # For example, obj_id = frame_index * 100 + hand_index
    # so that no two hands across frames share the same ID. (A naive approach.)
    for i in tqdm(range(total_frames), desc="Adding bounding boxes to SAM2"):
        bboxes = bounding_boxes_per_frame[i]
        for j, box in enumerate(bboxes):
            x_min, y_min, x_max, y_max = box
            obj_id = i * 100 + j  # naive unique ID
            box_arr = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

            # Add this bounding box to SAM2 on frame i
            # No clicks in this example, just the box.
            _out_obj_ids, _out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=i,
                obj_id=obj_id,
                box=box_arr
            )
    print("All bounding boxes have been added to SAM2.")

    ##############################
    # 5. Propagate to get masks
    ##############################
    # This will yield final masks for all objects on all frames.
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        # out_mask_logits: (num_objs, H, W)
        # out_obj_ids: list of object IDs
        # threshold them to get binary masks
        frame_masks = {}
        for i, obj_id in enumerate(out_obj_ids):
            mask_bin = (out_mask_logits[i] > 0).cpu().numpy().astype(np.uint8)
            frame_masks[obj_id] = mask_bin
        video_segments[out_frame_idx] = frame_masks

    ##############################
    # 6. Render new video with masks
    ##############################
    # We read each original frame again, overlay all the masks, and write to output_video_path
    # (Alternatively, you could just write them as JPEG frames.)
    first_frame = cv2.imread(all_frame_paths[0])
    H, W, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30  # or match the original input video fps
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    for i in tqdm(range(total_frames), desc="Writing output video"):
        frame_bgr = cv2.imread(all_frame_paths[i])
        masks_dict = video_segments.get(i, {})
        # Overlay each mask
        overlay_multi_masks_on_frame(frame_bgr, masks_dict)
        out_writer.write(frame_bgr)

    out_writer.release()
    print(f"Output video with masks saved to: {output_video_path}")

    # Optionally, clean up temporary frames:
    # shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    segment_hands_with_sam2(
        input_video_path="../data/test.mp4",
        output_video_path="../data/output.mp4",
        sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt",
        sam2_config="../sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        tmp_dir="../tmp_frames",
        max_frames=100  # Optional: limit to first 100 frames for testing
    )