# sam2_integration.py
def segment_hands_with_sam2(
    input_video_path,
    output_video_path,
    sam2_checkpoint,
    sam2_config,
    tmp_dir="./tmp_frames",
    max_frames=None,
    mediapipe_model_path="../models/hand_landmarker.task",
    prompt_mode="box",  # "box", "point", or "both"
    overlay_mode="both",  # "none", "mask", "bbox", or "both"
    overlay_original=True  # NEW FLAG
):
    """
    1) Extract frames from input_video_path into tmp_dir.
    2) Use MediaPipe to detect bounding boxes per frame.
    3) Initialize SAM2 on that folder of frames.
    4) Add prompts (box, point, or both).
    5) Propagate to get spatio-temporal masks.
    6) Render and save the annotated (or mask-only) video.

    Args:
        input_video_path (str): Path to the input video.
        output_video_path (str): Path to the output video with masks.
        sam2_checkpoint (str): Path to the SAM2 checkpoint.
        sam2_config (str): Path to the SAM2 config file.
        tmp_dir (str): Directory for temporarily storing extracted frames.
        max_frames (int|None): If not None, limit processing to the first N frames.
        mediapipe_model_path (str): Path to MediaPipe hand_landmarker .task model.
        prompt_mode (str): One of ["box","point","both"].
                          "box" -> bounding boxes only,
                          "point" -> single center point,
                          "both" -> bounding boxes + center point.
        overlay_mode (str): One of ["none", "mask", "bbox", "both"].
                            "none"  -> No overlay video generated.
                            "mask"  -> Overlay segmentation masks only.
                            "bbox"  -> Overlay bounding boxes only.
                            "both"  -> Overlay both masks and bounding boxes.
        overlay_original (bool): 
            If True, overlay the masks (and possibly bounding boxes) on the original video.
            If False, output only a color-coded mask video (black background).
    """

    import os
    import cv2
    import numpy as np
    import mediapipe as mp
    import torch
    import shutil
    from tqdm import tqdm

    ##############################
    # 0. Prep temporary directory
    ##############################
    os.makedirs(tmp_dir, exist_ok=True)
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
    print(f"[DEBUG] Extracted {total_frames} frames to {tmp_dir}")

    if total_frames == 0:
        print("No frames extracted. Exiting.")
        return

    ##############################
    # 2. Run MediaPipe to get bounding boxes
    ##############################
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    mp_hands = mp.solutions.hands
    BaseOptions = mp_python.BaseOptions
    VisionRunningMode = vision.RunningMode
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions

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

    if mediapipe_model_path is None:
        raise ValueError(
            "Please provide a valid .task model path for MediaPipe or use the default."
        )
    mp_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mediapipe_model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5
    )
    bounding_boxes_per_frame = [[] for _ in range(total_frames)]

    print("[DEBUG] Starting MediaPipe hand detection...")
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
            # Debug print each frame's bounding box
            print(f"[DEBUG] Frame {i}, bounding boxes: {bboxes}")

    ##############################
    # 3. Initialize SAM2
    ##############################
    from sam2.build_sam import build_sam2_video_predictor
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # Example: use bfloat16 or float32
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    print(f"[DEBUG] Initializing SAM2 predictor on device='{device}'")
    predictor = build_sam2_video_predictor(
        config_file=sam2_config,
        ckpt_path=sam2_checkpoint,
        device=device,
        max_masks_per_prompt=1,
    )

    predictor.eval()

    # Force float32 if desired
    for name, param in predictor.named_parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()
            print(f"[DEBUG] Converting param '{name}' to float32")
    for name, buf in predictor.named_buffers():
        if buf.dtype in (torch.float16, torch.bfloat16):
            buf.data = buf.data.float()
            print(f"[DEBUG] Converting buffer '{name}' to float32")

    print("[DEBUG] Creating inference state...")
    inference_state = predictor.init_state(video_path=tmp_dir)
    predictor.reset_state(inference_state)

    ##############################
    # 4. Add prompts (box, point, or both)
    ##############################
    print("[DEBUG] Adding prompts to SAM2...")

    # Modify the prompt addition loop to limit to two prompts per frame
    for i in tqdm(range(total_frames), desc="Adding prompts"):
        bboxes = bounding_boxes_per_frame[i]
        
        # Limit to two bounding boxes per frame
        if len(bboxes) > 2:
            bboxes = bboxes[:2]
            print(f"[DEBUG] Frame {i}: More than two bounding boxes detected. Limiting to first two.")
        
        for j, box in enumerate(bboxes):
            x_min, y_min, x_max, y_max = box
            obj_id = j  # Assign consistent obj_id per hand across frames (0 and 1)
            
            # Prepare bounding box
            box_arr = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

            # Prepare point -> e.g., center of the bounding box
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            point_coords = np.array([[cx, cy]], dtype=np.float32)  # shape (1, 2)
            point_labels = np.array([1], dtype=np.int32)  # '1' => positive

            print(f"[DEBUG] Frame={i}, obj_id={obj_id}, box={box_arr.tolist()}")
            print(f"[DEBUG] Prompt center point=({cx:.1f}, {cy:.1f})")

            if prompt_mode == "box":
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=i,
                    obj_id=obj_id,
                    box=box_arr,
                )
            elif prompt_mode == "point":
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=i,
                    obj_id=obj_id,
                    points=point_coords,
                    labels=point_labels,
                )
            elif prompt_mode == "both":
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=i,
                    obj_id=obj_id,
                    box=box_arr,
                    points=point_coords,
                    labels=point_labels,
                )
            else:
                raise ValueError(
                    "Invalid prompt_mode. Must be one of ['box','point','both']."
                )

    print("All prompts have been added to SAM2.")

    ##############################
    # 5. Propagate to get masks
    ##############################
    print("[DEBUG] Starting propagate_in_video...")
    video_segments = {}
    try:
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            # out_mask_logits shape: (num_objs, H, W)
            frame_masks = {}
            for idx_in_batch, obj_id in enumerate(out_obj_ids):
                mask_bin = (out_mask_logits[idx_in_batch, 0] > 0).cpu().numpy().astype(np.uint8)
                frame_masks[obj_id] = mask_bin

            video_segments[out_frame_idx] = frame_masks
            print(f"[DEBUG] Stored masks for frame {out_frame_idx} with IDs {out_obj_ids}")
    except Exception as e:
        print("[DEBUG] Exception in propagate_in_video:")
        print(e)
        raise

    ##############################
    # 6. Render new video
    ##############################

    # Utility overlay functions
    def overlay_sam_mask_on_frame(frame_bgr, mask, color=(0, 255, 0), alpha=0.5):
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        overlay = np.zeros_like(frame_bgr, dtype=np.uint8)
        overlay[mask == 1] = color
        blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
        mask_3ch = np.dstack([mask]*3)
        frame_bgr[mask_3ch == 1] = blended[mask_3ch == 1]


    def overlay_multi_masks_on_frame(frame_bgr, masks_dict, alpha=0.5):
        # Some simple color palette
        palette = [
            (0, 255, 0),   # green
            (0, 0, 255),   # red
            (255, 0, 0),   # blue
            (0, 255, 255), # yellow
            (255, 0, 255), # magenta
            (255, 255, 0), # cyan
        ]
        for i, (obj_id, mask) in enumerate(masks_dict.items()):
            color = palette[i % len(palette)]
            overlay_sam_mask_on_frame(frame_bgr, mask, color=color, alpha=alpha)

    def overlay_bounding_boxes_on_frame(frame_bgr, bboxes, color=(255, 0, 0), thickness=2):
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), color, thickness)

    def overlay_combined_on_frame(frame_bgr, masks_dict, bboxes, alpha=0.5):
        # Overlays both masks (with alpha-blending) and bounding boxes
        overlay_multi_masks_on_frame(frame_bgr, masks_dict, alpha=alpha)
        overlay_bounding_boxes_on_frame(frame_bgr, bboxes, color=(255, 0, 0), thickness=2)


    if overlay_mode != "none":
        print("[DEBUG] Rendering output video...")
        first_frame = cv2.imread(all_frame_paths[0])
        H, W, _ = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'avc1' for H.264
        fps = 30
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

        for i in tqdm(range(total_frames), desc="Writing output video"):
            frame_bgr = cv2.imread(all_frame_paths[i])
            masks_dict = video_segments.get(i, {})
            bboxes = bounding_boxes_per_frame[i]

            if not overlay_original:
                # Start with a black canvas (same size as original frame)
                out_frame = np.zeros_like(frame_bgr, dtype=np.uint8)
                
                if overlay_mode == "mask":
                    # Render color-coded masks on black
                    palette = [
                        (0, 255, 0),   # green
                        (0, 0, 255),   # red
                        (255, 0, 0),   # blue
                        (0, 255, 255), # yellow
                        (255, 0, 255), # magenta
                        (255, 255, 0), # cyan
                    ]
                    for idx_obj, (obj_id, mask) in enumerate(masks_dict.items()):
                        color = palette[idx_obj % len(palette)]
                        out_frame[mask == 1] = color

                elif overlay_mode == "bbox":
                    # Draw bounding boxes on black
                    overlay_bounding_boxes_on_frame(out_frame, bboxes, color=(255, 0, 0), thickness=2)

                elif overlay_mode == "both":
                    # First put color-coded masks, then bounding boxes
                    palette = [
                        (0, 255, 0),   # green
                        (0, 0, 255),   # red
                        (255, 0, 0),   # blue
                        (0, 255, 255), # yellow
                        (255, 0, 255), # magenta
                        (255, 255, 0), # cyan
                    ]
                    # Color-coded masks
                    for idx_obj, (obj_id, mask) in enumerate(masks_dict.items()):
                        color = palette[idx_obj % len(palette)]
                        out_frame[mask == 1] = color
                    # Then bounding boxes
                    overlay_bounding_boxes_on_frame(out_frame, bboxes, color=(255, 0, 0), thickness=2)

                elif overlay_mode == "none":
                    # Black frame only
                    out_frame = np.zeros_like(frame_bgr, dtype=np.uint8)

                else:
                    raise ValueError(
                        "Invalid overlay_mode. Must be one of ['none', 'mask', 'bbox', 'both']."
                    )

            else:
                # Overlay on the original frame
                if overlay_mode == "mask":
                    overlay_multi_masks_on_frame(frame_bgr, masks_dict, alpha=0.5)
                    out_frame = frame_bgr

                elif overlay_mode == "bbox":
                    overlay_bounding_boxes_on_frame(frame_bgr, bboxes, color=(255, 0, 0), thickness=2)
                    out_frame = frame_bgr

                elif overlay_mode == "both":
                    overlay_combined_on_frame(frame_bgr, masks_dict, bboxes, alpha=0.5)
                    out_frame = frame_bgr

                else:
                    raise ValueError(
                        "Invalid overlay_mode. Must be one of ['none', 'mask', 'bbox', 'both']."
                    )

            out_writer.write(out_frame)

        out_writer.release()
        print(f"[DEBUG] Output video saved to: {output_video_path}")
    else:
        print("[DEBUG] Overlay video generation is disabled (overlay_mode='none').")

    # Clean up temporary frames
    shutil.rmtree(tmp_dir)
    print("[DEBUG] Temporary frames cleaned up.")



if __name__ == "__main__":

    segment_hands_with_sam2(
    input_video_path="../data/test.mp4",
    output_video_path="../output/hands_mask_only.mp4",
    sam2_checkpoint="../sam2/checkpoints/sam2.1_hiera_large.pt",
    sam2_config="../sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
    tmp_dir="../tmp_frames_mask",
    max_frames=10,
    mediapipe_model_path="../models/hand_landmarker.task",
    prompt_mode="box",
    overlay_mode="mask",   # or "both" / "bbox" / "none"
    overlay_original=False)
   
    segment_hands_with_sam2(
    input_video_path="../data/test.mp4",
    output_video_path="../output/hands.mp4",
    sam2_checkpoint="../sam2/checkpoints/sam2.1_hiera_large.pt",
    sam2_config="../sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
    tmp_dir="../tmp_frames_mask",
    max_frames=10,
    mediapipe_model_path="../models/hand_landmarker.task",
    prompt_mode="box",
    overlay_mode="mask",   # or "both" / "bbox" / "none"
    overlay_original=True)
