# main.py

import argparse
from sam2_integration import segment_hands_with_sam2

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Segment hands in a video using MediaPipe and SAM2."
    )

    # Required arguments
    parser.add_argument(
        "--input_video_path",
        type=str,
        required=True,
        help="Path to the input video file."
    )
    parser.add_argument(
        "--output_video_path",
        type=str,
        required=True,
        help="Path to save the output video with masks."
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        required=True,
        help="Path to the SAM2 checkpoint file."
    )
    parser.add_argument(
        "--sam2_config",
        type=str,
        required=True,
        help="Path to the SAM2 config file."
    )

    # Optional arguments
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="./tmp_frames",
        help="Temporary directory to store extracted frames."
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process."
    )
    parser.add_argument(
        "--mediapipe_model_path",
        type=str,
        default="../models/hand_landmarker.task",
        help="Path to the MediaPipe hand landmarker .task model."
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        choices=["box", "point", "both"],
        default="box",
        help="Mode for adding prompts: 'box', 'point', or 'both'."
    )
    parser.add_argument(
        "--overlay_mode",
        type=str,
        choices=["none", "mask", "bbox", "both"],
        default="both",
        help="Overlay mode for output video: 'none', 'mask', 'bbox', or 'both'."
    )
    parser.add_argument(
        "--overlay_original",
        action='store_true',
        help="If set, overlays masks and bounding boxes on the original video."
    )

    # Allow multiple runs by accepting multiple output configurations
    parser.add_argument(
        "--additional_runs",
        nargs='*',
        help=(
            "Additional runs with different parameters. "
            "Each run should be specified as a JSON string with keys matching the arguments."
            "Example: --additional_runs '{\"output_video_path\":\"out1.mp4\",\"overlay_original\":false}' '{\"output_video_path\":\"out2.mp4\",\"overlay_original\":true}'"
        )
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Primary run
    print("Starting primary run...")
    segment_hands_with_sam2(
        input_video_path=args.input_video_path,
        output_video_path=args.output_video_path,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        tmp_dir=args.tmp_dir,
        max_frames=args.max_frames,
        mediapipe_model_path=args.mediapipe_model_path,
        prompt_mode=args.prompt_mode,
        overlay_mode=args.overlay_mode,
        overlay_original=args.overlay_original
    )

    # Handle additional runs if any
    if args.additional_runs:
        import json
        for run_idx, run_config in enumerate(args.additional_runs, start=1):
            try:
                config = json.loads(run_config)
                print(f"Starting additional run {run_idx} with config: {config}")
                segment_hands_with_sam2(
                    input_video_path=config.get("input_video_path", args.input_video_path),
                    output_video_path=config["output_video_path"],
                    sam2_checkpoint=config.get("sam2_checkpoint", args.sam2_checkpoint),
                    sam2_config=config.get("sam2_config", args.sam2_config),
                    tmp_dir=config.get("tmp_dir", args.tmp_dir),
                    max_frames=config.get("max_frames", args.max_frames),
                    mediapipe_model_path=config.get("mediapipe_model_path", args.mediapipe_model_path),
                    prompt_mode=config.get("prompt_mode", args.prompt_mode),
                    overlay_mode=config.get("overlay_mode", args.overlay_mode),
                    overlay_original=config.get("overlay_original", args.overlay_original)
                )
            except json.JSONDecodeError as e:
                print(f"Error parsing additional run {run_idx}: {e}")
            except KeyError as e:
                print(f"Missing required key in additional run {run_idx}: {e}")

if __name__ == "__main__":
    main()


