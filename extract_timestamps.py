# Import necessary libraries
import cv2
import paddleocr
import paddle # Import paddle for GPU info
import sys
import os
import csv
import re
# import pandas as pd # No longer needed
from datetime import timedelta
import time
import numpy as np
import json # For loading/saving ROIs
import math
import glob # For handling file paths and patterns
import traceback # Import traceback module
import argparse # For robust argument parsing

# --- Configuration ---
ROI_CONFIG_FILE = 'roi_config.json'
ROI_TARGET_WIDTH = 300 # Target width for straightened patches
# --- >>> Character Whitelist for Post-Processing <<< ---
# After OCR, we will filter the result to only keep these characters.
# Includes space, hyphen, and underscore for prefixes like FRONT_CAM_C2
TIMESTAMP_CHAR_WHITELIST = "0123456789:.-_ "

# --- >>> Display Configuration <<< ---
# Set the desired width for the display window. Height will be calculated to maintain aspect ratio.
DISPLAY_WIDTH = 2000

# --- >>> Preprocessing Flags (Applied BEFORE ROI Warp) <<< ---
ROTATE_FRAME_90_CW = False
ROTATE_FRAME_90_CCW = False
ROTATE_FRAME_180 = False
FLIP_HORIZONTAL = False
# --- >>> ROI Enhancement Flags (Applied AFTER ROI Warp/Transform) <<< ---
APPLY_GRAYSCALE = True
ENHANCE_CONTRAST = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
APPLY_THRESHOLDING = False # Optional: Apply Otsu's thresholding after contrast enhancement
# --- >>> PaddleOCR Tuning Parameters <<< ---
PADDLE_UNCLIP_RATIO = 2.2
PADDLE_BOX_THRESH = 0.5
PADDLE_DB_THRESH = 0.3
# --- >>> Optimization Parameters <<< ---
NORMAL_ORIENTATION_CONF_THRESHOLD = 0.85

# --- Global Variables for Interaction ---
rois = [] # List of ROI dicts
next_roi_id = 0
current_roi_points = []
defining_roi = False
paused = False
show_warped = False

# Define colors for ROI bounding boxes
ROI_COLORS = [
    ((255, 0, 0), "Blue"), ((0, 0, 255), "Red"), ((0, 255, 255), "Yellow"),
    ((255, 0, 255), "Magenta"), ((0, 255, 0), "Green"), ((255, 255, 0), "Cyan"),
    ((0, 165, 255), "Orange"), ((128, 0, 128), "Purple"), ((0, 0, 0), "Black"),
    ((255, 255, 255), "White"), ((128, 128, 128), "Gray"), ((0, 90, 180), "Brown")
]
# Define transformations to test during optimization
TRANSFORMATIONS_TO_TEST = []
for r in [0, 90, 180, 270]:
    TRANSFORMATIONS_TO_TEST.append({'rotate': r, 'flip_code': None}) # No flip
    TRANSFORMATIONS_TO_TEST.append({'rotate': r, 'flip_code': 1})    # Horizontal Flip
    TRANSFORMATIONS_TO_TEST.append({'rotate': r, 'flip_code': 0})    # Vertical Flip
    TRANSFORMATIONS_TO_TEST.append({'rotate': r, 'flip_code': -1})   # Both Flips

# --- Helper Functions ---

def format_timedelta(td):
    """Formats a timedelta object into a HH:MM:SS.fff string."""
    if td is None: return "N/A"
    total_seconds = td.total_seconds();
    if total_seconds < 0: total_seconds = 0
    hours, rem = divmod(total_seconds, 3600); minutes, sec_float = divmod(rem, 60)
    sec_int = int(sec_float); ms = int((sec_float - sec_int) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(sec_int):02}.{ms:03}"

def load_rois(config_path):
    """Loads ROIs (points and transform) from config file."""
    global rois, next_roi_id
    loaded_data = []; validated_rois = []; max_id = -1
    default_transform = {'rotate': 0, 'flip_code': None}
    try:
        with open(config_path, 'r') as f: loaded_data = json.load(f)
        if isinstance(loaded_data, list):
            for i, item in enumerate(loaded_data):
                if isinstance(item, dict) and 'id' in item and 'points' in item:
                     transform = item.get('transform', default_transform)
                     if 'flip_code' not in transform: transform['flip_code'] = None
                     if (isinstance(item['points'], list) and len(item['points']) == 4 and
                         isinstance(transform, dict) and 'rotate' in transform and 'flip_code' in transform):
                        if all(isinstance(p, list) and len(p) == 2 and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)) for p in item['points']):
                             item['transform'] = transform; validated_rois.append(item); max_id = max(max_id, item['id'])
                        else: print(f"W: Invalid points format ROI #{item.get('id', i)}. Skipping.")
                     else: print(f"W: Invalid structure/transform ROI #{item.get('id', i)}. Skipping.")
                else: print(f"W: Invalid item format ROI #{i}. Skipping.")
        else: print(f"W: Invalid format in {config_path}. Starting empty.")
        rois = validated_rois; next_roi_id = max_id + 1
        print(f"Loaded {len(rois)} ROIs from {config_path}")
        return True
    except FileNotFoundError: print(f"ROI config file {config_path} not found."); rois = []; next_roi_id = 0; return False
    except Exception as e: print(f"Error loading ROI config: {e}. Starting empty."); rois = []; next_roi_id = 0; return False

def save_rois(config_path, current_rois):
    """Saves current ROIs (including transform) to config file."""
    try:
        rois_to_save = []
        for roi in current_rois:
             points_list = [[int(p[0]), int(p[1])] for p in roi['points']]
             transform = roi.get('transform', {'rotate': 0, 'flip_code': None})
             if 'flip_code' not in transform: transform['flip_code'] = None
             rois_to_save.append({'id': roi['id'], 'points': points_list, 'transform': transform})
        with open(config_path, 'w') as f: json.dump(rois_to_save, f, indent=4)
        print(f"Saved {len(current_rois)} ROIs to {config_path}")
    except Exception as e: print(f"Error saving ROI config: {e}")

def apply_roi_transform(patch, transform):
    """Applies rotation and flip to an image patch based on transform dict."""
    if patch is None: return None
    transformed_patch = patch.copy(); angle = transform.get('rotate', 0); flip_code = transform.get('flip_code', None)
    if angle == 90: transformed_patch = cv2.rotate(transformed_patch, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180: transformed_patch = cv2.rotate(transformed_patch, cv2.ROTATE_180)
    elif angle == 270: transformed_patch = cv2.rotate(transformed_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if flip_code is not None: transformed_patch = cv2.flip(transformed_patch, flip_code)
    return transformed_patch

def calculate_ocr_stats(ocr_result):
    """
    Calculates average and minimum confidence from PaddleOCR result for a single ROI.
    """
    confidences = []
    if ocr_result and ocr_result[0] is not None:
        for line in ocr_result[0]:
             if line and len(line) == 2:
                text_info = line[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) > 1:
                    confidence = text_info[1]
                    if isinstance(confidence, (float, int)):
                        confidences.append(confidence)
    if not confidences:
        return 0.0, 0.0
    avg_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)
    return avg_conf, min_conf

def estimate_roi_dims(roi_points):
    """Estimates the width and height of the quadrilateral defined by roi_points."""
    try:
        pts = np.array(roi_points, dtype=np.float32)
        dist_01 = np.linalg.norm(pts[0] - pts[1]); dist_12 = np.linalg.norm(pts[1] - pts[2])
        dist_23 = np.linalg.norm(pts[2] - pts[3]); dist_30 = np.linalg.norm(pts[3] - pts[0])
        est_width = (dist_01 + dist_23) / 2.0; est_height = (dist_12 + dist_30) / 2.0
        if est_width < 1e-6 or est_height < 1e-6: # Use a small epsilon instead of just 1
             print(f"Warning: Degenerate ROI points detected (width or height near zero): {roi_points}. Using default dims.")
             return ROI_TARGET_WIDTH, 50 # Return a default aspect ratio
        return est_width, est_height
    except Exception as e: print(f"W: Could not estimate ROI dims: {e}"); return ROI_TARGET_WIDTH, 50

def calculate_roi_angle(roi_points):
    """Estimates the angle of the longer dimension of the ROI quadrilateral."""
    try:
        pts = np.array(roi_points, dtype=np.float32)
        vec12 = pts[1] - pts[0]; vec23 = pts[2] - pts[1]; vec34 = pts[3] - pts[2]; vec41 = pts[0] - pts[3]
        len_sq_12 = vec12[0]**2 + vec12[1]**2; len_sq_23 = vec23[0]**2 + vec23[1]**2
        len_sq_34 = vec34[0]**2 + vec34[1]**2; len_sq_41 = vec41[0]**2 + vec41[1]**2
        sides = [(len_sq_12, vec12), (len_sq_23, vec23), (len_sq_34, vec34), (len_sq_41, vec41)]
        sides.sort(key=lambda item: item[0], reverse=True); longest_vec = sides[0][1]
        angle_rad = math.atan2(longest_vec[1], longest_vec[0]); angle_deg = math.degrees(angle_rad)
        return angle_deg
    except Exception as e: print(f"W: Could not calculate ROI angle: {e}"); return 0

def get_closest_rotation(angle_deg):
    """Determines the best 0/90/180/270 rotation to apply based on the angle."""
    angle = angle_deg % 360
    if (angle >= -45 and angle < 45) or (angle >= 315): return 0
    elif angle >= 45 and angle < 135: return 270
    elif angle >= 135 and angle < 225: return 180
    elif angle >= 225 and angle < 315: return 90
    else: return 0

def enhance_roi_patch(patch):
    """Applies preprocessing (grayscale, contrast enhancement, optional thresholding) to the ROI patch."""
    if patch is None: return None
    enhanced_patch = patch
    if APPLY_GRAYSCALE:
        if len(enhanced_patch.shape) == 3: enhanced_patch = cv2.cvtColor(enhanced_patch, cv2.COLOR_BGR2GRAY)
    if ENHANCE_CONTRAST and len(enhanced_patch.shape) == 2:
        try:
            clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
            enhanced_patch = clahe.apply(enhanced_patch)
        except Exception as e:
            print(f"W: Failed to apply CLAHE: {e}"); return enhanced_patch
    if APPLY_THRESHOLDING and len(enhanced_patch.shape) == 2:
         try: _ , enhanced_patch = cv2.threshold(enhanced_patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
         except Exception as e: print(f"W: Failed to apply thresholding: {e}")
    return enhanced_patch

def post_process_text(text):
    """Applies character filtering and simple regex fixes to common OCR issues."""
    if not text: return ""
    filtered_chars = [char for char in text.upper() if char in TIMESTAMP_CHAR_WHITELIST]
    filtered_text = "".join(filtered_chars)
    fixed_text = re.sub(r'(\d{4}-\d{2}-\d{2})(\d{2}:\d{2})', r'\1 \2', filtered_text)
    fixed_text = re.sub(r'(\d+\.)(\d{4}-)', r'\1 \2', fixed_text)
    fixed_text = re.sub(r'\b(\d{2}:\d{2}:\d{2})(\d{2})\b', r'\1.\2', fixed_text)
    return fixed_text

def optimize_roi_transform(roi_index, frame, ocr_engine):
    """Finds the best flip state for the geometry-estimated rotation."""
    global rois
    if roi_index < 0 or roi_index >= len(rois): return
    roi_info = rois[roi_index]; roi_points = roi_info['points']; roi_id = roi_info['id']
    print(f"--- Optimizing ROI #{roi_id} (Geometry-Guided + Orientation Priority) ---")
    best_score = -1.0; estimated_angle = calculate_roi_angle(roi_points)
    target_rotation = get_closest_rotation(estimated_angle)
    print(f"  Estimated Angle: {estimated_angle:.1f} deg => Target Rotation: {target_rotation} deg")
    best_transform = {'rotate': target_rotation, 'flip_code': None}
    try:
        est_w, est_h = estimate_roi_dims(roi_points);
        if est_w < 1e-6:
             print(f"  Warning: Estimated width for ROI #{roi_id} is near zero. Using default aspect ratio for warp.")
             target_h = 50
        else:
             target_h = max(1, int(ROI_TARGET_WIDTH * (est_h / est_w)))
        roi_dst_points_opt = np.float32([[0, 0], [ROI_TARGET_WIDTH - 1, 0], [ROI_TARGET_WIDTH - 1, target_h - 1], [0, target_h - 1]])
        roi_src_points = np.float32(roi_points)
        matrix = cv2.getPerspectiveTransform(roi_src_points, roi_dst_points_opt)
        initial_warped_patch = cv2.warpPerspective(frame, matrix, (ROI_TARGET_WIDTH, target_h))
        if initial_warped_patch is None: raise ValueError("Warp failed")

        # Check standard orientation first
        standard_transform = {'rotate': 0, 'flip_code': None}
        standard_patch = apply_roi_transform(initial_warped_patch, standard_transform)
        standard_confidence = 0.0
        if standard_patch is not None:
            enhanced_standard_patch = enhance_roi_patch(standard_patch)
            if enhanced_standard_patch is not None:
                ocr_result_standard = ocr_engine.ocr(enhanced_standard_patch, cls=True)
                standard_confidence, _ = calculate_ocr_stats(ocr_result_standard)
                print(f"  Confidence for Standard Orientation (Rot 0, No Flip): {standard_confidence:.6f}")
                if standard_confidence >= NORMAL_ORIENTATION_CONF_THRESHOLD:
                    print(f"  Prioritizing standard orientation."); best_transform = standard_transform.copy(); best_score = standard_confidence
                    rois[roi_index]['transform'] = best_transform; print(f"--- Opt Complete ROI #{roi_id} ---"); print(f"  Best: {best_transform}, Score: {best_score:.6f}"); return

        # If standard not good enough, test flips for target rotation
        print(f"  Standard conf below threshold. Testing flips for target rot {target_rotation}...")
        transform_params_target_noflip = {'rotate': target_rotation, 'flip_code': None}
        transformed_patch_target_noflip = apply_roi_transform(initial_warped_patch, transform_params_target_noflip)
        best_score = 0.0 # Reset best score
        if transformed_patch_target_noflip is not None:
             enhanced_patch_target_noflip = enhance_roi_patch(transformed_patch_target_noflip)
             if enhanced_patch_target_noflip is not None:
                  ocr_result_target_noflip = ocr_engine.ocr(enhanced_patch_target_noflip, cls=True)
                  best_score, _ = calculate_ocr_stats(ocr_result_target_noflip)
                  best_transform = transform_params_target_noflip.copy()

        # Test other flips for the target rotation
        for flip_code_test in [1, 0, -1]: # H, V, Both
            transform_params = {'rotate': target_rotation, 'flip_code': flip_code_test}
            transformed_patch = apply_roi_transform(initial_warped_patch, transform_params)
            if transformed_patch is None: continue
            enhanced_patch_for_ocr = enhance_roi_patch(transformed_patch)
            if enhanced_patch_for_ocr is None: continue
            ocr_result = ocr_engine.ocr(enhanced_patch_for_ocr, cls=True)
            avg_confidence, _ = calculate_ocr_stats(ocr_result)
            if avg_confidence > best_score: best_score = avg_confidence; best_transform = transform_params.copy()

        rois[roi_index]['transform'] = best_transform
        print(f"--- Optimization Complete for ROI #{roi_id} ---")
        print(f"  Best Transform (TargetRot {target_rotation}): {best_transform}, Score: {best_score:.6f}")
    except Exception as e:
        print(f"Error during optimization for ROI #{roi_id}: {e}")
        rois[roi_index]['transform'] = {'rotate': 0, 'flip_code': None}


def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks for defining ROI points."""
    global current_roi_points, defining_roi, rois, next_roi_id, frame_for_opt
    if not defining_roi: return
    if event == cv2.EVENT_LBUTTONDOWN:
        # Unpack params
        ocr_engine, frame_ref_list, scale_x, scale_y = param
        # Scale click coordinates back to original frame size
        original_x = int(x / scale_x)
        original_y = int(y / scale_y)

        if len(current_roi_points) < 4:
            current_roi_points.append([original_x, original_y])
            print(f"ROI point {len(current_roi_points)} added: ({original_x}, {original_y})")
            if len(current_roi_points) == 4:
                new_roi = {'id': next_roi_id, 'points': current_roi_points.copy(), 'transform': {'rotate': 0, 'flip_code': None}}
                rois.append(new_roi); new_roi_index = len(rois) - 1
                print(f"ROI #{next_roi_id} defined. Optimizing transform...")
                current_roi_id = next_roi_id; next_roi_id += 1
                current_frame_for_opt = frame_ref_list[0] # Get the current frame
                if current_frame_for_opt is not None: optimize_roi_transform(new_roi_index, current_frame_for_opt, ocr_engine)
                else: print("Warning: Cannot optimize ROI immediately, no frame available.")
                current_roi_points = []; defining_roi = False
                print(f"ROI #{current_roi_id} optimization done. Press 's' to save all ROIs to config file.")


def process_single_frame_ocr(frame, frame_idx, ocr_engine):
    """
    Performs OCR on all ROIs for a single frame.
    Returns:
        dict: {roi_id: {'text': str, 'confidence': float, 'raw_text': str, 'min_confidence': float}}
        float: Average confidence across all ROIs in this frame (used for best-of-3 selection).
    """
    if frame is None or not rois:
        return {}, 0.0

    frame_results = {}
    total_avg_confidence_sum = 0.0 # Sum of average confidences *per ROI*
    rois_processed_count = 0

    for i, roi_info in enumerate(rois):
        roi_id = roi_info['id']; roi_points = roi_info['points']
        roi_transform = roi_info.get('transform', {'rotate': 0, 'flip_code': None})
        recognized_text_raw = "ERROR"; roi_text_fragments = []
        avg_conf_this_roi = 0.0
        min_conf_this_roi = 0.0
        try:
            if not (isinstance(roi_points, (list, np.ndarray)) and len(roi_points) == 4): continue
            roi_src_points = np.float32(roi_points)
            est_w, est_h = estimate_roi_dims(roi_points)
            if est_w < 1: est_w = 1
            target_h = max(1, int(ROI_TARGET_WIDTH * (est_h / est_w)))
            current_dst_points = np.float32([[0, 0], [ROI_TARGET_WIDTH - 1, 0], [ROI_TARGET_WIDTH - 1, target_h - 1], [0, target_h - 1]])
            matrix = cv2.getPerspectiveTransform(roi_src_points, current_dst_points)
            warped_roi = cv2.warpPerspective(frame, matrix, (ROI_TARGET_WIDTH, target_h))
            optimally_transformed_patch = apply_roi_transform(warped_roi, roi_transform)
            if optimally_transformed_patch is None: raise ValueError("Apply transform failed")
            enhanced_patch_for_ocr = enhance_roi_patch(optimally_transformed_patch)
            if enhanced_patch_for_ocr is None: raise ValueError("Enhancement failed")

            ocr_result = ocr_engine.ocr(enhanced_patch_for_ocr, cls=True)

            # Extract text and calculate average/min confidence for *this ROI*
            avg_conf_this_roi, min_conf_this_roi = calculate_ocr_stats(ocr_result) # Use helper

            # Extract raw text fragments
            if ocr_result and ocr_result[0] is not None:
                 for line in ocr_result[0]:
                    if line and len(line) == 2:
                        text_info = line[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) > 0:
                            roi_text_fragments.append(text_info[0])

            recognized_text_raw = " ".join(roi_text_fragments).strip()
            recognized_text_processed = post_process_text(recognized_text_raw)

        except Exception as e:
            print(f"\nError processing ROI ID {roi_id} on frame {frame_idx}: {e}");
            recognized_text_processed = f'Error: {e}'
            recognized_text_raw = "ERROR" # Ensure raw text reflects error too
            avg_conf_this_roi = 0.0 # Assign 0 confidence on error
            min_conf_this_roi = 0.0

        # Store detailed results for this ROI
        frame_results[roi_id] = {
            'text': recognized_text_processed,
            'confidence': avg_conf_this_roi, # Average confidence for this ROI in this frame
            'raw_text': recognized_text_raw,
            'min_confidence': min_conf_this_roi # Minimum line confidence for this ROI in this frame
        }
        total_avg_confidence_sum += avg_conf_this_roi # Sum average confidences for overall average later
        rois_processed_count += 1

    # Calculate average confidence for this frame across all ROIs processed
    overall_avg_confidence = total_avg_confidence_sum / rois_processed_count if rois_processed_count > 0 else 0.0
    return frame_results, overall_avg_confidence

# --- Main Processing Function ---
frame_for_opt = None # Global to hold frame for optimization callback

def process_video_file(video_path, csv_writer, ocr_engine, sample_interval_sec=1.0, reoptimize_interval_sec=0, display_enabled=True):
    """
    Processes a single video file with time-based sampling and frame skipping display.
    Selects the best frame among [target-1, target, target+1] based on ROI wins.
    Handles ROI processing, OCR, display, and CSV writing.
    Returns True if successful, False otherwise.
    """
    global rois, current_roi_points, defining_roi, paused, next_roi_id, frame_for_opt, show_warped

    print(f"\n===== Processing Video: {video_path} =====")
    base_filename = os.path.basename(video_path)

    # --- Video Loading ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'. Skipping.")
        return False

    # --- Get Video Properties ---
    fps = cap.get(cv2.CAP_PROP_FPS); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_duration_sec = (total_frames / fps) if fps and fps > 0 else 0

    # --- FPS Check ---
    if fps <= 0:
        print(f"Error: Video FPS is {fps} for {video_path}. Cannot process. Skipping.")
        cap.release()
        return False

    print(f"  Resolution: {frame_width}x{frame_height}, Total Frames: {total_frames}, FPS: {fps:.2f}")
    if total_duration_sec > 0: print(f"  Estimated Duration: {timedelta(seconds=total_duration_sec)}")
    print(f"  OCR Sampling Interval: {sample_interval_sec} seconds")
    if reoptimize_interval_sec > 0: print(f"  Re-optimization Interval: {reoptimize_interval_sec} seconds")


    # --- Read First Frame (needed for potential optimization) ---
    ret, first_frame = cap.read()
    if not ret: print("Error: Could not read the first frame."); cap.release(); return False
    frame_for_opt = first_frame # Make available globally

    # --- Optimize ROIs if needed (only if loaded, new ones optimized on creation) ---
    # Optimization now happens on load or when new ROI is defined.
    # Re-optimization can be triggered with 'o' key.

    # --- Setup OpenCV Window and Mouse Callback (if display enabled) ---
    window_name = f"ROI Definition & OCR Output - {base_filename}"
    frame_ref = None # Initialize defensively
    if display_enabled:
        cv2.namedWindow(window_name)
        # Calculate scale factors for mouse callback
        display_height = int(frame_height * (DISPLAY_WIDTH / frame_width))
        scale_x = DISPLAY_WIDTH / frame_width
        scale_y = display_height / frame_height

        frame_ref = [first_frame] # Use mutable list as a reference
        cv2.setMouseCallback(window_name, mouse_callback, param=(ocr_engine, frame_ref, scale_x, scale_y))
        print("\n  --- Controls ---")
        print("   Mouse Click: Define ROI corners (after pressing 'n')")
        print("    n: Start NEW ROI | d: DELETE last | c: CLEAR all")
        print("    s: SAVE ROIs | l: LOAD ROIs | o: Re-OPTIMIZE all ROIs")
        print("    v: Toggle VIEW warped/optimized patches (when PAUSED)")
        print("    SPACE: PAUSE / RESUME | q: QUIT (this video)")
        print("  ----------------")
    # Reset state variables for each video
    paused = False; defining_roi = False; current_roi_points = []; show_warped = False


    # --- Processing Loop for this video file ---
    processed_ocr_count = 0; total_ocr_time = 0
    current_frame = first_frame # Holds the frame currently displayed or being processed
    last_processed_time_sec = -sample_interval_sec # Ensures first frame can be processed
    last_reoptimize_time_sec = 0.0 # Time of last periodic re-optimization

    while True:
        # --- Handle Paused State ---
        if paused:
            if display_enabled:
                # Keep displaying the last frame and handling keys while paused
                display_frame = current_frame.copy() # Use the frame captured before pausing
                # Draw ROIs, points being defined, PAUSED text etc.
                for i, roi_info in enumerate(rois):
                    color_tuple, color_name = ROI_COLORS[i % len(ROI_COLORS)]
                    try: points = np.array(roi_info['points'], dtype=np.int32).reshape((-1, 1, 2)); cv2.polylines(display_frame, [points], isClosed=True, color=color_tuple, thickness=1); cv2.putText(display_frame, str(roi_info['id']), tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_tuple, 1)
                    except Exception as draw_e: print(f"W: Could not draw ROI ID {roi_info['id']}: {draw_e}")
                if defining_roi:
                    for idx, pt in enumerate(current_roi_points): cv2.circle(display_frame, tuple(pt), 3, (0, 255, 255), -1); cv2.putText(display_frame, str(idx+1), (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
                    prompt_y_pos = 30; prompt_text = "";
                    if len(current_roi_points) < 4: prompt_text = f"Click point {len(current_roi_points)+1}/4 for ROI ID {next_roi_id}"
                    if prompt_text: cv2.putText(display_frame, prompt_text, (10, prompt_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, "PAUSED", (frame_width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if show_warped: cv2.putText(display_frame, "Showing Warped Patches ('v' pressed)", (frame_width - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Resize before showing
                resized_display = cv2.resize(display_frame, (DISPLAY_WIDTH, display_height), interpolation=cv2.INTER_AREA)
                cv2.imshow(window_name, resized_display)

                # Show warped patches if requested ('v' key while paused)
                if show_warped and rois:
                    # ... (Code to show warped patches remains the same) ...
                    print("Displaying warped/optimized/enhanced patches...")
                    patch_windows = []
                    for i, roi_info in enumerate(rois):
                        roi_id = roi_info['id']; roi_points = roi_info['points']
                        roi_transform = roi_info.get('transform', {'rotate': 0, 'flip_code': None})
                        try:
                            roi_src_points = np.float32(roi_points); est_w, est_h = estimate_roi_dims(roi_points)
                            if est_w < 1: est_w = 1
                            target_h = max(1, int(ROI_TARGET_WIDTH * (est_h / est_w)))
                            current_dst_points = np.float32([[0, 0], [ROI_TARGET_WIDTH - 1, 0], [ROI_TARGET_WIDTH - 1, target_h - 1], [0, target_h - 1]])
                            matrix = cv2.getPerspectiveTransform(roi_src_points, current_dst_points)
                            warped_roi = cv2.warpPerspective(current_frame, matrix, (ROI_TARGET_WIDTH, target_h))
                            optimally_transformed_patch = apply_roi_transform(warped_roi, roi_transform)
                            enhanced_display_patch = enhance_roi_patch(optimally_transformed_patch) # Show enhanced one
                            patch_window_name = f"ROI {roi_id} Enhanced (Optimal: {roi_transform})"
                            patch_windows.append(patch_window_name)
                            cv2.imshow(patch_window_name, enhanced_display_patch)
                        except Exception as patch_e: print(f"Could not display patch for ROI {roi_id}: {patch_e}")
                    print("Press any key in a patch window to close all patch windows.")
                    cv2.waitKey(0); show_warped = False
                    for w_name in patch_windows:
                         try: cv2.destroyWindow(w_name)
                         except: pass

            # Handle keys while paused
            key_pressed = cv2.waitKey(1) & 0xFF # Still need waitKey for interaction
            # Process keys (q, s, l, n, d, c, o, v, SPACE) - same logic as below
            if key_pressed == ord('q'): break
            elif key_pressed == ord('s'): save_rois(ROI_CONFIG_FILE, rois)
            elif key_pressed == ord('l'): # Load and Optimize
                rois_loaded = load_rois(ROI_CONFIG_FILE); current_roi_points = []; defining_roi = False
                if rois_loaded and rois and current_frame is not None:
                    print("Optimizing transforms for reloaded ROIs...")
                    for i in range(len(rois)): optimize_roi_transform(i, current_frame, ocr_engine)
                    print("Finished optimizing reloaded ROIs.")
                elif not rois: print("Config file loaded, but no valid ROIs found.")
                else: print("Could not optimize loaded ROIs (no current frame?).")
            elif key_pressed == ord('n'): # New ROI
                if defining_roi and current_roi_points: print("Cancelled defining current ROI.")
                print("\nStarting new ROI definition: Click 4 points on the window.")
                current_roi_points = []; defining_roi = True
                # Already paused
            elif key_pressed == ord('d'): # Delete last ROI
                if rois: deleted_roi = rois.pop(); print(f"Deleted last ROI (ID {deleted_roi['id']}).")
                else: print("No ROIs to delete.")
                if defining_roi: current_roi_points = []; defining_roi = False
            elif key_pressed == ord('c'): # Clear all ROIs
                print("Cleared all ROIs."); rois = []; current_roi_points = []; defining_roi = False; next_roi_id = 0
            elif key_pressed == ord('o'): # Optimize all current ROIs
                 if rois and current_frame is not None:
                     print("Re-optimizing transforms for all current ROIs...")
                     for i in range(len(rois)): optimize_roi_transform(i, current_frame, ocr_engine)
                     print("Finished re-optimizing ROIs.")
                 elif not rois: print("No ROIs defined to optimize.")
                 else: print("Could not optimize ROIs (no current frame?).")
            elif key_pressed == ord('v'): # Toggle warped patch view
                 show_warped = True; print("Will show enhanced/optimized patches now (press any key in patch window to close).")
            elif key_pressed == ord(' '): # Toggle Pause
                paused = not paused; status_msg = "Paused." if paused else "Resumed."; print(f"\n{status_msg}")
                if not paused and defining_roi: current_roi_points = []; defining_roi = False; print("Cancelling ROI definition.")
                if not paused: # Destroy patch windows on resume
                     for i, roi_info in enumerate(rois):
                         try: cv2.destroyWindow(f"ROI {roi_info['id']} Enhanced (Optimal: {roi_info.get('transform')})")
                         except: pass
                     show_warped = False
            continue # Skip the rest of the loop if paused

        # --- If not paused, calculate next sample time and seek ---
        next_sample_time_sec = last_processed_time_sec + sample_interval_sec
        target_frame_number = int(next_sample_time_sec * fps)

        if target_frame_number >= total_frames:
            print("\nReached end of video based on sampling interval.")
            break

        # --- Seek and Read the Target Frame ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_number)
        ret, frame = cap.read()
        if not ret:
            print(f"\nWarning: Could not read target frame {target_frame_number}. End of video?")
            break
        current_frame = frame
        actual_processed_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if display_enabled and frame_ref is not None:
            frame_ref[0] = current_frame

        # --- Periodic Re-optimization Check ---
        current_processed_time_sec = actual_processed_frame_num / fps
        if reoptimize_interval_sec > 0 and (current_processed_time_sec - last_reoptimize_time_sec >= reoptimize_interval_sec):
            print(f"\n--- Periodic Re-optimization Triggered at {format_timedelta(timedelta(seconds=current_processed_time_sec))} ---")
            for i in range(len(rois)):
                optimize_roi_transform(i, current_frame, ocr_engine)
            last_reoptimize_time_sec = current_processed_time_sec


        # --- OCR Processing for the Sampled Frame ---
        processed_ocr_count += 1
        last_processed_time_sec = actual_processed_frame_num / fps
        video_timestamp_str = format_timedelta(timedelta(seconds=last_processed_time_sec))
        progress_percent = (actual_processed_frame_num / total_frames * 100) if total_frames > 0 else 0
        print(f"\n--- Processing OCR Frame {actual_processed_frame_num} / {total_frames} ({progress_percent:.1f}%) | Video Time: {video_timestamp_str} ---")

        ocr_start_time = time.time()
        if rois:
            print(f"  Processing {len(rois)} ROIs...")
            for i, roi_info in enumerate(rois):
                roi_id = roi_info['id']; roi_points = roi_info['points']
                roi_transform = roi_info.get('transform', {'rotate': 0, 'flip_code': None})
                try:
                    est_w, est_h = estimate_roi_dims(roi_points)
                    if est_w < 1: est_w = 1
                    target_h = max(1, int(ROI_TARGET_WIDTH * (est_h / est_w)))
                    dst_pts = np.float32([[0, 0], [ROI_TARGET_WIDTH - 1, 0], [ROI_TARGET_WIDTH - 1, target_h - 1], [0, target_h - 1]])
                    matrix = cv2.getPerspectiveTransform(np.float32(roi_points), dst_pts)
                    warped_roi = cv2.warpPerspective(current_frame, matrix, (ROI_TARGET_WIDTH, target_h))
                    optimally_transformed_patch = apply_roi_transform(warped_roi, roi_transform)
                    if optimally_transformed_patch is None: raise ValueError("Apply transform failed")
                    enhanced_patch_for_ocr = enhance_roi_patch(optimally_transformed_patch)
                    if enhanced_patch_for_ocr is None: raise ValueError("Enhancement failed")
                    ocr_result = ocr_engine.ocr(enhanced_patch_for_ocr, cls=True)
                    avg_conf, _ = calculate_ocr_stats(ocr_result)

                    text_fragments = []
                    if ocr_result and ocr_result[0] is not None:
                        for line in ocr_result[0]:
                            text_fragments.append(line[1][0])
                    raw_text = " ".join(text_fragments).strip()
                    rec_text = post_process_text(raw_text)

                    print(f"    > ROI ID {roi_id}: Text = '{rec_text}' (Raw: '{raw_text}') (Conf: {avg_conf:.6f}) (Transform: {roi_transform})")
                    row_data = [video_timestamp_str, actual_processed_frame_num, base_filename, roi_id, rec_text, raw_text, f"{avg_conf:.6f}", str(roi_transform)]
                    csv_writer.writerow(row_data)
                except Exception as e:
                    print(f"\nError processing ROI ID {roi_id}: {e}")
                    csv_writer.writerow([video_timestamp_str, actual_processed_frame_num, base_filename, roi_id, f'Error: {e}', '', 0.0, ''])
            total_ocr_time += (time.time() - ocr_start_time)

        # --- Display Sampled Frame (if enabled) ---
        if display_enabled:
            display_frame = current_frame.copy()
            for i, roi_info in enumerate(rois):
                color_tuple, _ = ROI_COLORS[i % len(ROI_COLORS)]
                try:
                    points = np.array(roi_info['points'], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [points], isClosed=True, color=color_tuple, thickness=1)
                    cv2.putText(display_frame, str(roi_info['id']), tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_tuple, 1)
                except: pass
            resized_display = cv2.resize(display_frame, (DISPLAY_WIDTH, display_height), interpolation=cv2.INTER_AREA)
            cv2.imshow(window_name, resized_display)

        # --- Handle Key Press ---
        key_pressed = -1
        if display_enabled:
            key_pressed = cv2.waitKey(1) & 0xFF
        elif not paused:
             time.sleep(0.001)

        if key_pressed == ord('q'):
             print(f"\n'q' pressed, stopping.")
             break # Exit loop for this video file
        elif key_pressed == ord(' '):
            paused = not paused
            print(f"\n{'Paused' if paused else 'Resumed'}.")


    # --- Cleanup for this video file ---
    cap.release()
    if display_enabled:
        try: cv2.destroyWindow(window_name)
        except: pass
    print(f"Finished processing {base_filename}. Processed OCR on {processed_ocr_count} frames.")
    if processed_ocr_count > 0 and rois:
        avg_ocr_time = total_ocr_time / (processed_ocr_count * len(rois)) if len(rois) > 0 else 0
        print(f"  Average OCR time per ROI per sampled frame: {avg_ocr_time:.4f} seconds")
    return True


# --- Script Execution Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from ROIs in video files using OCR.")
    parser.add_argument("input_paths", nargs='+', help="Path(s) to video file(s), directory, or glob pattern.")
    parser.add_argument("-o", "--output-csv", default="ocr_results.csv", help="Path for the output CSV file.")
    parser.add_argument("-i", "--interval", type=float, default=1.0, dest='sample_interval_sec', help="Seconds between OCR attempts.")
    parser.add_argument("-t", "--reoptimize-interval", type=float, default=0, dest='reoptimize_interval_sec', help="Re-optimize ROI transforms every N seconds (0 to disable).")
    parser.add_argument("-r", "--roi-config", default='roi_config.json', dest='roi_config_file', help="Configuration file for ROIs (default: 'roi_config.json').")
    parser.add_argument("--gpu-id", type=int, default=0, help="ID of the GPU to use.")
    parser.add_argument("--no-display", action="store_true", help="Disable the GUI display window.")
    parser.add_argument("--force-cpu", action="store_true", help="Force PaddleOCR to use CPU.")
    parser.add_argument("--threshold", action="store_true", dest="apply_thresholding", help="Apply Otsu's thresholding.")
    args = parser.parse_args()

    # (The rest of the __main__ block remains the same as previous version)
    if args.sample_interval_sec <= 0: args.sample_interval_sec = 1.0
    ENABLE_DISPLAY = not args.no_display
    FORCE_CPU = args.force_cpu
    APPLY_THRESHOLDING = args.apply_thresholding
    ROI_CONFIG_FILE = args.roi_config_file
    video_files = []
    for path_pattern in args.input_paths:
        if os.path.isdir(path_pattern): video_files.extend(glob.glob(os.path.join(path_pattern, '**', '*.avi'), recursive=True))
        elif '*' in path_pattern or '?' in path_pattern: video_files.extend(glob.glob(path_pattern, recursive=True))
        elif os.path.isfile(path_pattern): video_files.append(path_pattern)
        else: print(f"W: Input path '{path_pattern}' is not valid. Skipping.")
    video_files = sorted(list(set(video_files)))
    if not video_files: print(f"No valid video files found. Exiting."); sys.exit(0)

    print(f"\n--- Configuration ---")
    print(f"Input(s): {', '.join(args.input_paths)}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Sample Interval: {args.sample_interval_sec} seconds")
    print(f"Re-optimize Interval: {args.reoptimize_interval_sec if args.reoptimize_interval_sec > 0 else 'Disabled'} seconds")
    print(f"Display: {ENABLE_DISPLAY}, Force CPU: {FORCE_CPU}, Apply Threshold: {APPLY_THRESHOLDING}")
    print(f"ROI Config File: {ROI_CONFIG_FILE}")
    if not FORCE_CPU: print(f"GPU ID: {args.gpu_id}")
    print(f"Found {len(video_files)} video file(s):"); [print(f"  - {f}") for f in video_files]
    print(f"--------------------\n")

    if not FORCE_CPU:
        try:
            if paddle.is_compiled_with_cuda():
                device_count = paddle.device.cuda.device_count()
                if device_count > 0:
                    if args.gpu_id >= device_count: print(f"W: GPU ID {args.gpu_id} out of range. Using GPU 0."); args.gpu_id = 0
                else: print("W: No GPUs found by Paddle. Forcing CPU."); FORCE_CPU = True
            else: print("W: Paddle not compiled with CUDA. Using CPU."); FORCE_CPU = True
        except Exception as e: print(f"W: Error getting GPU info: {e}. Using CPU."); FORCE_CPU = True

    print("\nInitializing PaddleOCR...")
    try:
        gpu_id_to_use = args.gpu_id if not FORCE_CPU else 0
        ocr_engine_main = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', use_gpu=(not FORCE_CPU), gpu_id=gpu_id_to_use, show_log=False, det_db_unclip_ratio=PADDLE_UNCLIP_RATIO, det_db_thresh=PADDLE_DB_THRESH, det_db_box_thresh=PADDLE_BOX_THRESH)
        print("PaddleOCR initialized.");
        if FORCE_CPU: print("PaddleOCR running on CPU.")
        else: print(f"PaddleOCR attempting GPU {gpu_id_to_use}.")
    except Exception as e: print(f"Error initializing PaddleOCR: {e}"); sys.exit(1)

    rois_loaded = load_rois(ROI_CONFIG_FILE)
    if not rois_loaded and ENABLE_DISPLAY: print("No ROI config found. Define ROIs for the first video.")
    elif not rois_loaded and not ENABLE_DISPLAY: print("Error: No ROI config found and display is disabled. Exiting."); sys.exit(1)

    csv_headers = ['Video Time', 'Frame', 'Filename', 'ROI ID', 'Recognized Text', 'Raw Text', 'Confidence', 'Best Transform']
    file_exists = os.path.exists(args.output_csv)
    csvfile = None; csv_writer = None
    try:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        csvfile = open(args.output_csv, 'a', newline='', encoding='utf-8')
        csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        if not file_exists or os.path.getsize(args.output_csv) == 0:
            csv_writer.writerow(csv_headers); print(f"\nWriting new CSV file: {args.output_csv}")
        else: print(f"\nAppending to existing CSV file: {args.output_csv}")
    except IOError as e: print(f"Error opening CSV file {args.output_csv}: {e}"); sys.exit(1)

    overall_success = True
    try:
        for video_file_path in video_files:
            paused = False
            success = process_video_file(video_file_path, csv_writer, ocr_engine_main, sample_interval_sec=args.sample_interval_sec, reoptimize_interval_sec=args.reoptimize_interval_sec, display_enabled=ENABLE_DISPLAY)
            if not success: overall_success = False
    except Exception as e:
        print(f"\nAn critical error occurred during processing: {e}")
        print("\n----- Traceback -----"); traceback.print_exc(); print("---------------------\n")
        overall_success = False
    finally:
        if csvfile and not csvfile.closed: csvfile.close(); print("CSV file closed.")
        if ENABLE_DISPLAY:
            try: cv2.destroyAllWindows()
            except Exception: pass

    print("\n===== Processing Complete =====")
    if not overall_success: print("Some files may not have been processed successfully."); sys.exit(1)
    else: print("All files processed successfully."); sys.exit(0)
