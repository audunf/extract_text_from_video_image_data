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
# --- >>> GUI Toggle <<< ---
# ENABLE_DISPLAY = True # Controlled by command-line arg now
# --- >>> CPU/GPU Toggle <<< ---
# FORCE_CPU = False # Controlled by command-line arg now
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
APPLY_THRESHOLDING = False # Optional: Apply Otsu's thresholding after CLAHE
# --- >>> PaddleOCR Tuning Parameters <<< ---
PADDLE_UNCLIP_RATIO = 2.0
PADDLE_BOX_THRESH = 0.6
PADDLE_DB_THRESH = 0.3
# --- >>> Optimization Parameters <<< ---
NORMAL_ORIENTATION_CONF_THRESHOLD = 0.85
# --- >>> Best-of-3 Sampling Parameters <<< ---
CHECK_ADJACENT_CONF_THRESHOLD = 1.1 # Currently not used in ROI-Win logic

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
    Args:
        ocr_result: The direct output from paddle_ocr.ocr() for a single image patch.
                    Expected format: [[line1], [line2], ...] where line = [[box], (text, confidence)]
                    OR None if no text detected.
    Returns:
        tuple: (average_confidence, minimum_confidence)
               Returns (0.0, 0.0) if no valid confidence scores found.
    """
    confidences = []
    # --- FIX: Handle the outer list structure from PaddleOCR ---
    if ocr_result and ocr_result[0] is not None:
        for line in ocr_result[0]: # Iterate through the actual lines detected
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
        if len(enhanced_patch.shape) == 3 and enhanced_patch.shape[2] == 3:
            enhanced_patch = cv2.cvtColor(enhanced_patch, cv2.COLOR_BGR2GRAY)
        # else: already grayscale or invalid

    if ENHANCE_CONTRAST and len(enhanced_patch.shape) == 2: # CLAHE needs grayscale
        try:
            clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
            enhanced_patch = clahe.apply(enhanced_patch)
        except Exception as e:
            print(f"Warning: Failed to apply CLAHE enhancement: {e}")
            # Fallback to original grayscale if CLAHE fails
            if len(patch.shape) == 3 and patch.shape[2] == 3:
                 enhanced_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            elif len(patch.shape) == 2:
                 enhanced_patch = patch # Use original grayscale if already gray
            else: # If original wasn't color or gray, return original
                 enhanced_patch = patch

    if APPLY_THRESHOLDING and len(enhanced_patch.shape) == 2: # Thresholding needs grayscale
         try:
             # Apply Otsu's thresholding
             _ , enhanced_patch = cv2.threshold(enhanced_patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
         except Exception as e:
             print(f"Warning: Failed to apply thresholding: {e}")
             # Revert to previous state (grayscale or CLAHE'd) if thresholding fails
             # This requires re-applying grayscale if contrast wasn't applied
             if APPLY_GRAYSCALE and not ENHANCE_CONTRAST:
                 if len(patch.shape) == 3 and patch.shape[2] == 3:
                     enhanced_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                 elif len(patch.shape) == 2:
                     enhanced_patch = patch
                 else:
                     enhanced_patch = patch # Fallback to original
             # If CLAHE was applied, enhanced_patch already holds that result
             # If neither was applied, enhanced_patch holds original

    return enhanced_patch


def post_process_text(text):
    """Applies simple regex fixes to common OCR spacing/punctuation issues."""
    if not text: return ""
    text = re.sub(r'(\d{4}-\d{2}-\d{2})(\d{2}:\d{2})', r'\1 \2', text) # DateTime space
    text = re.sub(r'(\d+\.)(\d{4}-)', r'\1 \2', text) # PrefixNumber Date space
    text = re.sub(r'\b(\d{2}:\d{2}:\d{2})(\d{2})\b', r'\1.\2', text) # SSms -> SS.ms
    return text

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
                standard_confidence, _ = calculate_ocr_stats(ocr_result_standard) # Use new helper
                print(f"  Confidence for Standard Orientation (Rot 0, No Flip): {standard_confidence:.6f}") # Increased precision
                if standard_confidence >= NORMAL_ORIENTATION_CONF_THRESHOLD:
                    print(f"  Prioritizing standard orientation."); best_transform = standard_transform.copy(); best_score = standard_confidence
                    rois[roi_index]['transform'] = best_transform; print(f"--- Opt Complete ROI #{roi_id} ---"); print(f"  Best: {best_transform}, Score: {best_score:.6f}"); return # Increased precision

        # If standard not good enough, test flips for target rotation
        print(f"  Standard conf below threshold. Testing flips for target rot {target_rotation}...")
        transform_params_target_noflip = {'rotate': target_rotation, 'flip_code': None}
        transformed_patch_target_noflip = apply_roi_transform(initial_warped_patch, transform_params_target_noflip)
        best_score = 0.0 # Reset best score
        if transformed_patch_target_noflip is not None:
             enhanced_patch_target_noflip = enhance_roi_patch(transformed_patch_target_noflip)
             if enhanced_patch_target_noflip is not None:
                  ocr_result_target_noflip = ocr_engine.ocr(enhanced_patch_target_noflip, cls=True)
                  best_score, _ = calculate_ocr_stats(ocr_result_target_noflip) # Use new helper
                  best_transform = transform_params_target_noflip.copy()

        # Test other flips for the target rotation
        for flip_code_test in [1, 0, -1]: # H, V, Both
            transform_params = {'rotate': target_rotation, 'flip_code': flip_code_test}
            transformed_patch = apply_roi_transform(initial_warped_patch, transform_params)
            if transformed_patch is None: continue
            enhanced_patch_for_ocr = enhance_roi_patch(transformed_patch)
            if enhanced_patch_for_ocr is None: continue
            ocr_result = ocr_engine.ocr(enhanced_patch_for_ocr, cls=True)
            avg_confidence, _ = calculate_ocr_stats(ocr_result) # Use new helper
            if avg_confidence > best_score: best_score = avg_confidence; best_transform = transform_params.copy()

        rois[roi_index]['transform'] = best_transform
        print(f"--- Optimization Complete for ROI #{roi_id} ---")
        print(f"  Best Transform (TargetRot {target_rotation}): {best_transform}, Score: {best_score:.6f}") # Increased precision
    except Exception as e:
        print(f"Error during optimization for ROI #{roi_id}: {e}")
        rois[roi_index]['transform'] = {'rotate': 0, 'flip_code': None}


def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks for defining ROI points."""
    global current_roi_points, defining_roi, rois, next_roi_id, frame_for_opt
    if not defining_roi: return
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if len(current_roi_points) < 4:
            current_roi_points.append([x, y])
            print(f"ROI point {len(current_roi_points)} added: ({x}, {y})")
            if len(current_roi_points) == 4:
                new_roi = {'id': next_roi_id, 'points': current_roi_points.copy(), 'transform': {'rotate': 0, 'flip_code': None}}
                rois.append(new_roi); new_roi_index = len(rois) - 1
                print(f"ROI #{next_roi_id} defined. Optimizing transform...")
                current_roi_id = next_roi_id; next_roi_id += 1
                ocr_engine, current_frame_for_opt = param
                if current_frame_for_opt is not None: optimize_roi_transform(new_roi_index, current_frame_for_opt, ocr_engine)
                else: print("Warning: Cannot optimize ROI immediately, no frame available.")
                current_roi_points = []; defining_roi = False
                print(f"ROI #{current_roi_id} optimization done. Press 'n' for new ROI, 'd' to delete last, 's' to save.")


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

def process_video_file(video_path, csv_writer, ocr_engine, sample_interval_sec=1.0, display_enabled=True):
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


    # --- Read First Frame (needed for potential optimization) ---
    ret, first_frame = cap.read()
    if not ret: print("Error: Could not read the first frame."); cap.release(); return False
    frame_for_opt = first_frame # Make available globally

    # --- Optimize ROIs if needed (only if loaded, new ones optimized on creation) ---
    # Optimization now happens on load or when new ROI is defined.
    # Re-optimization can be triggered with 'o' key.

    # --- Setup OpenCV Window and Mouse Callback (if display enabled) ---
    window_name = f"ROI Definition & OCR Output - {base_filename}"
    if display_enabled:
        cv2.namedWindow(window_name)
        frame_ref = [first_frame]
        cv2.setMouseCallback(window_name, mouse_callback, param=(ocr_engine, frame_ref))
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
    next_sample_time_sec = 0.0 # Time for the next sample

    while True:
        # --- Handle Paused State ---
        if paused:
            if display_enabled:
                # Keep displaying the last frame and handling keys while paused
                display_frame = current_frame.copy() # Use the frame captured before pausing
                # Draw ROIs, points being defined, PAUSED text etc.
                for i, roi_info in enumerate(rois):
                    color_tuple, color_name = ROI_COLORS[i % len(ROI_COLORS)]
                    try: points = np.array(roi_info['points'], dtype=np.int32).reshape((-1, 1, 2)); cv2.polylines(display_frame, [points], isClosed=True, color=color_tuple, thickness=2); cv2.putText(display_frame, str(roi_info['id']), tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_tuple, 2)
                    except Exception as draw_e: print(f"W: Could not draw ROI ID {roi_info['id']}: {draw_e}")
                if defining_roi:
                    for idx, pt in enumerate(current_roi_points): cv2.circle(display_frame, tuple(pt), 5, (0, 255, 255), -1); cv2.putText(display_frame, str(idx+1), (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                    prompt_y_pos = 30; prompt_text = "";
                    if len(current_roi_points) < 4: prompt_text = f"Click point {len(current_roi_points)+1}/4 for ROI ID {next_roi_id}"
                    if prompt_text: cv2.putText(display_frame, prompt_text, (10, prompt_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, "PAUSED", (frame_width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if show_warped: cv2.putText(display_frame, "Showing Warped Patches ('v' pressed)", (frame_width - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(window_name, display_frame)

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

        # Ensure target frame is valid
        target_frame_number = max(0, target_frame_number)
        if target_frame_number >= total_frames:
            print("\nTarget frame exceeds total frames.")
            break

        # --- Seek and Read the Target Frame and Neighbors ---
        frames_to_check = {} # {frame_idx: frame_object}
        neighbor_indices = [target_frame_number - 1, target_frame_number, target_frame_number + 1]

        for frame_idx_to_read in neighbor_indices:
            # Skip invalid frame numbers (negative or beyond end)
            if frame_idx_to_read < 0 or frame_idx_to_read >= total_frames:
                continue

            # print(f"Seeking/Reading frame {frame_idx_to_read}") # Debug
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_to_read)
            ret, frame = cap.read()
            if ret:
                frames_to_check[frame_idx_to_read] = frame
            else:
                print(f"Warning: Failed to read frame {frame_idx_to_read}")

        if not frames_to_check:
             print(f"Warning: Could not read target frame {target_frame_number} or its neighbors. Skipping sample.")
             last_processed_time_sec = next_sample_time_sec # Advance time anyway to avoid getting stuck
             continue


        # --- OCR Processing and Best Frame Selection ---
        best_frame_index = -1
        best_frame_results = {}
        best_frame_object = None
        highest_win_count = -1
        frame_wins = {idx: 0 for idx in frames_to_check.keys()} # {frame_idx: win_count}
        frame_data = {} # {frame_idx: {'results': roi_results_dict, 'avg_conf': avg_conf}}
        # --- Store max confidences per ROI across checked frames ---
        roi_max_confidences = {} # {roi_id: max_avg_confidence_found}


        ocr_start_time = time.time()

        # Process each candidate frame
        for frame_idx, frame_obj in frames_to_check.items():
            roi_results, avg_conf = process_single_frame_ocr(frame_obj, frame_idx, ocr_engine)
            frame_data[frame_idx] = {'results': roi_results, 'avg_conf': avg_conf} # Store results and confidences

        # Compare ROI confidences across frames
        if len(frames_to_check) > 1 and rois:
            print(f"  Comparing frames: {list(frames_to_check.keys())}")
            # Use set of all ROI IDs present in the results across frames being checked
            all_roi_ids_in_frames = set()
            for fd in frame_data.values():
                all_roi_ids_in_frames.update(fd['results'].keys())

            for roi_id in all_roi_ids_in_frames:
                best_conf_for_roi = -1.0
                best_frame_for_roi = -1

                for frame_idx in frames_to_check.keys():
                    # Check if this ROI ID exists in this frame's results
                    if roi_id in frame_data[frame_idx]['results']:
                        roi_conf = frame_data[frame_idx]['results'][roi_id]['confidence']
                        if roi_conf > best_conf_for_roi:
                            best_conf_for_roi = roi_conf
                            best_frame_for_roi = frame_idx
                        # --- Store the max confidence found for this ROI ---
                        roi_max_confidences[roi_id] = max(roi_max_confidences.get(roi_id, -1.0), roi_conf)


                if best_frame_for_roi != -1:
                    frame_wins[best_frame_for_roi] += 1
                    # print(f"    ROI {roi_id} best in frame {best_frame_for_roi} (Conf: {best_conf_for_roi:.6f})") # Debug


            # Find the frame with the most wins
            best_frame_index = max(frame_wins, key=frame_wins.get)
            highest_win_count = frame_wins[best_frame_index]
            print(f"  Selected best frame: {best_frame_index} (won {highest_win_count} ROIs)")

        elif frames_to_check: # Only one frame was checked
            best_frame_index = list(frames_to_check.keys())[0]
            print(f"  Using single processed frame: {best_frame_index}")
            # Populate max confidences for the single frame
            if best_frame_index in frame_data:
                 for roi_id, res_data in frame_data[best_frame_index]['results'].items():
                     roi_max_confidences[roi_id] = res_data['confidence']


        # Get the data for the selected best frame
        if best_frame_index != -1:
            best_frame_results = frame_data[best_frame_index]['results']
            best_frame_object = frames_to_check[best_frame_index]
            current_frame = best_frame_object # Update current frame for display/next pause
            if display_enabled: frame_ref[0] = current_frame
            actual_processed_frame_num = best_frame_index
        else:
            # Fallback if something went wrong
            print("Error: Could not determine best frame.")
            actual_processed_frame_num = target_frame_number # Fallback
            best_frame_results = {} # Empty results
            best_frame_object = frames_to_check.get(target_frame_number)
            if best_frame_object: current_frame = best_frame_object # Update if possible
            if display_enabled and current_frame: frame_ref[0] = current_frame


        total_ocr_time += (time.time() - ocr_start_time)
        processed_ocr_count += 1 # Count this sample interval

        # --- Write results for the BEST frame to CSV ---
        current_processed_time_sec = actual_processed_frame_num / fps
        video_timestamp_str = format_timedelta(timedelta(seconds=current_processed_time_sec))
        progress_percent = (actual_processed_frame_num / total_frames * 100) if total_frames > 0 else 0

        print(f"\n--- Writing results for Frame {actual_processed_frame_num} / {total_frames} ({progress_percent:.1f}%) | Video Time: {video_timestamp_str} ---")
        if best_frame_results:
             print(f"  Processing {len(best_frame_results)} ROIs...") # Print header every time
             for roi_id, result_data in best_frame_results.items():
                 # --- Calculate Min Avg Confidence Across Frames for this ROI ---
                 min_avg_conf_across_frames = 1.0 # Initialize high
                 for frame_idx in frames_to_check.keys():
                     if roi_id in frame_data[frame_idx]['results']:
                         min_avg_conf_across_frames = min(min_avg_conf_across_frames, frame_data[frame_idx]['results'][roi_id]['confidence'])
                 if min_avg_conf_across_frames == 1.0: # Handle case where ROI wasn't found in any frame
                     min_avg_conf_across_frames = 0.0

                 # Get the confidence from the winning frame
                 conf_winning_frame = result_data['confidence']

                 # --- Print with correct confidence values and labels ---
                 print(f"    > ROI ID {roi_id}: Text = '{result_data['text']}' (Raw: '{result_data['raw_text']}') (Confidence: {conf_winning_frame:.6f}) (MinConfidence: {min_avg_conf_across_frames:.6f})")
                 # --- Write correct confidence to CSV ---
                 row_data = [video_timestamp_str, actual_processed_frame_num, base_filename, roi_id, result_data['text'], result_data['raw_text'], f"{conf_winning_frame:.6f}"]
                 try: csv_writer.writerow(row_data)
                 except Exception as csv_e: print(f"Error writing to CSV file: {csv_e}")
        else:
             print("    No results to write for the selected frame.")


        # --- Update last processed time ---
        last_processed_time_sec = current_processed_time_sec

        # --- Display Best Sampled Frame (if enabled) ---
        if display_enabled and best_frame_object is not None:
            display_frame = best_frame_object.copy() # Use the best frame
            # Draw ROIs
            for i, roi_info in enumerate(rois):
                color_tuple, color_name = ROI_COLORS[i % len(ROI_COLORS)]
                try: points = np.array(roi_info['points'], dtype=np.int32).reshape((-1, 1, 2)); cv2.polylines(display_frame, [points], isClosed=True, color=color_tuple, thickness=2); cv2.putText(display_frame, str(roi_info['id']), tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_tuple, 2)
                except Exception as draw_e: print(f"W: Could not draw ROI ID {roi_info['id']}: {draw_e}")
            cv2.imshow(window_name, display_frame)


        # --- Handle Key Press (if display enabled) ---
        key_pressed = -1
        if display_enabled:
            key_pressed = cv2.waitKey(1) & 0xFF # Wait 1ms
        elif not paused: # If display disabled, add delay only when not paused
             time.sleep(0.001) # Prevent 100% CPU usage when headless

        # Process keys (q, s, l, n, d, c, o, v, SPACE)
        if key_pressed == ord('q'):
             print(f"\n'q' pressed, stopping processing for {base_filename}.")
             break # Exit loop for this video file
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
            if not paused: paused = True; print("Paused. Click 4 points.")
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
             if paused: show_warped = True; print("Will show enhanced/optimized patches now (press any key in patch window to close).")
             else: print("Press SPACE to pause first, then 'v' to view patches.")
        elif key_pressed == ord(' '): # Toggle Pause
            paused = not paused; status_msg = "Paused." if paused else "Resumed."; print(f"\n{status_msg}")
            if not paused and defining_roi: current_roi_points = []; defining_roi = False; print("Cancelling ROI definition.")
            if not paused: # Destroy patch windows on resume
                 for i, roi_info in enumerate(rois):
                     try: cv2.destroyWindow(f"ROI {roi_info['id']} Enhanced (Optimal: {roi_info.get('transform')})")
                     except: pass
                 show_warped = False
        # Removed redundant sleep when display disabled and paused


        # --- Loop Increment (Frame number is now set by seek) ---
        # No frame_number increment needed here when seeking


    # --- Cleanup for this video file ---
    cap.release()
    if display_enabled:
        try: cv2.destroyWindow(window_name) # Close the specific window for this video
        except Exception: pass # Ignore error if already closed
    print(f"Finished processing {base_filename}. Processed OCR on {processed_ocr_count} frames.")
    if processed_ocr_count > 0 and rois:
        # Calculate average time per frame where OCR was actually run
        avg_ocr_time_per_active_frame = total_ocr_time / processed_ocr_count if processed_ocr_count > 0 else 0
        print(f"  Average OCR processing time per sampled frame: {avg_ocr_time_per_active_frame:.4f} seconds")
    return True # Indicate success for this file


# --- Script Execution Entry Point ---
if __name__ == "__main__":
    # --- Argument Parsing (Using argparse) ---
    parser = argparse.ArgumentParser(description="Extract text from ROIs in video files using OCR.")
    parser.add_argument("input_paths", nargs='+', help="Path(s) to video file(s), directory, or glob pattern (e.g., 'videos/*.avi').")
    parser.add_argument("-o", "--output-csv", default="ocr_results.csv", help="Path for the output CSV file (defaults to 'ocr_results.csv').")
    parser.add_argument("-i", "--interval", type=float, default=1.0, dest='sample_interval_sec', help="Seconds between OCR attempts (defaults to 1.0).")
    parser.add_argument("--gpu-id", type=int, default=0, help="ID of the GPU to use (default: 0). Ignored if --force-cpu is used.")
    parser.add_argument("--no-display", action="store_true", help="Disable the GUI display window.")
    parser.add_argument("--force-cpu", action="store_true", help="Force PaddleOCR to use CPU even if GPU is available.")
    parser.add_argument("--threshold", action="store_true", dest="apply_thresholding", help="Apply Otsu's thresholding after contrast enhancement.") # New arg

    args = parser.parse_args()

    # Validate sample interval
    if args.sample_interval_sec <= 0:
        print("Warning: Sample interval must be positive. Using default 1.0 second.")
        args.sample_interval_sec = 1.0

    # Update global flags based on args
    ENABLE_DISPLAY = not args.no_display
    FORCE_CPU = args.force_cpu
    APPLY_THRESHOLDING = args.apply_thresholding # Set global flag

    # --- Find Video Files ---
    video_files = []
    for path_pattern in args.input_paths:
        # Check if it's a directory
        if os.path.isdir(path_pattern):
            print(f"Input path '{path_pattern}' is a directory. Searching...")
            supported_extensions = ['*.avi', '*.mp4', '*.mov', '*.mkv', '*.wmv']
            for ext in supported_extensions:
                video_files.extend(glob.glob(os.path.join(path_pattern, ext), recursive=True))
        # Check if it's a glob pattern
        elif '*' in path_pattern or '?' in path_pattern:
            print(f"Input path '{path_pattern}' is a glob pattern. Searching...")
            recursive_glob = '**' in path_pattern
            video_files.extend(glob.glob(path_pattern, recursive=recursive_glob))
        # Check if it's a single file
        elif os.path.isfile(path_pattern):
            video_files.append(path_pattern)
        else:
            print(f"Warning: Input path '{path_pattern}' is not a valid file, directory, or glob pattern. Skipping.")

    # Remove duplicates and sort
    video_files = sorted(list(set(video_files)))

    if not video_files: print(f"No valid video files found matching input. Exiting."); sys.exit(0)

    print(f"\n--- Configuration ---")
    print(f"Input Path(s)/Pattern(s): {', '.join(args.input_paths)}")
    print(f"Output CSV File:    {args.output_csv}")
    print(f"Sample Interval:    {args.sample_interval_sec} seconds")
    print(f"Display Enabled:    {ENABLE_DISPLAY}")
    print(f"Force CPU:          {FORCE_CPU}")
    if not FORCE_CPU:
        print(f"Target GPU ID:      {args.gpu_id}")
    print(f"Apply Thresholding: {APPLY_THRESHOLDING}") # Print new flag status
    print(f"Found {len(video_files)} video file(s) to process:")
    for f in video_files: print(f"  - {f}")
    print(f"--------------------\n")

    # --- Print GPU Info ---
    if not FORCE_CPU:
        try:
            if paddle.is_compiled_with_cuda():
                device_count = paddle.device.cuda.device_count()
                print(f"Found {device_count} CUDA-enabled GPU(s):")
                if device_count > 0:
                    for i in range(device_count):
                        print(f"  - GPU {i}: {paddle.device.cuda.get_device_name(i)}")
                    if args.gpu_id >= device_count:
                        print(f"Warning: Specified GPU ID {args.gpu_id} is out of range. Using GPU 0.")
                        args.gpu_id = 0
                else:
                    print("  No GPUs found by PaddlePaddle. Will attempt CPU.")
                    FORCE_CPU = True # Force CPU if no devices listed
            else:
                print("PaddlePaddle was not compiled with CUDA support. Using CPU.")
                FORCE_CPU = True
        except Exception as e:
            print(f"Error getting GPU info: {e}. Using CPU.")
            FORCE_CPU = True


    # --- Initialize PaddleOCR (once) ---
    print("\nInitializing PaddleOCR...")
    try:
        # Set GPU ID only if not forcing CPU
        gpu_id_to_use = args.gpu_id if not FORCE_CPU else 0 # Default to 0 if forced CPU
        ocr_engine_main = paddleocr.PaddleOCR(
            use_angle_cls=True, lang='en', use_gpu=(not FORCE_CPU), gpu_id=gpu_id_to_use, show_log=False,
            det_db_unclip_ratio=PADDLE_UNCLIP_RATIO,
            det_db_thresh=PADDLE_DB_THRESH,
            det_db_box_thresh=PADDLE_BOX_THRESH
        )
        print("PaddleOCR initialized.");
        print(f"  Using Params: UnclipRatio={PADDLE_UNCLIP_RATIO}, DBThresh={PADDLE_DB_THRESH}, BoxThresh={PADDLE_BOX_THRESH}")
        if FORCE_CPU: print("PaddleOCR running on CPU (forced).")
        else: print(f"PaddleOCR attempting GPU {gpu_id_to_use}.")
    except Exception as e: print(f"Error initializing PaddleOCR: {e}"); sys.exit(1)

    # --- Load ROIs (once) ---
    rois_loaded = load_rois(ROI_CONFIG_FILE)
    if not rois_loaded and ENABLE_DISPLAY: print("No ROI config found. Define ROIs for the first video.")
    elif not rois_loaded and not ENABLE_DISPLAY: print("Error: No ROI config found and display is disabled. Exiting."); sys.exit(1)

    # --- Setup CSV File (once) ---
    # --- FIX: Add Raw Text and Confidence to headers ---
    csv_headers = ['Video Time', 'Frame', 'Filename', 'ROI ID', 'Recognized Text', 'Raw Text', 'Confidence']
    file_exists = os.path.exists(args.output_csv)
    csvfile = None; csv_writer = None
    try:
        # Ensure the directory for the output file exists
        output_dir = os.path.dirname(args.output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        csvfile = open(args.output_csv, 'a', newline='', encoding='utf-8')
        csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        if not file_exists or os.path.getsize(args.output_csv) == 0:
            csv_writer.writerow(csv_headers)
            print(f"\nWriting new CSV file: {args.output_csv}")
        else: print(f"\nAppending to existing CSV file: {args.output_csv}")
    except IOError as e: print(f"Error opening CSV file {args.output_csv}: {e}"); sys.exit(1)

    # --- Process Each Video File ---
    overall_success = True
    try:
        for index, video_file_path in enumerate(video_files):
            print(f"\n--- Processing Video {index + 1}/{len(video_files)}: {video_file_path} ---")
            paused = False # Ensure each video starts unpaused
            success = process_video_file(
                video_file_path,
                csv_writer,
                ocr_engine_main,
                sample_interval_sec=args.sample_interval_sec, # Pass parsed time interval
                display_enabled=ENABLE_DISPLAY
            )
            if not success:
                overall_success = False
                # break # Optional: Stop processing further files on error

    except Exception as e: # Catch broader exceptions here
        print(f"\nAn critical error occurred during processing loop: {e}")
        # ADD TRACEBACK HERE
        print("\n----- Traceback -----")
        traceback.print_exc()
        print("---------------------\n")
        overall_success = False
    finally:
        # --- Final Cleanup ---
        if csvfile and not csvfile.closed: csvfile.close(); print("CSV file closed.")
        if ENABLE_DISPLAY:
            try: cv2.destroyAllWindows()
            except Exception: pass

    print("\n===== Processing Complete =====")
    if not overall_success: print("Some files may not have been processed successfully."); sys.exit(1)
    else: print("All files processed successfully."); sys.exit(0)

