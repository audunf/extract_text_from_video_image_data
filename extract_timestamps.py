# Author: Audun Føyen <audun@audunfoyen.com>
# Copyright (c) 2025 Audun Føyen
# License: MIT


# Import necessary libraries
import cv2
import paddleocr
import sys
import os
import csv
import re
import pandas as pd # Keep pandas import for potential future use, though not used now
from datetime import timedelta
import time
import numpy as np
import json # For loading/saving ROIs
import math


# --- Configuration ---
ROI_CONFIG_FILE = 'roi_config.json'
# Define a TARGET width for the straightened ROI patches. Height will be calculated.
ROI_TARGET_WIDTH = 300 # Adjust as needed

# --- >>> GUI Toggle <<< ---
ENABLE_DISPLAY = True
# --- >>> CPU/GPU Toggle <<< ---
FORCE_CPU = False
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
# --- >>> PaddleOCR Tuning Parameters <<< ---
PADDLE_UNCLIP_RATIO = 2.0
PADDLE_BOX_THRESH = 0.6
PADDLE_DB_THRESH = 0.3
# --- >>> Optimization Parameters <<< ---
NORMAL_ORIENTATION_CONF_THRESHOLD = 0.85


# --- Global Variables for Interaction ---
rois = [] # List of ROI dicts
next_roi_id = 0
current_roi_points = []
defining_roi = False
paused = False
show_warped = False # Flag to control warped ROI display

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

def calculate_avg_confidence(ocr_result):
    """Calculates average confidence from PaddleOCR result."""
    total_confidence = 0; count = 0
    if ocr_result and ocr_result[0] is not None:
        for line in ocr_result[0]:
            if line and len(line) == 2:
                text_info = line[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) > 1:
                    confidence = text_info[1]
                    if isinstance(confidence, (float, int)): total_confidence += confidence; count += 1
    return total_confidence / count if count > 0 else 0.0

def estimate_roi_dims(roi_points):
    """Estimates the width and height of the quadrilateral defined by roi_points."""
    try:
        pts = np.array(roi_points, dtype=np.float32)
        dist_01 = np.linalg.norm(pts[0] - pts[1]); dist_12 = np.linalg.norm(pts[1] - pts[2])
        dist_23 = np.linalg.norm(pts[2] - pts[3]); dist_30 = np.linalg.norm(pts[3] - pts[0])
        est_width = (dist_01 + dist_23) / 2.0; est_height = (dist_12 + dist_30) / 2.0
        if est_width < 1 or est_height < 1: return ROI_TARGET_WIDTH, 50 # Default aspect ratio
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
    """Applies preprocessing (grayscale, contrast enhancement) to the ROI patch."""
    if patch is None: return None
    enhanced_patch = patch
    if APPLY_GRAYSCALE:
        if len(enhanced_patch.shape) == 3 and enhanced_patch.shape[2] == 3:
            enhanced_patch = cv2.cvtColor(enhanced_patch, cv2.COLOR_BGR2GRAY)
    if ENHANCE_CONTRAST and len(enhanced_patch.shape) == 2: # CLAHE needs grayscale
        try:
            clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
            enhanced_patch = clahe.apply(enhanced_patch)
        except Exception as e:
            print(f"Warning: Failed to apply CLAHE enhancement: {e}")
            if len(patch.shape) == 3 and patch.shape[2] == 3: enhanced_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            elif len(patch.shape) == 2: enhanced_patch = patch
            else: enhanced_patch = patch
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
        if est_w < 1: est_w = 1
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
                standard_confidence = calculate_avg_confidence(ocr_result_standard)
                print(f"  Confidence for Standard Orientation (Rot 0, No Flip): {standard_confidence:.4f}")
                if standard_confidence >= NORMAL_ORIENTATION_CONF_THRESHOLD:
                    print(f"  Prioritizing standard orientation."); best_transform = standard_transform.copy(); best_score = standard_confidence
                    rois[roi_index]['transform'] = best_transform; print(f"--- Opt Complete ROI #{roi_id} ---"); print(f"  Best: {best_transform}, Score: {best_score:.4f}"); return

        # If standard not good enough, test flips for target rotation
        print(f"  Standard conf below threshold. Testing flips for target rot {target_rotation}...")
        transform_params_target_noflip = {'rotate': target_rotation, 'flip_code': None}
        transformed_patch_target_noflip = apply_roi_transform(initial_warped_patch, transform_params_target_noflip)
        best_score = 0.0 # Reset best score
        if transformed_patch_target_noflip is not None:
             enhanced_patch_target_noflip = enhance_roi_patch(transformed_patch_target_noflip)
             if enhanced_patch_target_noflip is not None:
                  ocr_result_target_noflip = ocr_engine.ocr(enhanced_patch_target_noflip, cls=True)
                  best_score = calculate_avg_confidence(ocr_result_target_noflip)
                  best_transform = transform_params_target_noflip.copy()

        # Test other flips for the target rotation
        for flip_code_test in [1, 0, -1]: # H, V, Both
            transform_params = {'rotate': target_rotation, 'flip_code': flip_code_test}
            transformed_patch = apply_roi_transform(initial_warped_patch, transform_params)
            if transformed_patch is None: continue
            enhanced_patch_for_ocr = enhance_roi_patch(transformed_patch)
            if enhanced_patch_for_ocr is None: continue
            ocr_result = ocr_engine.ocr(enhanced_patch_for_ocr, cls=True)
            avg_confidence = calculate_avg_confidence(ocr_result)
            if avg_confidence > best_score: best_score = avg_confidence; best_transform = transform_params.copy()

        rois[roi_index]['transform'] = best_transform
        print(f"--- Optimization Complete for ROI #{roi_id} ---")
        print(f"  Best Transform (TargetRot {target_rotation}): {best_transform}, Score: {best_score:.4f}")
    except Exception as e:
        print(f"Error during optimization for ROI #{roi_id}: {e}")
        rois[roi_index]['transform'] = {'rotate': 0, 'flip_code': None}


def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks for defining ROI points."""
    global current_roi_points, defining_roi, rois, next_roi_id, frame_for_opt
    if not defining_roi: return
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < 0: x = 0
        if y < 0: y = 0
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


# --- Main Processing Logic ---
frame_for_opt = None # Global to hold frame for optimization callback

def process_video(video_path, output_csv_path, sample_interval_sec=1.0, display_enabled=True):
    """
    Main function modified for time-based sampling.
    """
    global rois, current_roi_points, defining_roi, paused, next_roi_id, frame_for_opt, show_warped

    # --- Initialize PaddleOCR ---
    print("Initializing PaddleOCR (attempting GPU)...")
    try:
        ocr_engine = paddleocr.PaddleOCR(
            use_angle_cls=True, lang='en', use_gpu=(not FORCE_CPU), show_log=False,
            det_db_unclip_ratio=PADDLE_UNCLIP_RATIO,
            det_db_thresh=PADDLE_DB_THRESH,
            det_db_box_thresh=PADDLE_BOX_THRESH
        )
        print("PaddleOCR initialized.");
        print(f"  Using Params: UnclipRatio={PADDLE_UNCLIP_RATIO}, DBThresh={PADDLE_DB_THRESH}, BoxThresh={PADDLE_BOX_THRESH}")
        if FORCE_CPU: print("PaddleOCR running on CPU (forced).")
        else: print("PaddleOCR attempting GPU.")
    except Exception as e: print(f"Error initializing PaddleOCR: {e}"); raise ImportError("Failed PaddleOCR init.") from e

    # --- Video Loading ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"Error: Could not open video file '{video_path}'.")

    # --- Get Video Properties & Read First Frame ---
    fps = cap.get(cv2.CAP_PROP_FPS); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_duration_sec = (total_frames / fps) if fps and fps > 0 else 0

    # --- FPS Check for Time-Based Sampling ---
    if fps <= 0:
        print(f"Error: Video FPS is {fps}. Cannot use time-based sampling.")
        cap.release()
        return # Exit if FPS is invalid

    print(f"Video loaded: '{os.path.basename(video_path)}'")
    print(f"Resolution: {frame_width}x{frame_height}, Total Frames: {total_frames}, FPS: {fps:.2f}")
    if total_duration_sec > 0: print(f"Estimated Duration: {timedelta(seconds=total_duration_sec)}")
    else: print("Could not determine video duration.")
    print(f"OCR Sampling Interval: {sample_interval_sec} seconds")


    ret, first_frame = cap.read()
    if not ret: print("Error: Could not read the first frame."); cap.release(); return
    frame_for_opt = first_frame

    # --- Load Initial ROIs & Optimize Them ---
    rois_loaded = load_rois(ROI_CONFIG_FILE)
    if rois_loaded and rois:
        print("Optimizing transforms for loaded ROIs...")
        for i in range(len(rois)): optimize_roi_transform(i, first_frame, ocr_engine)
        print("Finished optimizing loaded ROIs.")

    # --- Setup OpenCV Window and Mouse Callback (if display enabled) ---
    window_name = "ROI Definition & OCR Output"
    if display_enabled:
        cv2.namedWindow(window_name)
        frame_ref = [first_frame]
        cv2.setMouseCallback(window_name, mouse_callback, param=(ocr_engine, frame_ref))
        # Print controls
        print("\n--- Controls ---")
        print(" Mouse Click: Define ROI corners (after pressing 'n')")
        print("  n: Start NEW ROI | d: DELETE last | c: CLEAR all")
        print("  s: SAVE ROIs | l: LOAD ROIs | o: Re-OPTIMIZE all ROIs")
        print("  v: Toggle VIEW warped/optimized patches (when PAUSED)")
        print("  SPACE: PAUSE / RESUME | q: QUIT")
        print("----------------")
    else: print("Display window disabled.")

    # --- Setup CSV File for Incremental Writing ---
    csv_headers = ['Video Time', 'Frame', 'ROI ID', 'Recognized Text']
    file_exists = os.path.exists(output_csv_path)
    try:
        csvfile = open(output_csv_path, 'a', newline='', encoding='utf-8')
        csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        if not file_exists or os.path.getsize(output_csv_path) == 0:
            csv_writer.writerow(csv_headers)
            print(f"Writing new CSV file: {output_csv_path}")
        else: print(f"Appending to existing CSV file: {output_csv_path}")
    except IOError as e: print(f"Error opening CSV file {output_csv_path}: {e}"); cap.release(); return

    # --- Processing Loop ---
    frame_number = 0; processed_frame_count = 0; total_ocr_time = 0
    current_frame = first_frame
    last_processed_time_sec = -sample_interval_sec # Ensure first frame is processed
    # Removed redundant roi_dst_points definition here

    try: # Wrap main loop for finally block to close CSV
        while True:
            # --- Read Frame ---
            if not paused:
                ret, frame = cap.read();
                if not ret: break
                current_frame = frame
                if display_enabled: frame_ref[0] = current_frame
            if current_frame is None: break # Should not happen

            # --- Prepare Display Frame ---
            display_frame = current_frame.copy()
            current_video_time_sec = frame_number / fps

            # --- Determine if OCR should run this iteration ---
            should_process_ocr = not paused and (current_video_time_sec - last_processed_time_sec >= sample_interval_sec)

            if should_process_ocr:
                processed_frame_count += 1 # Count OCR processing attempts
                last_processed_time_sec = current_video_time_sec # Update time of last processing
                video_timestamp_str = format_timedelta(timedelta(seconds=current_video_time_sec))
                progress_percent = (frame_number / total_frames * 100) if total_frames > 0 else 0

                # --- Print Progress ---
                print_interval = 10 # Print every N processed samples
                if processed_frame_count == 1 or processed_frame_count % print_interval == 0:
                    print(f"\n--- Processing Frame {frame_number} / {total_frames} ({progress_percent:.1f}%) | Video Time: {video_timestamp_str} ---")
                    print(f"  (Processing at ~{sample_interval_sec} sec interval)")

                # --- Process Defined ROIs ---
                ocr_start_time = time.time()
                if rois: # Check if ROIs are defined
                    if processed_frame_count == 1 or processed_frame_count % print_interval == 0:
                        print(f"  Processing {len(rois)} ROIs...")

                    for i, roi_info in enumerate(rois):
                        roi_id = roi_info['id']; roi_points = roi_info['points']
                        roi_transform = roi_info.get('transform', {'rotate': 0, 'flip_code': None})
                        recognized_text_raw = "ERROR"; roi_text_fragments = []
                        try:
                            if not (isinstance(roi_points, (list, np.ndarray)) and len(roi_points) == 4): continue
                            roi_src_points = np.float32(roi_points)

                            # 1. Estimate aspect ratio and define destination points
                            est_w, est_h = estimate_roi_dims(roi_points)
                            if est_w < 1: est_w = 1
                            target_h = max(1, int(ROI_TARGET_WIDTH * (est_h / est_w)))
                            # Define destination points for warp *dynamically*
                            current_dst_points = np.float32([[0, 0], [ROI_TARGET_WIDTH - 1, 0], [ROI_TARGET_WIDTH - 1, target_h - 1], [0, target_h - 1]])

                            # 2. Warp ROI
                            matrix = cv2.getPerspectiveTransform(roi_src_points, current_dst_points)
                            warped_roi = cv2.warpPerspective(current_frame, matrix, (ROI_TARGET_WIDTH, target_h))

                            # 3. Apply optimal rotation/flip
                            optimally_transformed_patch = apply_roi_transform(warped_roi, roi_transform)
                            if optimally_transformed_patch is None: raise ValueError("Apply transform failed")

                            # 4. Apply Enhancement
                            enhanced_patch_for_ocr = enhance_roi_patch(optimally_transformed_patch)
                            if enhanced_patch_for_ocr is None: raise ValueError("Enhancement failed")

                            # 5. Perform OCR
                            ocr_result = ocr_engine.ocr(enhanced_patch_for_ocr, cls=True)

                            # 6. Extract & Post-process Text
                            if ocr_result and ocr_result[0] is not None:
                                for line in ocr_result[0]:
                                    if line and len(line) == 2:
                                        text_info = line[1]
                                        if isinstance(text_info, (list, tuple)) and len(text_info) > 0:
                                            roi_text_fragments.append(text_info[0])
                            recognized_text_raw = " ".join(roi_text_fragments).strip()
                            recognized_text_processed = post_process_text(recognized_text_raw)


                            if recognized_text_processed:
                                 if not display_enabled or processed_frame_count % print_interval == 0 or processed_frame_count == 1:
                                     print(f"    > ROI ID {roi_id}: Text = '{recognized_text_processed}' (Raw: '{recognized_text_raw}') (Transform: {roi_transform})")

                        except Exception as e:
                            print(f"\nError processing ROI ID {roi_id}: {e}");
                            recognized_text_processed = f'Error: {e}' # Store error

                        # --- Write result row to CSV ---
                        row_data = [video_timestamp_str, frame_number, roi_id, recognized_text_processed]
                        try:
                            csv_writer.writerow(row_data)
                        except Exception as csv_e:
                            print(f"Error writing to CSV file: {csv_e}")

                total_ocr_time += (time.time() - ocr_start_time) # Add time only when OCR runs

            # --- Display Frame and ROIs (Always update display if enabled) ---
            if display_enabled:
                # Draw ROIs
                for i, roi_info in enumerate(rois):
                    color_tuple, color_name = ROI_COLORS[i % len(ROI_COLORS)]
                    try: points = np.array(roi_info['points'], dtype=np.int32).reshape((-1, 1, 2)); cv2.polylines(display_frame, [points], isClosed=True, color=color_tuple, thickness=2); cv2.putText(display_frame, str(roi_info['id']), tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_tuple, 2)
                    except Exception as draw_e: print(f"W: Could not draw ROI ID {roi_info['id']}: {draw_e}")
                # Draw points being defined
                if defining_roi:
                    for idx, pt in enumerate(current_roi_points): cv2.circle(display_frame, tuple(pt), 5, (0, 255, 255), -1); cv2.putText(display_frame, str(idx+1), (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                    prompt_y_pos = 30; prompt_text = "";
                    if len(current_roi_points) < 4: prompt_text = f"Click point {len(current_roi_points)+1}/4 for ROI ID {next_roi_id}"
                    if prompt_text: cv2.putText(display_frame, prompt_text, (10, prompt_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # Draw PAUSED text
                if paused: cv2.putText(display_frame, "PAUSED", (frame_width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if show_warped and paused: cv2.putText(display_frame, "Showing Warped Patches ('v' pressed)", (frame_width - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Show main window
                cv2.imshow(window_name, display_frame)

                # Show warped patches if requested ('v' key while paused)
                if paused and show_warped and rois:
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


            # --- Handle Key Press (if display enabled) ---
            key_pressed = -1
            if display_enabled: key_pressed = cv2.waitKey(1) & 0xFF

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
            elif not display_enabled and not paused: time.sleep(0.001) # Delay if headless


            # Increment frame number only if not paused
            if not paused:
                frame_number += 1

    # --- Cleanup ---
    finally: # Ensure CSV file is closed even if errors occur
        if 'csvfile' in locals() and csvfile and not csvfile.closed:
            csvfile.close()
            print("CSV file closed.")
        cap.release()
        if display_enabled:
            try: cv2.destroyAllWindows()
            except Exception as e: print(f"Warning: Error destroying OpenCV windows: {e}")

    # --- Final Summary ---
    print(f"\nVideo processing finished or stopped. Processed OCR on {processed_frame_count} frames.")
    if processed_frame_count > 0 and rois:
      # Calculate average time per frame where OCR was actually run
      avg_ocr_time_per_active_frame = total_ocr_time / processed_frame_count if processed_frame_count > 0 else 0
      print(f"Average OCR processing time per sampled frame: {avg_ocr_time_per_active_frame:.4f} seconds")

    # --- Final Save Results (Removed - now saving incrementally) ---
    print(f"Output data saved incrementally to '{output_csv_path}'")


# --- Script Execution Entry Point ---
if __name__ == "__main__":
    # Argument parsing - updated for sample_interval_sec
    if len(sys.argv) < 2:
        print("\nUsage: python your_script_name.py <path_to_video.avi> [output_csv_file] [sample_interval_sec]")
        print("\nArguments:")
        print("  <path_to_video.avi>: Path to input video (required).")
        print("  [output_csv_file]:   Optional. Path for output CSV. Defaults to 'roi_ocr_results.csv'.")
        print("  [sample_interval_sec]: Optional. Seconds between OCR attempts. Defaults to 1.0.")
        print("\nRequires: paddleocr, paddlepaddle-gpu (or paddlepaddle), opencv-python, pandas, numpy")
        print("\nExample:")
        print("  python your_script_name.py movie.avi roi_results.csv 0.5") # Sample every 0.5 seconds
        sys.exit(1)

    video_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "roi_ocr_results.csv"
    interval_sec = 1.0 # Default sample interval
    if len(sys.argv) > 3:
        try:
            interval_sec = float(sys.argv[3])
            if interval_sec <= 0:
                 print("Warning: Sample interval must be positive. Using default 1.0 second.")
                 interval_sec = 1.0
        except ValueError:
            print("Error: Sample interval must be a number. Using default 1.0 second.")
            interval_sec = 1.0

    # Run the main processing function
    try:
        process_video(video_file, output_file, sample_interval_sec=interval_sec, display_enabled=ENABLE_DISPLAY)
    except (FileNotFoundError, IOError, ImportError, Exception) as e:
        print(f"\nAn critical error occurred: {e}")
        # Attempt cleanup even on critical error
        if 'cap' in locals() and 'cap' in globals() and cap.isOpened(): cap.release()
        if ENABLE_DISPLAY:
             try: cv2.destroyAllWindows()
             except: pass
        if 'csvfile' in locals() and 'csvfile' in globals() and csvfile and not csvfile.closed: csvfile.close()
        sys.exit(1)

