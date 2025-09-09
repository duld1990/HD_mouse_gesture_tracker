# Merged Open-Field and Stand/Lean/Head Gesture Analysis Script
# Reads all coordinates from a single CSV file and skips NaN rows
# Author: Lida Du (ldu13@jh.edu)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from scipy.ndimage import gaussian_filter

# Add XGBoost import
try:
    import xgboost as xgb
    import joblib
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost joblib")

def remove_tracking_outliers(df, max_jump_distance=50.0, window_size=5):
    """
    Remove tracking outliers where points jump to faraway locations.
    """
    df_clean = df.copy()
    
    # Columns to check for outliers
    coord_columns = [col for col in df.columns if col.endswith(('_x', '_y'))]
    
    for col in coord_columns:
        if col in df.columns:
            coords = df[col].values
            
            # Calculate distances between consecutive frames
            distances = np.abs(np.diff(coords))
            
            # Find jumps that exceed threshold
            outlier_mask = np.zeros_like(coords, dtype=bool)
            outlier_mask[1:] = distances > max_jump_distance
            
            # Check for isolated valid points surrounded by NaN
            for i in range(1, len(coords) - 1):
                if not np.isnan(coords[i]) and (np.isnan(coords[i-1]) or np.isnan(coords[i+1])):
                    # Check if this point is far from its neighbors
                    if not np.isnan(coords[i-1]) and abs(coords[i] - coords[i-1]) > max_jump_distance:
                        outlier_mask[i] = True
                    if not np.isnan(coords[i+1]) and abs(coords[i] - coords[i+1]) > max_jump_distance:
                        outlier_mask[i] = True
            
            # Additional check: look for points that are far from the median of surrounding frames
            for i in range(window_size, len(coords) - window_size):
                if not np.isnan(coords[i]):
                    # Get surrounding valid points
                    surrounding = coords[i-window_size:i+window_size+1]
                    surrounding = surrounding[~np.isnan(surrounding)]
                    if len(surrounding) > 2:  # Need at least 3 points for meaningful median
                        median_val = np.median(surrounding)
                        if abs(coords[i] - median_val) > max_jump_distance * 1.5:
                            outlier_mask[i] = True
            
            # Replace outliers with NaN
            df_clean.loc[outlier_mask, col] = np.nan
    
    return df_clean

def load_and_clean_csv(csv_path, max_jump_distance=50.0):
    """Load CSV and clean tracking outliers."""
    df_full = pd.read_csv(csv_path)
    
    # Remove tracking outliers
    df_full = remove_tracking_outliers(df_full, max_jump_distance)
    
    # Only require body_center_x and body_center_y for main analyses
    required_main = ['body_center_x', 'body_center_y']
    valid_mask = df_full[required_main].notna().all(axis=1)
    df_valid = df_full[valid_mask].copy()
    
    return df_full, df_valid, valid_mask

def signed_angle(v1, v2):
    """Calculate signed angle between two vectors."""
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    angle = np.degrees(angle2 - angle1)
    return (angle + 180) % 360 - 180

def validate_csv_structure(df):
    """
    Validate CSV structure and handle common issues.
    """
    print("Validating CSV structure...")
    
    # Check required columns
    required_columns = ['body_center_x', 'body_center_y', 'tail_base_x', 'tail_base_y', 
                       'forelimbs_x', 'forelimbs_y', 'head_x', 'head_y']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"Missing required columns: {missing_columns}"
        print(f"Error: {error_msg}")
        return False, error_msg, None
    
    # Check for empty DataFrame
    if len(df) == 0:
        error_msg = "CSV file is empty"
        print(f"Error: {error_msg}")
        return False, error_msg, None
    
    # Check data types and convert to numeric
    cleaned_df = df.copy()
    for col in required_columns:
        if col in cleaned_df.columns:
            # Convert to numeric, errors='coerce' will convert non-numeric to NaN
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            
            # Count NaN values
            nan_count = cleaned_df[col].isna().sum()
            if nan_count > 0:
                print(f"Column {col}: {nan_count} NaN values ({nan_count/len(df)*100:.1f}%)")
    
    # Check if we have enough valid data
    total_rows = len(cleaned_df)
    valid_rows = cleaned_df[required_columns].notna().all(axis=1).sum()
    
    print(f"Total rows: {total_rows}")
    print(f"Valid rows: {valid_rows} ({valid_rows/total_rows*100:.1f}%)")
    
    if valid_rows == 0:
        error_msg = "No valid rows found in CSV"
        print(f"Error: {error_msg}")
        return False, error_msg, None
    
    if valid_rows < total_rows * 0.1:  # Less than 10% valid data
        print(f"Warning: Only {valid_rows/total_rows*100:.1f}% of rows have valid data")
    
    print("CSV validation completed successfully")
    return True, "", cleaned_df

def calculate_xgboost_features(df_valid, sampling_rate):
    """
    Calculate features required by XGBoost model from raw coordinate data.
    """
    print("Calculating XGBoost features...")
    
    # Create a copy of the data
    df_features = df_valid.copy()
    total_frames = len(df_features)
    
    print(f"Processing {total_frames} frames...")
    
    # Handle NaN values first - replace with reasonable defaults or interpolate
    print("Handling NaN values...")
    
    # Fill NaN coordinates with reasonable defaults based on available data
    coord_columns = ['body_center_x', 'body_center_y', 'tail_base_x', 'tail_base_y', 
                     'forelimbs_x', 'forelimbs_y', 'head_x', 'head_y']
    
    for col in coord_columns:
        if col in df_features.columns:
            # Get non-NaN values for this column
            valid_values = df_features[col].dropna()
            if len(valid_values) > 0:
                # Use median of valid values as default
                default_value = valid_values.median()
                # Fill NaN with default value
                df_features[col].fillna(default_value, inplace=True)
                print(f"Filled NaN in {col} with default value: {default_value:.2f}")
            else:
                # If no valid values, use 0 as fallback
                df_features[col].fillna(0, inplace=True)
                print(f"Filled NaN in {col} with fallback value: 0")
    
    # Calculate body speed
    print("Calculating body speed...")
    body_dx = np.diff(df_features['body_center_x'].values)
    body_dy = np.diff(df_features['body_center_y'].values)
    
    # Handle any remaining NaN values in differences
    body_dx = np.nan_to_num(body_dx, nan=0.0)
    body_dy = np.nan_to_num(body_dy, nan=0.0)
    
    body_speed = np.sqrt(body_dx**2 + body_dy**2) * sampling_rate
    body_speed = np.insert(body_speed, 0, 0)
    df_features['body_speed'] = body_speed
    
    # Calculate body acceleration
    print("Calculating body acceleration...")
    body_accel = np.diff(body_speed)
    body_accel = np.nan_to_num(body_accel, nan=0.0)
    body_accel = np.insert(body_accel, 0, 0)
    df_features['body_accel'] = body_accel
    
    # Calculate angular speed
    print("Calculating angular speed...")
    
    # Ensure we have valid coordinates for angle calculation
    body_center_x = df_features['body_center_x'].values
    body_center_y = df_features['body_center_y'].values
    tail_base_x = df_features['tail_base_x'].values
    tail_base_y = df_features['tail_base_y'].values
    
    # Calculate body axis angle with NaN handling
    body_axis_angle = np.full(total_frames, 0.0)  # Default to 0
    
    for i in range(total_frames):
        if (not np.isnan(body_center_x[i]) and not np.isnan(body_center_y[i]) and
            not np.isnan(tail_base_x[i]) and not np.isnan(tail_base_y[i])):
            # Calculate angle only if all coordinates are valid
            body_axis_angle[i] = np.arctan2(
                body_center_y[i] - tail_base_y[i],
                body_center_x[i] - tail_base_x[i]
            )
        else:
            # Use previous valid angle or 0
            if i > 0:
                body_axis_angle[i] = body_axis_angle[i-1]
            else:
                body_axis_angle[i] = 0.0
    
    body_axis_angle_deg = np.degrees(body_axis_angle)
    
    # Calculate angular speed with NaN handling
    angular_speed = np.abs(np.diff(body_axis_angle_deg))
    angular_speed = np.minimum(angular_speed, 360 - angular_speed)
    angular_speed = angular_speed * sampling_rate
    angular_speed = np.nan_to_num(angular_speed, nan=0.0)
    angular_speed = np.insert(angular_speed, 0, 0)
    df_features['angular_speed'] = angular_speed
    
    # Calculate head-body angle
    print("Calculating head-body angle...")
    head_x = df_features['head_x'].values
    head_y = df_features['head_y'].values
    forelimbs_x = df_features['forelimbs_x'].values
    forelimbs_y = df_features['forelimbs_y'].values
    
    head_body_angle = np.full(total_frames, 0.0)  # Default to 0
    
    for i in range(total_frames):
        # Check if we have all required coordinates for this frame
        coords_valid = (
            not np.isnan(body_center_x[i]) and not np.isnan(body_center_y[i]) and
            not np.isnan(tail_base_x[i]) and not np.isnan(tail_base_y[i]) and
            not np.isnan(head_x[i]) and not np.isnan(head_y[i]) and
            not np.isnan(forelimbs_x[i]) and not np.isnan(forelimbs_y[i])
        )
        
        if coords_valid:
            # Calculate vectors
            head_body_vec_x = head_x[i] - body_center_x[i]
            head_body_vec_y = head_y[i] - body_center_y[i]
            body_axis_vec_x = body_center_x[i] - tail_base_x[i]
            body_axis_vec_y = body_center_y[i] - tail_base_y[i]
            
            # Calculate angle
            try:
                angle = signed_angle(
                    (body_axis_vec_x, body_axis_vec_y),
                    (head_body_vec_x, head_body_vec_y)
                )
                head_body_angle[i] = angle
            except:
                # If angle calculation fails, use previous valid angle or 0
                if i > 0:
                    head_body_angle[i] = head_body_angle[i-1]
                else:
                    head_body_angle[i] = 0.0
        else:
            # Use previous valid angle or 0
            if i > 0:
                head_body_angle[i] = head_body_angle[i-1]
            else:
                head_body_angle[i] = 0.0
        
        if i % 1000 == 0:
            progress = (i / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({i}/{total_frames} frames)")
    
    df_features['head_body_angle'] = head_body_angle
    
    # Final NaN check and cleanup
    print("Final NaN cleanup...")
    for col in ['body_speed', 'body_accel', 'angular_speed', 'head_body_angle']:
        if col in df_features.columns:
            # Replace any remaining NaN with 0
            df_features[col].fillna(0, inplace=True)
            print(f"Cleaned {col}: {df_features[col].isna().sum()} NaN values remaining")
    
    print("Feature calculation complete!")
    print(f"Features added: body_speed, body_accel, angular_speed, head_body_angle")
    print(f"DataFrame shape: {df_features.shape}")
    
    return df_features

def classify_behaviors_hybrid(df_valid, valid_mask, sampling_rate, xgb_model=None, scaler=None, label_encoder=None, feature_names=None, output_dir=None, total_frames=None):
    """
    Hybrid behavior classification: Original algorithm first, then XGBoost replacement (except for 'other').
    
    Steps:
    1. Scan whole CSV with original algorithm
    2. Scan whole CSV with XGBoost (if available)
    3. Replace original predictions with XGBoost results, EXCEPT when XGBoost says 'other'
    """
    print("Starting hybrid behavior classification...")
    print(f"Debug: df_valid length: {len(df_valid)}, total_frames: {total_frames}")
    
    # Step 1: Run original algorithm on whole CSV
    print("Step 1: Running original algorithm classification...")
    original_results = classify_behaviors_original(df_valid, valid_mask, sampling_rate)
    
    if original_results is None:
        print("Original classification failed, cannot proceed with hybrid approach")
        return None
    
    # Initialize final results with original predictions
    final_predictions = original_results['original_prediction'].copy()
    final_confidence = original_results['original_confidence'].copy()
    
    # Step 2: Run XGBoost classification (if available)
    if (xgb_model is not None and scaler is not None and 
        label_encoder is not None and feature_names is not None and XGBOOST_AVAILABLE):
        
        print("Step 2: Running XGBoost classification...")
        
        # Calculate features for XGBoost
        df_with_features = calculate_xgboost_features(df_valid, sampling_rate)
        
        # Save enhanced features CSV if output directory is provided
        if output_dir:
            enhanced_csv_path = os.path.join(output_dir, 'enhanced_features_for_xgboost.csv')
            df_with_features.to_csv(enhanced_csv_path, index=False)
            print(f"Enhanced CSV with calculated features saved to: {enhanced_csv_path}")
        
        # Run XGBoost classification
        xgb_results = classify_behaviors_with_xgboost(
            df_with_features, xgb_model, scaler, label_encoder, feature_names, sampling_rate
        )
        
        if xgb_results is not None:
            print("Step 3: Applying hybrid replacement rules...")
            
            # Step 3: Replace original predictions with XGBoost results, EXCEPT when XGBoost says 'other'
            xgb_predictions = xgb_results['xgb_prediction'].values
            xgb_confidence = xgb_results['xgb_confidence'].values
            
            replacement_count = 0
            other_preserved_count = 0
            
            for i, (xgb_pred, xgb_conf) in enumerate(zip(xgb_predictions, xgb_confidence)):
                original_pred = final_predictions.iloc[i]
                
                # Replace original prediction with XGBoost result, EXCEPT when XGBoost says 'other'
                if xgb_pred != 'other':
                    final_predictions.iloc[i] = xgb_pred
                    final_confidence.iloc[i] = xgb_conf
                    replacement_count += 1
                else:
                    # Keep original prediction when XGBoost says 'other'
                    other_preserved_count += 1
            
            print(f"Hybrid classification completed:")
            print(f"  - Original predictions replaced: {replacement_count}")
            print(f"  - Original predictions preserved (XGBoost said 'other'): {other_preserved_count}")
            print(f"  - Total frames processed: {len(final_predictions)}")
            
            # Create comprehensive results DataFrame
            hybrid_results = pd.DataFrame({
                'frame_index': df_valid.index,
                'original_prediction': original_results['original_prediction'],
                'original_confidence': original_results['original_confidence'],
                'xgb_prediction': xgb_predictions,
                'xgb_confidence': xgb_confidence,
                'final_prediction': final_predictions,
                'final_confidence': final_confidence
            })
            
            # If total_frames is provided and different from df_valid length, create full-frame results
            if total_frames is not None and total_frames != len(df_valid):
                print(f"Mapping {len(df_valid)} valid frames to {total_frames} total frames...")
                print(f"Debug: total_frames type: {type(total_frames)}, value: {total_frames}")
                print(f"Debug: df_valid length type: {type(len(df_valid))}, value: {len(df_valid)}")
                
                # Create full-frame arrays with default values
                full_original_pred = np.full(total_frames, 'other')
                full_original_conf = np.full(total_frames, 0.0)
                full_xgb_pred = np.full(total_frames, 'none')
                full_xgb_conf = np.full(total_frames, 0.0)
                full_final_pred = np.full(total_frames, 'other')
                full_final_conf = np.full(total_frames, 0.0)
                
                # Map valid frame results to full frame range
                # Use sequential mapping since df_valid is already cleaned and indexed sequentially
                for i in range(len(df_valid)):
                    if i < total_frames:
                        full_original_pred[i] = original_results.iloc[i]['original_prediction']
                        full_original_conf[i] = original_results.iloc[i]['original_confidence']
                        full_xgb_pred[i] = xgb_predictions[i]
                        full_xgb_conf[i] = xgb_confidence[i]
                        full_final_pred[i] = final_predictions.iloc[i]
                        full_final_conf[i] = final_confidence.iloc[i]
                
                print(f"Frame mapping completed: {len(df_valid)} frames mapped to positions 0-{len(df_valid)-1}")
                
                # Create full-frame results DataFrame
                full_hybrid_results = pd.DataFrame({
                    'frame_index': range(total_frames),
                    'original_prediction': full_original_pred,
                    'original_confidence': full_original_conf,
                    'xgb_prediction': full_xgb_pred,
                    'xgb_confidence': full_xgb_conf,
                    'final_prediction': full_final_pred,
                    'final_confidence': full_final_conf
                })
                
                return full_hybrid_results, original_results, xgb_results
            
            return hybrid_results, original_results, xgb_results
            
        else:
            print("XGBoost classification failed, using only original results")
            # Create results with only original classification
            hybrid_results = pd.DataFrame({
                'frame_index': df_valid.index,
                'original_prediction': original_results['original_prediction'],
                'original_confidence': original_results['original_confidence'],
                'xgb_prediction': ['none'] * len(df_valid),
                'xgb_confidence': [0.0] * len(df_valid),
                'final_prediction': final_predictions,
                'final_confidence': final_confidence
            })
            
            return hybrid_results, original_results, None
    
    else:
        print("XGBoost model not available, using only original results")
        # Create results with only original classification
        hybrid_results = pd.DataFrame({
            'frame_index': df_valid.index,
            'original_prediction': original_results['original_prediction'],
            'original_confidence': original_results['original_confidence'],
            'xgb_prediction': ['none'] * len(df_valid),
            'xgb_confidence': [0.0] * len(df_valid),
            'final_prediction': final_predictions,
            'final_confidence': final_confidence
        })
        
        return hybrid_results, original_results, None

def classify_behaviors_with_xgboost(df_valid, xgb_model, scaler, label_encoder, feature_names, sampling_rate):
    """
    Classify mouse behaviors using trained XGBoost model with fallback to original logic.
    """
    if not XGBOOST_AVAILABLE or xgb_model is None:
        return None
    
    try:
        print("Preparing features for XGBoost classification...")
        
        # Check if all required features exist
        missing_features = [f for f in feature_names if f not in df_valid.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            print("Available columns:", list(df_valid.columns))
            return None
        
        # Prepare features for XGBoost
        features_df = df_valid[feature_names].copy()
        
        # Handle missing values in features
        print("Handling missing values in features...")
        for col in feature_names:
            if col in features_df.columns:
                # Count NaN values
                nan_count = features_df[col].isna().sum()
                if nan_count > 0:
                    print(f"Column {col}: {nan_count} NaN values found")
                    
                    # Try interpolation first for small gaps
                    features_df[col] = features_df[col].interpolate(method='linear', limit=5)
                    
                    # Fill remaining NaN with forward fill, then backward fill
                    features_df[col] = features_df[col].fillna(method='ffill').fillna(method='bfill')
                    
                    # If still NaN, fill with 0
                    remaining_nan = features_df[col].isna().sum()
                    if remaining_nan > 0:
                        features_df[col].fillna(0, inplace=True)
                        print(f"Filled remaining {remaining_nan} NaN values in {col} with 0")
        
        # Final check - ensure no NaN values remain
        total_nan = features_df.isna().sum().sum()
        if total_nan > 0:
            print(f"Warning: {total_nan} NaN values still present after cleanup")
            # Fill any remaining NaN with 0
            features_df.fillna(0, inplace=True)
        
        print("Features prepared successfully")
        print(f"Feature shape: {features_df.shape}")
        print(f"Feature columns: {list(features_df.columns)}")
        
        # Scale features
        print("Scaling features...")
        features_scaled = scaler.transform(features_df)
        
        # Make predictions
        print("Making XGBoost predictions...")
        predictions = xgb_model.predict(features_scaled)
        probabilities = xgb_model.predict_proba(features_scaled)
        
        # Decode predictions
        predicted_behaviors = label_encoder.inverse_transform(predictions)
        
        # Get confidence scores (max probability for each prediction)
        confidence_scores = np.max(probabilities, axis=1)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'frame_index': df_valid.index,
            'xgb_prediction': predicted_behaviors,
            'xgb_confidence': confidence_scores
        })
        
        # Add probability columns for each behavior class
        for i, behavior in enumerate(label_encoder.classes_):
            results_df[f'prob_{behavior}'] = probabilities[:, i]
        
        print(f"XGBoost classification completed successfully")
        print(f"Predictions shape: {results_df.shape}")
        
        return results_df
        
    except Exception as e:
        print(f"XGBoost classification error: {e}")
        import traceback
        traceback.print_exc()
        return None

def classify_behaviors_original(df_valid, valid_mask, sampling_rate):
    """
    Classify mouse behaviors using optimized original logic with better behavior prioritization.
    """
    total_frames = len(df_valid)
    
    # Initialize arrays for predictions and confidence scores
    original_predictions = np.full(total_frames, 'other')
    original_confidence = np.full(total_frames, 0.0)
    
    # Extract coordinate data
    nose_x = df_valid['head_x'].values
    nose_y = df_valid['head_y'].values
    body_x = df_valid['body_center_x'].values
    body_y = df_valid['body_center_y'].values
    back_x = df_valid['tail_base_x'].values
    back_y = df_valid['tail_base_y'].values
    forelimb_x = df_valid['forelimbs_x'].values
    forelimb_y = df_valid['forelimbs_y'].values
    
    # Head angle threshold (degrees)
    gesture_angle_threshold = 15
    max_valid_head_turn = 45
    
    # Speed threshold (cm/s)
    walk_speed_threshold = 3.0
    
    # Distance thresholds (cm) - tightened for more specific classification
    rearing_dist_thresh = 15.0  # Reduced from 20.0 for more specific rearing
    stand_dist_thresh = 20.0    # Reduced from 25.0 for more specific stand
    grooming_dist_thresh = 18.0 # New threshold for grooming behavior
    lean_dist_thresh = 22.0     # New threshold for lean behavior
    
    # Calculate features for original logic
    head_angle = np.full(total_frames, np.nan)
    speed = np.full(total_frames, np.nan)
    distance_to_forelimb = np.full(total_frames, np.nan)
    head_body_dist = np.full(total_frames, np.nan)
    forelimb_body_dist = np.full(total_frames, np.nan)
    
    # Calculate body speed
    body_dx = np.diff(body_x)
    body_dy = np.diff(body_y)
    body_speed = np.sqrt(body_dx**2 + body_dy**2) * sampling_rate
    body_speed = np.insert(body_speed, 0, np.nan)  # Insert 0 for first frame
    
    for i in range(total_frames):
        # Calculate head angle
        head_vec_x = nose_x[i] - forelimb_x[i]
        head_vec_y = nose_y[i] - forelimb_y[i]
        axis_x = body_x[i] - back_x[i]
        axis_y = body_y[i] - back_y[i]
        
        if not (np.isnan(axis_x) or np.isnan(axis_y) or np.isnan(head_vec_x) or np.isnan(head_vec_y)):
            angle = signed_angle((axis_x, axis_y), (head_vec_x, head_vec_y))
            
            # Check if angle is beyond 90 degrees and correct vector direction if needed
            if np.abs(angle) > 90:
                # Flip the head vector direction and recalculate
                head_vec_x_flipped = -head_vec_x
                head_vec_y_flipped = -head_vec_y
                angle_flipped = signed_angle((axis_x, axis_y), (head_vec_x_flipped, head_vec_y_flipped))
                # Use the smaller angle (closer to 0)
                if np.abs(angle_flipped) < np.abs(angle):
                    head_angle[i] = angle_flipped
                else:
                    head_angle[i] = angle
            else:
                head_angle[i] = angle
        
        # Set speed and distances
        speed[i] = body_speed[i]
        distance_to_forelimb[i] = np.sqrt((body_x[i] - forelimb_x[i])**2 + (body_y[i] - forelimb_y[i])**2)
        head_body_dist[i] = np.sqrt((nose_x[i] - body_x[i])**2 + (nose_y[i] - body_y[i])**2)
        forelimb_body_dist[i] = np.sqrt((forelimb_x[i] - body_x[i])**2 + (forelimb_y[i] - body_y[i])**2)
    
    # Apply optimized original logic thresholds with better prioritization
    
    # 1. Head turn detection (highest priority - clear gesture)
    head_turn_mask = (np.abs(head_angle) > gesture_angle_threshold) & (np.abs(head_angle) < max_valid_head_turn)
    
    # 2. Movement-based behaviors (medium priority)
    walking_mask = speed > walk_speed_threshold
    freezing_mask = (speed <= walk_speed_threshold) & (speed > 0.5)  # Add minimum speed threshold
    
    # 3. Stand behaviors with more specific criteria (lower priority)
    
    # Rearing: head and forelimb very close, indicating upright posture
    rearing_mask = (distance_to_forelimb < rearing_dist_thresh) & (head_body_dist < rearing_dist_thresh)
    
    # Grooming: head close to forelimbs, moderate body distances
    grooming_mask = (distance_to_forelimb < grooming_dist_thresh) & (head_body_dist < grooming_dist_thresh) & \
                   (forelimb_body_dist < grooming_dist_thresh) & \
                   ~rearing_mask  # Don't overlap with rearing
    
    # Lean: all points moderately close, but not as tight as rearing/grooming
    lean_mask = (distance_to_forelimb < lean_dist_thresh) & (head_body_dist < lean_dist_thresh) & \
                (forelimb_body_dist < lean_dist_thresh) & \
                ~rearing_mask & ~grooming_mask  # Don't overlap with other behaviors
    
    # General stand: only when all other criteria fail and points are reasonably close
    general_stand_mask = (distance_to_forelimb < stand_dist_thresh) & (head_body_dist < stand_dist_thresh) & \
                        (forelimb_body_dist < stand_dist_thresh) & \
                        ~rearing_mask & ~grooming_mask & ~lean_mask  # Don't overlap with specific behaviors
    
    # Apply predictions with proper hierarchy (order matters!)
    # Start with 'other' as default, then override with more specific behaviors
    
    # Apply head turn first (highest priority)
    original_predictions[head_turn_mask] = 'head_turn'
    
    # Apply movement-based behaviors (but not if head turn was detected)
    walk_applicable = walking_mask & ~head_turn_mask
    freeze_applicable = freezing_mask & ~head_turn_mask
    original_predictions[walk_applicable] = 'walk'
    original_predictions[freeze_applicable] = 'freeze'
    
    # Apply stand behaviors (lowest priority, only if no other behavior detected)
    no_other_behavior = ~(head_turn_mask | walk_applicable | freeze_applicable)
    
    # Apply specific stand behaviors in order of specificity
    rearing_applicable = rearing_mask & no_other_behavior
    grooming_applicable = grooming_mask & no_other_behavior & ~rearing_applicable
    lean_applicable = lean_mask & no_other_behavior & ~rearing_applicable & ~grooming_applicable
    general_stand_applicable = general_stand_mask & no_other_behavior & ~rearing_applicable & ~grooming_applicable & ~lean_applicable
    
    original_predictions[rearing_applicable] = 'stand_rearing'
    original_predictions[grooming_applicable] = 'stand_grooming'
    original_predictions[lean_applicable] = 'stand_lean'
    original_predictions[general_stand_applicable] = 'stand'
    
    # Calculate confidence scores for original logic with more nuanced scoring
    
    # For head turn, confidence is based on angle magnitude
    head_turn_confidence = np.full(total_frames, 0.0)
    head_turn_confidence[head_turn_mask] = np.minimum(np.abs(head_angle[head_turn_mask]) / max_valid_head_turn, 1.0)
    
    # For walking/freezing, confidence is based on speed deviation from threshold
    walk_freeze_confidence = np.full(total_frames, 0.0)
    walk_freeze_confidence[walking_mask] = np.minimum(speed[walking_mask] / (walk_speed_threshold * 2), 1.0)
    walk_freeze_confidence[freezing_mask] = np.minimum((walk_speed_threshold - speed[freezing_mask]) / walk_speed_threshold, 1.0)
    
    # For stand behaviors, confidence is based on how close the distances are
    stand_confidence = np.full(total_frames, 0.0)
    
    # Rearing confidence: based on how close head and forelimb are
    rearing_confidence = np.full(total_frames, 0.0)
    rearing_confidence[rearing_applicable] = np.maximum(0, 1.0 - (distance_to_forelimb[rearing_applicable] / rearing_dist_thresh))
    
    # Grooming confidence: based on average of all distances
    grooming_confidence = np.full(total_frames, 0.0)
    grooming_confidence[grooming_applicable] = np.maximum(0, 1.0 - (
        (distance_to_forelimb[grooming_applicable] + head_body_dist[grooming_applicable] + forelimb_body_dist[grooming_applicable]) / (3 * grooming_dist_thresh)
    ))
    
    # Lean confidence: based on average of all distances
    lean_confidence = np.full(total_frames, 0.0)
    lean_confidence[lean_applicable] = np.maximum(0, 1.0 - (
        (distance_to_forelimb[lean_applicable] + head_body_dist[lean_applicable] + forelimb_body_dist[lean_applicable]) / (3 * lean_dist_thresh)
    ))
    
    # General stand confidence: based on average of all distances
    general_stand_confidence = np.full(total_frames, 0.0)
    general_stand_confidence[general_stand_applicable] = np.maximum(0, 1.0 - (
        (distance_to_forelimb[general_stand_applicable] + head_body_dist[general_stand_applicable] + forelimb_body_dist[general_stand_applicable]) / (3 * stand_dist_thresh)
    ))
    
    # Combine all confidence scores
    original_confidence_combined = np.full(total_frames, 0.0)
    original_confidence_combined[head_turn_mask] = head_turn_confidence[head_turn_mask]
    original_confidence_combined[walk_applicable] = walk_freeze_confidence[walk_applicable]
    original_confidence_combined[freeze_applicable] = walk_freeze_confidence[freeze_applicable]
    original_confidence_combined[rearing_applicable] = rearing_confidence[rearing_applicable]
    original_confidence_combined[grooming_applicable] = grooming_confidence[grooming_applicable]
    original_confidence_combined[lean_applicable] = lean_confidence[lean_applicable]
    original_confidence_combined[general_stand_applicable] = general_stand_confidence[general_stand_applicable]
    
    # Create results DataFrame for original logic with proper index
    original_results_df = pd.DataFrame({
        'frame_index': df_valid.index,
        'original_prediction': original_predictions,
        'original_confidence': original_confidence_combined
    })
    
    return original_results_df

def create_behavior_timeline_plot(combined_results, original_results, sampling_rate, output_dir, use_xgboost=False):
    """
    Create a time-plot showing mouse behavior over time with different colors for each behavior class.
    """
    print("Creating behavior timeline plot...")
    
    # Create time axis (seconds)
    total_frames = len(combined_results)
    time_axis = np.arange(total_frames) / sampling_rate
    
    # Define behavior colors
    behavior_colors = {
        'freeze': '#FF6B6B',      # Red
        'walk': '#4ECDC4',        # Teal
        'rotate': '#45B7D1',      # Blue
        'head_turn': '#96CEB4',   # Green
        'stand_grooming': '#FFEAA7', # Yellow
        'stand_rearing': '#DDA0DD',  # Plum
        'stand_lean': '#98D8C8',    # Mint
        'other': '#F7F7F7'          # Light gray
    }
    
    # Create the main behavior timeline plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Final behavior classification over time
    ax1.set_title('Mouse Behavior Timeline - Final Classification', fontsize=14, fontweight='bold')
    
    # Create colored segments for each behavior
    current_behavior = None
    start_time = 0
    start_frame = 0
    
    for i, behavior in enumerate(combined_results['final_prediction']):
        if behavior != current_behavior:
            # Plot the previous segment
            if current_behavior is not None:
                color = behavior_colors.get(current_behavior, behavior_colors['other'])
                ax1.axvspan(start_time, time_axis[i], alpha=0.6, color=color, 
                           label=current_behavior if start_frame == 0 else "")
            
            # Start new segment
            current_behavior = behavior
            start_time = time_axis[i]
            start_frame = i
    
    # Plot the last segment
    if current_behavior is not None:
        color = behavior_colors.get(current_behavior, behavior_colors['other'])
        ax1.axvspan(start_time, time_axis[-1], alpha=0.6, color=color, 
                   label=current_behavior if start_frame == 0 else "")
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Behavior Class')
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: XGBoost vs Original Logic comparison (if XGBoost was used)
    if use_xgboost and 'xgb_prediction' in combined_results.columns:
        ax2.set_title('XGBoost vs Original Logic Classification', fontsize=14, fontweight='bold')
        
        # Plot XGBoost predictions
        xgb_predictions = combined_results['xgb_prediction'].values
        xgb_active = xgb_predictions != 'none'
        
        # Create XGBoost activity timeline
        for i, (is_active, prediction) in enumerate(zip(xgb_active, xgb_predictions)):
            if is_active:
                color = behavior_colors.get(prediction, behavior_colors['other'])
                ax2.axvspan(time_axis[i], time_axis[i] + 1/sampling_rate, 
                           alpha=0.8, color=color, linewidth=0)
        
        # Add legend for XGBoost
        ax2.text(0.02, 0.95, 'XGBoost Active (Colored bars)', transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add original logic predictions as reference
        if original_results is not None:
            orig_predictions = original_results['original_prediction'].values
            for i, prediction in enumerate(orig_predictions):
                if prediction != 'other':
                    color = behavior_colors.get(prediction, behavior_colors['other'])
                    ax2.axvline(time_axis[i], color=color, alpha=0.3, linewidth=0.5)
            
            ax2.text(0.02, 0.85, 'Original Logic (Vertical lines)', transform=ax2.transAxes, 
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # If no XGBoost, show original logic timeline
        ax2.set_title('Original Logic Classification Timeline', fontsize=14, fontweight='bold')
        
        if original_results is not None:
            orig_predictions = original_results['original_prediction'].values
            
            # Create colored segments for original logic
            current_behavior = None
            start_time = 0
            
            for i, behavior in enumerate(orig_predictions):
                if behavior != current_behavior:
                    # Plot the previous segment
                    if current_behavior is not None:
                        color = behavior_colors.get(current_behavior, behavior_colors['other'])
                        ax2.axvspan(start_time, time_axis[i], alpha=0.6, color=color, 
                                   label=current_behavior if start_time == 0 else "")
                    
                    # Start new segment
                    current_behavior = behavior
                    start_time = time_axis[i]
            
            # Plot the last segment
            if current_behavior is not None:
                color = behavior_colors.get(current_behavior, behavior_colors['other'])
                ax2.axvspan(start_time, time_axis[-1], alpha=0.6, color=color, 
                           label=current_behavior if start_time == 0 else "")
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Behavior Class')
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timeline_path = os.path.join(output_dir, 'behavior_timeline.png')
    fig.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Behavior timeline plot saved to: {timeline_path}")
    return timeline_path

def create_xgboost_activity_plot(combined_results, sampling_rate, output_dir):
    """
    Create a detailed time-plot showing when XGBoost is active and what it predicts.
    """
    print("Creating XGBoost activity plot...")
    
    # Create time axis (seconds)
    total_frames = len(combined_results)
    time_axis = np.arange(total_frames) / sampling_rate
    
    # Define behavior colors
    behavior_colors = {
        'freeze': '#FF6B6B',      # Red
        'walk': '#4ECDC4',        # Teal
        'rotate': '#45B7D1',      # Blue
        'head_turn': '#96CEB4',   # Green
        'stand_grooming': '#FFEAA7', # Yellow
        'stand_rearing': '#DDA0DD',  # Plum
        'stand_lean': '#98D8C8',    # Mint
        'other': '#F7F7F7'          # Light gray
    }
    
    # Create the XGBoost activity plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: XGBoost predictions over time
    ax1.set_title('XGBoost Behavior Predictions Over Time', fontsize=14, fontweight='bold')
    
    xgb_predictions = combined_results['xgb_prediction'].values
    xgb_confidence = combined_results['xgb_confidence'].values
    
    # Create colored segments for XGBoost predictions
    current_behavior = None
    start_time = 0
    start_frame = 0
    
    for i, (behavior, confidence) in enumerate(zip(xgb_predictions, xgb_confidence)):
        if behavior != 'none' and behavior != current_behavior:
            # Plot the previous segment
            if current_behavior is not None:
                color = behavior_colors.get(current_behavior, behavior_colors['other'])
                ax1.axvspan(start_time, time_axis[i], alpha=0.7, color=color, 
                           label=current_behavior if start_frame == 0 else "")
            
            # Start new segment
            current_behavior = behavior
            start_time = time_axis[i]
            start_frame = i
    
    # Plot the last segment
    if current_behavior is not None and current_behavior != 'none':
        color = behavior_colors.get(current_behavior, behavior_colors['other'])
        ax1.axvspan(start_time, time_axis[-1], alpha=0.7, color=color, 
                   label=current_behavior if start_frame == 0 else "")
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('XGBoost Predictions')
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: XGBoost confidence over time
    ax2.set_title('XGBoost Confidence Scores Over Time', fontsize=14, fontweight='bold')
    
    # Plot confidence scores
    ax2.plot(time_axis, xgb_confidence, color='purple', linewidth=1, alpha=0.8)
    ax2.fill_between(time_axis, xgb_confidence, alpha=0.3, color='purple')
    
    # Add confidence threshold line
    confidence_threshold = 0.7  # Default threshold
    ax2.axhline(y=confidence_threshold, color='red', linestyle='--', alpha=0.7, 
                label=f'Confidence Threshold ({confidence_threshold})')
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Confidence Score')
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: XGBoost activity heatmap
    ax3.set_title('XGBoost Activity Heatmap', fontsize=14, fontweight='bold')
    
    # Create activity matrix
    xgb_active = (xgb_predictions != 'none').astype(int)
    
    # Reshape for better visualization (every 10 frames as one row)
    frame_chunk_size = 10
    num_chunks = total_frames // frame_chunk_size
    if total_frames % frame_chunk_size != 0:
        num_chunks += 1
    
    activity_matrix = np.zeros((num_chunks, frame_chunk_size))
    
    for i in range(num_chunks):
        start_idx = i * frame_chunk_size
        end_idx = min(start_idx + frame_chunk_size, total_frames)
        chunk_length = end_idx - start_idx
        activity_matrix[i, :chunk_length] = xgb_active[start_idx:end_idx]
    
    # Create heatmap
    im = ax3.imshow(activity_matrix, cmap='RdYlBu', aspect='auto', 
                    extent=[0, total_frames/frame_chunk_size, 0, num_chunks])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('XGBoost Active (1) / Inactive (0)')
    
    ax3.set_xlabel('Time Chunks (10 frames each)')
    ax3.set_ylabel('Frame Groups')
    ax3.set_title('XGBoost Activity Pattern')
    
    plt.tight_layout()
    
    # Save the plot
    xgb_activity_path = os.path.join(output_dir, 'xgboost_activity_timeline.png')
    fig.savefig(xgb_activity_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"XGBoost activity plot saved to: {xgb_activity_path}")
    return xgb_activity_path

class OpenFieldApp:
    def __init__(self, root):
        self.root = root
        root.title('Open-Field & Head Gesture Analysis')
        
        # Set custom icon
        icon_path = os.path.join(os.path.dirname(__file__), 'mouse_icon.ico')
        try:
            root.iconbitmap(icon_path)
        except Exception:
            pass
        try:
            import PIL.Image, PIL.ImageTk
            icon_img = tk.PhotoImage(file=icon_path)
            root.iconphoto(True, icon_img)
        except Exception:
            pass
            
        root.geometry('600x450')
        
        # Variables
        self.csv_path = tk.StringVar()
        self.sampling_rate = tk.IntVar(value=30)
        self.arena_width = tk.DoubleVar(value=40.0)
        self.arena_height = tk.DoubleVar(value=40.0)
        self.output_dir = tk.StringVar()
        self.heatmap_cmap = tk.StringVar(value='jet')
        self.heatmap_vmin = tk.StringVar(value='')
        self.heatmap_vmax = tk.StringVar(value='')
        self.max_jump_distance = tk.DoubleVar(value=50.0)
        
        # XGBoost variables
        self.xgb_model_path = tk.StringVar()
        self.xgb_model = None
        self.xgb_scaler = None
        self.xgb_label_encoder = None
        self.xgb_feature_names = None
        self.use_xgboost = tk.BooleanVar(value=False)  # Leave unchecked at beginning
        self.xgb_confidence_threshold = tk.DoubleVar(value=0.7)

        # GUI setup
        self.notebook = ttk.Notebook(root)
        self.frame_analysis = tk.Frame(self.notebook)
        self.frame_xgboost = tk.Frame(self.notebook)
        self.frame_help = tk.Frame(self.notebook)
        self.notebook.add(self.frame_analysis, text='Analysis')
        self.notebook.add(self.frame_xgboost, text='XGBoost Model')
        self.notebook.add(self.frame_help, text='Help')
        self.notebook.pack(fill='both', expand=True)

        # Analysis Tab
        row = 0
        tk.Label(self.frame_analysis, text='CSV File:').grid(row=row, column=0, sticky='e', padx=5, pady=5)
        tk.Entry(self.frame_analysis, textvariable=self.csv_path, width=40).grid(row=row, column=1, padx=5)
        tk.Button(self.frame_analysis, text='Browse', command=self.browse_csv).grid(row=row, column=2, padx=5)
        
        row += 1
        tk.Label(self.frame_analysis, text='Sampling Rate (fps):').grid(row=row, column=0, sticky='e', padx=5, pady=5)
        tk.Entry(self.frame_analysis, textvariable=self.sampling_rate, width=10).grid(row=row, column=1, sticky='w', padx=5)
        
        row += 1
        tk.Label(self.frame_analysis, text='Arena Width (cm):').grid(row=row, column=0, sticky='e', padx=5, pady=5)
        tk.Entry(self.frame_analysis, textvariable=self.arena_width, width=10).grid(row=row, column=1, sticky='w', padx=5)
        
        row += 1
        tk.Label(self.frame_analysis, text='Arena Height (cm):').grid(row=row, column=0, sticky='e', padx=5, pady=5)
        tk.Entry(self.frame_analysis, textvariable=self.arena_height, width=10).grid(row=row, column=1, sticky='w', padx=5)
        
        row += 1
        tk.Label(self.frame_analysis, text='Max Jump Distance (pixels):').grid(row=row, column=0, sticky='e', padx=5, pady=5)
        tk.Entry(self.frame_analysis, textvariable=self.max_jump_distance, width=10).grid(row=row, column=1, sticky='w', padx=5)
        
        row += 1
        tk.Label(self.frame_analysis, text='Heatmap Color Scheme:').grid(row=row, column=0, sticky='e', padx=5, pady=5)
        cmap_options = ['hot', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet']
        ttk.OptionMenu(self.frame_analysis, self.heatmap_cmap, self.heatmap_cmap.get(), *cmap_options).grid(row=row, column=1, sticky='w', padx=5)
        
        row += 1
        tk.Label(self.frame_analysis, text='Heatmap Colorbar Min:').grid(row=row, column=0, sticky='e', padx=5, pady=5)
        tk.Entry(self.frame_analysis, textvariable=self.heatmap_vmin, width=10).grid(row=row, column=1, sticky='w', padx=5)
        
        row += 1
        tk.Label(self.frame_analysis, text='Heatmap Colorbar Max:').grid(row=row, column=0, sticky='e', padx=5, pady=5)
        tk.Entry(self.frame_analysis, textvariable=self.heatmap_vmax, width=10).grid(row=row, column=1, sticky='w', padx=5)
        
        row += 1
        tk.Label(self.frame_analysis, text='Output Folder:').grid(row=row, column=0, sticky='e', padx=5, pady=5)
        tk.Entry(self.frame_analysis, textvariable=self.output_dir, width=40).grid(row=row, column=1, padx=5)
        tk.Button(self.frame_analysis, text='Browse', command=self.browse_output).grid(row=row, column=2, padx=5)
        
        row += 1
        tk.Button(self.frame_analysis, text='Run Analysis', command=self.run_analysis).grid(row=row, column=1, pady=20)

        # XGBoost Tab
        row = 0
        tk.Label(self.frame_xgboost, text='XGBoost Model File (.joblib):').grid(row=row, column=0, sticky='e', padx=5, pady=5)
        tk.Entry(self.frame_xgboost, textvariable=self.xgb_model_path, width=40).grid(row=row, column=1, padx=5)
        tk.Button(self.frame_xgboost, text='Browse', command=self.browse_xgb_model).grid(row=row, column=2, padx=5)
        
        row += 1
        tk.Button(self.frame_xgboost, text='Load Model', command=self.load_xgb_model).grid(row=row, column=1, pady=10)
        
        row += 1
        # Add refresh button for auto-detection
        tk.Button(self.frame_xgboost, text='🔄 Refresh Auto-Detection', command=self.auto_detect_xgb_models).grid(row=row, column=1, pady=5)
        
        row += 1
        tk.Checkbutton(self.frame_xgboost, text='Use XGBoost for behavior classification', variable=self.use_xgboost).grid(row=row, column=0, columnspan=3, pady=5)
        
        row += 1
        tk.Label(self.frame_xgboost, text='Confidence Threshold:').grid(row=row, column=0, sticky='e', padx=5, pady=5)
        tk.Entry(self.frame_xgboost, textvariable=self.xgb_confidence_threshold, width=10).grid(row=row, column=1, sticky='w', padx=5)
        tk.Label(self.frame_xgboost, text='(0.0-1.0, higher = more confident)').grid(row=row, column=2, sticky='w', padx=5)
        
        row += 1
        # Model status display
        self.xgb_status_label = tk.Label(self.frame_xgboost, text='No model loaded', fg='red')
        self.xgb_status_label.grid(row=row, column=0, columnspan=3, pady=10)
        
        row += 1
        # Model info display
        self.xgb_info_text = tk.Text(self.frame_xgboost, height=8, width=60, wrap=tk.WORD)
        self.xgb_info_text.grid(row=row, column=0, columnspan=3, padx=5, pady=5)
        self.xgb_info_text.insert(tk.END, "Load an XGBoost model to see information here.")
        self.xgb_info_text.config(state=tk.DISABLED)
        
        row += 1
        # Progress bar for feature calculation
        self.progress_label = tk.Label(self.frame_xgboost, text='Ready for analysis')
        self.progress_label.grid(row=row, column=0, columnspan=3, pady=5)
        
        self.progress_bar = ttk.Progressbar(self.frame_xgboost, mode='indeterminate')
        self.progress_bar.grid(row=row+1, column=0, columnspan=3, padx=5, pady=5, sticky='ew')

        # Check XGBoost availability and auto-detect models
        self.check_xgboost_availability()
        
        # Schedule auto-detection after GUI is fully initialized
        self.root.after(1000, self.auto_detect_xgb_models)

        # Help Tab
        help_text = (
            'Required CSV columns:\n'
            'body_center_x, body_center_y, tail_base_x, tail_base_y, forelimbs_x, forelimbs_y, head_x, head_y\n\n'
            'Each row represents a video frame. Missing or bad data should be left blank or as NaN.\n'
            'Sampling rate is the video frame rate (e.g., 30 for 30 fps).\n'
            'Arena width/height should be in centimeters.\n'
            'Max jump distance: Maximum allowed distance between consecutive frames (pixels).\n'
            'Points that jump further than this will be marked as NaN.\n\n'
            'Heatmap options:\n'
            '- Color scheme: Choose from hot, viridis, plasma, inferno, magma, cividis, jet. Default is "hot".\n'
            '- Colorbar min/max: Leave blank to auto-scale. Clear fields to reset to default.\n\n'
            'XGBoost Model:\n'
            '- The script automatically searches for .joblib models in the same directory.\n'
            '- Load a trained XGBoost model (.joblib file) for improved behavior classification.\n'
            '- The model will be used when confidence is above the threshold.\n'
            '- Original logic will be used as fallback when XGBoost is uncertain.\n'
            '- Higher confidence threshold = more reliance on XGBoost predictions.\n\n'
            'Timeline Plots:\n'
            '- Behavior Timeline: Shows mouse behavior over time with color-coded segments.\n'
            '- XGBoost Activity: Displays when XGBoost is active and its predictions.\n'
            '- Each behavior class has a unique color for easy identification.\n\n'
            'The script will output summary tables, plots, and a heatmap to the selected output folder.'
        )
        tk.Label(self.frame_help, text=help_text, justify='left', anchor='nw', wraplength=550).pack(fill='both', expand=True, padx=10, pady=10)

    def browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        if path:
            self.csv_path.set(path)

    def browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def browse_xgb_model(self):
        path = filedialog.askopenfilename(filetypes=[('Joblib Files', '*.joblib')])
        if path:
            self.xgb_model_path.set(path)

    def check_xgboost_availability(self):
        """Check if XGBoost is available and update GUI accordingly."""
        if not XGBOOST_AVAILABLE:
            self.xgb_status_label.config(text='XGBoost not available - install with: pip install xgboost joblib', fg='orange')
            self.xgb_info_text.config(state=tk.NORMAL)
            self.xgb_info_text.delete(1.0, tk.END)
            self.xgb_info_text.insert(tk.END, "XGBoost is not available in your environment.\n")
            self.xgb_info_text.insert(tk.END, "To use XGBoost behavior classification:\n")
            self.xgb_info_text.insert(tk.END, "1. Install XGBoost: pip install xgboost\n")
            self.xgb_info_text.insert(tk.END, "2. Install joblib: pip install joblib\n")
            self.xgb_info_text.insert(tk.END, "3. Restart the application\n\n")
            self.xgb_info_text.insert(tk.END, "Original logic will be used for behavior classification.")
            self.xgb_info_text.config(state=tk.DISABLED)
            self.use_xgboost.set(False)
            return False
        return True

    def update_progress(self, message, show_progress=True):
        """Update progress bar and status message."""
        self.progress_label.config(text=message)
        if show_progress:
            self.progress_bar.start()
        else:
            self.progress_bar.stop()
        self.root.update_idletasks()

    def load_xgb_model(self):
        model_path = self.xgb_model_path.get()
        if not model_path:
            messagebox.showerror('Error', 'Please select an XGBoost model file.')
            return
        try:
            # Load the model data
            model_data = joblib.load(model_path)
            
            # Extract components based on the structure from the training script
            if isinstance(model_data, dict):
                # New format from training script
                self.xgb_model = model_data['model']
                self.xgb_scaler = model_data['scaler']
                self.xgb_label_encoder = model_data['label_encoder']
                self.xgb_feature_names = model_data['feature_names']
            else:
                # Old format or different structure
                self.xgb_model = model_data
                self.xgb_scaler = None
                self.xgb_label_encoder = None
                self.xgb_feature_names = None
            
            # Update status and display
            self.xgb_status_label.config(text='Model loaded successfully', fg='green')
            self.xgb_info_text.config(state=tk.NORMAL)
            self.xgb_info_text.delete(1.0, tk.END)
            self.xgb_info_text.insert(tk.END, f"Model loaded from: {model_path}\n")
            
            if self.xgb_feature_names:
                self.xgb_info_text.insert(tk.END, f"Number of features: {len(self.xgb_feature_names)}\n")
                self.xgb_info_text.insert(tk.END, f"Feature names: {', '.join(self.xgb_feature_names[:5])}...\n")
            else:
                self.xgb_info_text.insert(tk.END, "Feature names: Not available\n")
            
            if self.xgb_label_encoder:
                self.xgb_info_text.insert(tk.END, f"Behavior classes: {', '.join(self.xgb_label_encoder.classes_)}\n")
            else:
                self.xgb_info_text.insert(tk.END, "Behavior classes: Not available\n")
            
            self.xgb_info_text.insert(tk.END, f"Confidence threshold: {self.xgb_confidence_threshold.get()}\n")
            
            if self.xgb_scaler and self.xgb_label_encoder:
                self.xgb_info_text.insert(tk.END, "Model is ready for behavior classification.\n")
            else:
                self.xgb_info_text.insert(tk.END, "Model loaded but some components missing.\n")
                self.xgb_info_text.insert(tk.END, "Original logic will be used as fallback.\n")
            
            self.xgb_info_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load model: {e}')
            self.xgb_status_label.config(text='Model load failed', fg='red')
            self.xgb_info_text.config(state=tk.DISABLED)

    def run_analysis(self):
        csv_path = self.csv_path.get()
        output_dir = self.output_dir.get()
        if not csv_path or not output_dir:
            messagebox.showerror('Error', 'Please select both a CSV file and an output folder.')
            return
        try:
            max_jump_distance = self.max_jump_distance.get()
            
            # Load and validate CSV first
            print("Loading CSV file...")
            df_full = pd.read_csv(csv_path)
            
            # Validate CSV structure
            is_valid, error_msg, cleaned_df = validate_csv_structure(df_full)
            if not is_valid:
                messagebox.showerror('CSV Error', error_msg)
                return
            
            # Now proceed with outlier removal and analysis
            print("Removing tracking outliers...")
            df_full = remove_tracking_outliers(cleaned_df, max_jump_distance)
            
            # Only require body_center_x and body_center_y for main analyses
            required_main = ['body_center_x', 'body_center_y']
            valid_mask = df_full[required_main].notna().all(axis=1)
            df_valid = df_full[valid_mask].copy()
            
            # Ensure proper index alignment
            print(f"Full DataFrame shape: {df_full.shape}")
            print(f"Valid DataFrame shape: {df_valid.shape}")
            print(f"Valid mask sum: {valid_mask.sum()}")
            
            # Reset indices to ensure proper alignment
            df_full = df_full.reset_index(drop=True)
            df_valid = df_valid.reset_index(drop=True)
            valid_mask = valid_mask.reset_index(drop=True)
            
            print(f"After reset - Full DataFrame shape: {df_full.shape}")
            print(f"After reset - Valid DataFrame shape: {df_valid.shape}")
            print(f"After reset - Valid mask sum: {valid_mask.sum()}")
            
            if len(df_full) == 0:
                messagebox.showerror('Error', 'No data rows found in the CSV file.')
                return
            
            # Set parameters
            sampling_rate = self.sampling_rate.get()
            arena_width = self.arena_width.get()
            arena_height = self.arena_height.get()
            
            # Run analyses and save outputs
            self.save_analysis_outputs(df_full, df_valid, valid_mask, sampling_rate, arena_width, arena_height, output_dir)
            messagebox.showinfo('Done', 'Analysis complete. Results saved to output folder.')
        except Exception as e:
            messagebox.showerror('Error', str(e))
            import traceback
            traceback.print_exc()

    def save_analysis_outputs(self, df_full, df_valid, valid_mask, sampling_rate, arena_width, arena_height, output_dir):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        import os
        
        total_frames = len(df_full)
        
        # Check if XGBoost is enabled
        use_xgboost = self.use_xgboost.get()
        
        if use_xgboost:
            # Use hybrid approach: Original algorithm first, then XGBoost replacement (except for 'other')
            print("Using hybrid behavior classification approach (XGBoost enabled)...")
            self.update_progress("Running hybrid behavior classification...", True)
            
            # Run hybrid classification
            hybrid_results = classify_behaviors_hybrid(
                df_valid, valid_mask, sampling_rate,
                self.xgb_model, self.xgb_scaler, 
                self.xgb_label_encoder, self.xgb_feature_names,
                output_dir, total_frames
            )
        else:
            # Use only original logic (XGBoost disabled)
            print("Using original logic only for behavior classification (XGBoost disabled)...")
            self.update_progress("Using original logic for behavior classification...", True)
            
            # Run original classification only
            original_results = classify_behaviors_original(df_valid, valid_mask, sampling_rate)
            if original_results is not None:
                original_results.to_csv(os.path.join(output_dir, 'original_predictions.csv'), index=False)
                
                # Calculate behavior statistics from original logic
                behavior_counts = pd.Series(original_results['original_prediction']).value_counts()
                behavior_stats = []
                for behavior, count in behavior_counts.items():
                    if behavior != 'other':
                        time_seconds = count / sampling_rate
                        behavior_stats.append({
                            'Behavior': behavior,
                            'Frames': count,
                            'Time (s)': round(time_seconds, 2),
                            'Percentage': round((count / total_frames) * 100, 1)
                        })
                
                if behavior_stats:
                    behavior_df = pd.DataFrame(behavior_stats)
                    behavior_df.to_csv(os.path.join(output_dir, 'original_behavior_summary.csv'), index=False)
                    
                    # Create behavior summary plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    behaviors = [stat['Behavior'] for stat in behavior_stats]
                    times = [stat['Time (s)'] for stat in behavior_stats]
                    
                    bars = ax.bar(behaviors, times, color='lightblue', alpha=0.7)
                    ax.set_title('Behavior Classification by Original Logic')
                    ax.set_xlabel('Behavior Type')
                    ax.set_ylabel('Time (seconds)')
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add value labels on bars
                    for bar, time in zip(bars, times):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{time}s', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    fig.savefig(os.path.join(output_dir, 'original_behavior_summary.png'), dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                    # Create timeline plot
                    self.update_progress("Creating timeline plots...", True)
                    combined_results = pd.DataFrame({
                        'frame_index': range(total_frames),
                        'final_prediction': original_results['original_prediction'].values,
                        'final_confidence': original_results['original_confidence'].values
                    })
                    create_behavior_timeline_plot(combined_results, original_results, sampling_rate, output_dir, use_xgboost=False)
                
                print(f"Original logic classification completed successfully")
                return
            
            else:
                print("Original classification failed")
                return
        
        # This line was duplicated and causing issues - removing it
        # hybrid_results = classify_behaviors_hybrid(...)
        
        if hybrid_results is not None:
            hybrid_results_df, original_results, xgb_results = hybrid_results
            
            # Save all results
            print("Saving hybrid classification results...")
            
            # Save hybrid results (main output)
            hybrid_results_df.to_csv(os.path.join(output_dir, 'hybrid_predictions.csv'), index=False)
            
            # Save original results
            if original_results is not None:
                original_results.to_csv(os.path.join(output_dir, 'original_predictions.csv'), index=False)
            
            # Save XGBoost results if available
            if xgb_results is not None:
                xgb_results.to_csv(os.path.join(output_dir, 'xgboost_predictions.csv'), index=False)
                
                # Enhanced features were already calculated during hybrid classification, no need to recalculate
                print("XGBoost results saved successfully")
            
            # Create behavior summary from final predictions
            self.update_progress("Creating behavior summary plots...", True)
            
            # Ensure we have the right number of predictions
            final_predictions = hybrid_results_df['final_prediction'].values
            final_confidence = hybrid_results_df['final_confidence'].values
            
            print(f"Debug: hybrid_results_df shape: {hybrid_results_df.shape}")
            print(f"Debug: hybrid_results_df columns: {list(hybrid_results_df.columns)}")
            print(f"Debug: final_predictions length: {len(final_predictions)}, total_frames: {total_frames}")
            
            # Verify array lengths match
            if len(final_predictions) != total_frames:
                print(f"Warning: Array length mismatch. final_predictions: {len(final_predictions)}, total_frames: {total_frames}")
                print(f"Debug: This suggests the frame mapping in hybrid function didn't work properly")
                
                # Try to fix the mismatch by creating a full-frame version
                print("Attempting to fix array length mismatch...")
                
                # Create full-frame arrays with default values
                full_final_pred = np.full(total_frames, 'other')
                full_final_conf = np.full(total_frames, 0.0)
                
                # Copy available predictions
                copy_length = min(len(final_predictions), total_frames)
                full_final_pred[:copy_length] = final_predictions[:copy_length]
                full_final_conf[:copy_length] = final_confidence[:copy_length]
                
                final_predictions = full_final_pred
                final_confidence = full_final_conf
                
                print(f"Fixed: final_predictions length now: {len(final_predictions)}")
            else:
                print("Array lengths match correctly!")
            
            behavior_counts = pd.Series(final_predictions).value_counts()
            behavior_stats = []
            for behavior, count in behavior_counts.items():
                if behavior != 'other':
                    time_seconds = count / sampling_rate
                    behavior_stats.append({
                        'Behavior': behavior,
                        'Frames': count,
                        'Time (s)': round(time_seconds, 2),
                        'Percentage': round((count / total_frames) * 100, 1)
                    })
            
            if behavior_stats:
                behavior_df = pd.DataFrame(behavior_stats)
                behavior_df.to_csv(os.path.join(output_dir, 'hybrid_behavior_summary.csv'), index=False)
                
                # Create behavior summary plot
                fig, ax = plt.subplots(figsize=(10, 6))
                behaviors = [stat['Behavior'] for stat in behavior_stats]
                times = [stat['Time (s)'] for stat in behavior_stats]
                
                bars = ax.bar(behaviors, times, color='lightgreen', alpha=0.7)
                ax.set_title('Behavior Classification by Hybrid Approach (Original + XGBoost)')
                ax.set_xlabel('Behavior Type')
                ax.set_ylabel('Time (seconds)')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, time in zip(bars, times):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{time}s', ha='center', va='bottom')
                
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, 'hybrid_behavior_summary.png'), dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # Create timeline plots
                self.update_progress("Creating timeline plots...", True)
                
                # Create combined results for timeline plotting
                # The hybrid function should now return properly sized results
                combined_results = hybrid_results_df.copy()
                
                # Ensure all necessary columns are present
                print(f"Debug: combined_results columns: {list(combined_results.columns)}")
                print(f"Debug: combined_results shape: {combined_results.shape}")
                
                # Check if XGBoost columns are missing and add them if needed
                if 'xgb_prediction' not in combined_results.columns:
                    print("Warning: xgb_prediction column missing, adding default values")
                    combined_results['xgb_prediction'] = ['none'] * len(combined_results)
                if 'xgb_confidence' not in combined_results.columns:
                    print("Warning: xgb_confidence column missing, adding default values")
                    combined_results['xgb_prediction'] = [0.0] * len(combined_results)
                
                create_behavior_timeline_plot(combined_results, original_results, sampling_rate, output_dir, use_xgboost=False)
                
                # Create XGBoost activity plot if available
                if xgb_results is not None:
                    create_xgboost_activity_plot(combined_results, sampling_rate, output_dir)
            
            # Print summary statistics
            print(f"\nHybrid Classification Summary:")
            print(f"Total frames: {total_frames}")
            print(f"Original predictions: {len(original_results) if original_results is not None else 0}")
            if xgb_results is not None:
                print(f"XGBoost predictions: {len(xgb_results)}")
                print(f"Final predictions: {len(hybrid_results_df)}")
                
                # Count replacements
                replacement_mask = hybrid_results_df['original_prediction'] != hybrid_results_df['final_prediction']
                replacement_count = replacement_mask.sum()
                print(f"Predictions replaced by XGBoost: {replacement_count}")
                print(f"Original predictions preserved: {len(hybrid_results_df) - replacement_count}")
            
        else:
            print("Hybrid classification failed, falling back to original logic only")
            self.update_progress("Hybrid classification failed, using original logic...", True)
            
            # Fallback to original logic only
            original_results = classify_behaviors_original(df_valid, valid_mask, sampling_rate)
            if original_results is not None:
                original_results.to_csv(os.path.join(output_dir, 'original_predictions.csv'), index=False)
                
                # Calculate behavior statistics from original logic
                behavior_counts = pd.Series(original_results['original_prediction']).value_counts()
                behavior_stats = []
                for behavior, count in behavior_counts.items():
                    if behavior != 'other':
                        time_seconds = count / sampling_rate
                        behavior_stats.append({
                            'Behavior': behavior,
                            'Frames': count,
                            'Time (s)': round(time_seconds, 2),
                            'Percentage': round((count / total_frames) * 100, 1)
                        })
                
                if behavior_stats:
                    behavior_df = pd.DataFrame(behavior_stats)
                    behavior_df.to_csv(os.path.join(output_dir, 'original_behavior_summary.csv'), index=False)
                    
                    # Create behavior summary plot for original logic
                    fig, ax = plt.subplots(figsize=(10, 6))
                    behaviors = [stat['Behavior'] for stat in behavior_stats]
                    times = [stat['Time (s)'] for stat in behavior_stats]
                    
                    bars = ax.bar(behaviors, times, color='lightcoral', alpha=0.7)
                    ax.set_title('Behavior Classification by Original Logic')
                    ax.set_xlabel('Behavior Type')
                    ax.set_ylabel('Time (seconds)')
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add value labels on bars
                    for bar, time in zip(bars, times):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{time}s', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    fig.savefig(os.path.join(output_dir, 'original_behavior_summary.png'), dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                    # Create timeline plots for original logic
                    self.update_progress("Creating timeline plots...", True)
                    # Create a simple combined results DataFrame for original logic
                    original_combined = pd.DataFrame({
                        'frame_index': range(total_frames),
                        'final_prediction': original_results['original_prediction'],
                        'final_confidence': original_results['original_confidence']
                    })
                    create_behavior_timeline_plot(original_combined, original_results, sampling_rate, output_dir, use_xgboost=False)
        
        # Save combined behavior analysis if both methods were used
        if use_xgboost and 'combined_results' in locals():
            # Create comparison plot between XGBoost and original logic
            try:
                self.update_progress("Creating comparison plots...", True)
                
                # Original results are already available from hybrid classification, no need to recalculate
                if 'original_results' in locals():
                    print("Using existing original results for comparison")
                else:
                    print("No original results available for comparison")
                if original_results is not None:
                    # Create comparison DataFrame
                    comparison_df = pd.DataFrame({
                        'frame_index': range(total_frames),
                        'xgb_prediction': combined_results['xgb_prediction'],
                        'xgb_confidence': combined_results['xgb_confidence'],
                        'original_prediction': original_results['original_prediction'],
                        'original_confidence': original_results['original_confidence'],
                        'final_prediction': combined_results['final_prediction']
                    })
                    
                    comparison_df.to_csv(os.path.join(output_dir, 'behavior_comparison.csv'), index=False)
                    
                    # Create comparison plot
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Plot 1: XGBoost vs Original predictions
                    xgb_counts = pd.Series(combined_results['xgb_prediction']).value_counts()
                    orig_counts = pd.Series(original_results['original_prediction']).value_counts()
                    
                    # Combine all behavior types
                    all_behaviors = list(set(list(xgb_counts.index) + list(orig_counts.index)))
                    all_behaviors = [b for b in all_behaviors if b != 'none' and b != 'other']
                    
                    xgb_values = [xgb_counts.get(b, 0) for b in all_behaviors]
                    orig_values = [orig_counts.get(b, 0) for b in all_behaviors]
                    
                    x = np.arange(len(all_behaviors))
                    width = 0.35
                    
                    ax1.bar(x - width/2, xgb_values, width, label='XGBoost', color='skyblue', alpha=0.7)
                    ax1.bar(x + width/2, orig_values, width, label='Original Logic', color='lightcoral', alpha=0.7)
                    
                    ax1.set_title('Behavior Classification Comparison: XGBoost vs Original Logic')
                    ax1.set_xlabel('Behavior Type')
                    ax1.set_ylabel('Number of Frames')
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(all_behaviors, rotation=45)
                    ax1.legend()
                    
                    # Plot 2: Confidence comparison
                    xgb_conf = combined_results['xgb_confidence'].values
                    orig_conf = original_results['original_confidence'].values
                    
                    ax2.hist(xgb_conf, bins=20, alpha=0.7, label='XGBoost Confidence', color='skyblue')
                    ax2.hist(orig_conf, bins=20, alpha=0.7, label='Original Logic Confidence', color='lightcoral')
                    ax2.set_title('Confidence Score Distribution')
                    ax2.set_xlabel('Confidence Score')
                    ax2.set_ylabel('Number of Frames')
                    ax2.legend()
                    
                    plt.tight_layout()
                    fig.savefig(os.path.join(output_dir, 'behavior_comparison_plot.png'), dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
            except Exception as e:
                print(f"Error creating comparison plot: {e}")
        
        # Final progress update
        self.update_progress("Analysis complete! Creating final outputs...", False)
        
        # Open-field analysis
        coordinates = df_valid[['body_center_x', 'body_center_y']].copy()
        coordinates.columns = ['X', 'Y']
        
        x_range = coordinates["X"].max() - coordinates["X"].min()
        y_range = coordinates["Y"].max() - coordinates["Y"].min()
        side_length = np.mean([x_range, y_range])
        scale = arena_width / side_length
        coordinates_scaled = coordinates * scale
        coordinates_normalized = coordinates_scaled.copy()
        coordinates_normalized["X"] -= coordinates_scaled["X"].min()
        coordinates_normalized["Y"] -= coordinates_scaled["Y"].min()
        
        norm_X_full = np.full(total_frames, np.nan)
        norm_Y_full = np.full(total_frames, np.nan)
        norm_X_full[valid_mask.values] = coordinates_normalized["X"].values
        norm_Y_full[valid_mask.values] = coordinates_normalized["Y"].values
        
        # Calculate metrics
        distances = np.sqrt(np.diff(norm_X_full)**2 + np.diff(norm_Y_full)**2)
        valid_dist = ~np.isnan(distances)
        total_distance_cm = np.nansum(distances[valid_dist])
        total_time_s = total_frames / sampling_rate
        average_speed_cmps = total_distance_cm / total_time_s if total_time_s > 0 else np.nan
        
        # Middle zone analysis
        middle_mask_full = (
            (norm_X_full >= 10) & (norm_X_full <= 30) &
            (norm_Y_full >= 10) & (norm_Y_full <= 30)
        )
        middle_mask_full[np.isnan(norm_X_full) | np.isnan(norm_Y_full)] = False
        time_middle_square_s = middle_mask_full.sum() / sampling_rate
        middle_entries = np.sum(middle_mask_full & ~np.roll(middle_mask_full, 1))
        if middle_mask_full[0]:
            middle_entries -= 1
        
        # Dwell times
        dwell_times = []
        in_zone = False
        counter = 0
        for point in middle_mask_full:
            if point:
                counter += 1
                in_zone = True
            elif in_zone:
                dwell_times.append(counter / sampling_rate)
                counter = 0
                in_zone = False
        if in_zone and counter > 0:
            dwell_times.append(counter / sampling_rate)
        
        # Save open-field results
        metrics = [
            "Total Distance (cm)",
            "Average Speed (cm/s)",
            "Dwell Time in Middle (s)",
            "Number of Entries",
            "Mean Dwell per Visit (s)"
        ]
        values = [
            round(total_distance_cm, 2),
            round(average_speed_cmps, 2),
            round(time_middle_square_s, 2),
            middle_entries,
            round(np.mean(dwell_times), 2) if dwell_times else 0
        ]
        pd.DataFrame([values], columns=metrics).to_csv(os.path.join(output_dir, 'openfield_summary.csv'), index=False)
        coordinates_normalized.to_csv(os.path.join(output_dir, 'normalized_track.csv'), index=False)
        pd.DataFrame({'dwell_times_s': dwell_times}).to_csv(os.path.join(output_dir, 'dwell_times.csv'), index=False)
        
        # Save track plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(coordinates_normalized["X"], coordinates_normalized["Y"], color='blue', linewidth=1)
        outer = plt.Rectangle((0, 0), arena_width, arena_height, fill=False, color='black', linewidth=1.5, linestyle='--')
        middle = plt.Rectangle((10, 10), 20, 20, fill=False, color='red', linewidth=1.5, linestyle=':')
        ax.add_patch(outer)
        ax.add_patch(middle)
        ax.set_title("Open-Field Track")
        ax.set_xlim(0, arena_width)
        ax.set_ylim(0, arena_height)
        ax.set_aspect('equal')
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'openfield_track.png'))
        plt.close(fig)
        
        # Save heatmap
        cmap = self.heatmap_cmap.get()
        vmin = self.heatmap_vmin.get()
        vmax = self.heatmap_vmax.get()
        vmin = float(vmin) if vmin.strip() != '' else None
        vmax = float(vmax) if vmax.strip() != '' else None
        
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        try:
            kde = sns.kdeplot(
                x=coordinates_normalized["X"],
                y=coordinates_normalized["Y"],
                fill=True, cmap=cmap, bw_adjust=0.1, thresh=0.05, ax=ax2,
                vmin=vmin, vmax=vmax
            )
            cbar = plt.colorbar(ax2.collections[0], ax=ax2, orientation='vertical')
            cbar.set_label('Time (ms)')
            ticks = cbar.get_ticks()
            time_ticks = ticks * (total_frames / sampling_rate) * 1000
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f'{t:.0f}' for t in time_ticks])
            if vmin is not None or vmax is not None:
                cbar.set_clim(vmin, vmax)
        except Exception:
            h = ax2.hist2d(
                coordinates_normalized["X"],
                coordinates_normalized["Y"], bins=50, cmap=cmap,
                vmin=vmin, vmax=vmax
            )
            h_time = h[0] / sampling_rate * 1000
            h_time_blur = gaussian_filter(h_time, sigma=2)
            ax2.clear()
            im = ax2.imshow(
                np.flipud(h_time_blur.T),
                extent=[0, arena_width, 0, arena_height],
                aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower'
            )
            plt.colorbar(im, ax=ax2, orientation='vertical', label='Time (ms)')
        
        ax2.set_title('Location Heatmap')
        ax2.set_xlim(0, arena_width)
        ax2.set_ylim(0, arena_height)
        ax2.set_aspect('equal')
        ax2.set_xlabel('X (cm)')
        ax2.set_ylabel('Y (cm)')
        plt.tight_layout()
        fig2.savefig(os.path.join(output_dir, 'location_heatmap.png'))
        plt.close(fig2)
        
        # Behavior analysis
        nose_x = np.full(total_frames, np.nan)
        nose_y = np.full(total_frames, np.nan)
        body_x = np.full(total_frames, np.nan)
        body_y = np.full(total_frames, np.nan)
        back_x = np.full(total_frames, np.nan)
        back_y = np.full(total_frames, np.nan)
        forelimb_x = np.full(total_frames, np.nan)
        forelimb_y = np.full(total_frames, np.nan)
        
        nose_x[valid_mask.values] = df_valid['head_x'].values
        nose_y[valid_mask.values] = df_valid['head_y'].values
        body_x[valid_mask.values] = df_valid['body_center_x'].values
        body_y[valid_mask.values] = df_valid['body_center_y'].values
        back_x[valid_mask.values] = df_valid['tail_base_x'].values
        back_y[valid_mask.values] = df_valid['tail_base_y'].values
        forelimb_x[valid_mask.values] = df_valid['forelimbs_x'].values
        forelimb_y[valid_mask.values] = df_valid['forelimbs_y'].values
        
        # Calculate speeds and distances
        dx = np.diff(body_x)
        dy = np.diff(body_y)
        speed = np.sqrt(dx**2 + dy**2) * sampling_rate
        speed = np.insert(speed, 0, np.nan)
        
        nose_body_dist = np.sqrt((nose_x - body_x)**2 + (nose_y - body_y)**2)
        body_back_dist = np.sqrt((body_x - back_x)**2 + (body_y - back_y)**2)
        
        # Thresholds
        speed_threshold = 0.1
        distance_threshold = 15.0
        edge_margin = 5.0
        gesture_angle_threshold = 15
        max_valid_head_turn = 45
        
        # Head angle calculation
        if np.isnan(back_x).all() or np.isnan(back_y).all():
            head_vec_x = nose_x - forelimb_x
            head_vec_y = nose_y - forelimb_y
            axis_x = forelimb_x - body_x
            axis_y = forelimb_y - body_y
        else:
            head_vec_x = nose_x - forelimb_x
            head_vec_y = nose_y - forelimb_y
            axis_x = body_x - back_x
            axis_y = body_y - back_y
        
        head_angle = np.full(total_frames, np.nan)
        for i in range(total_frames):
            if np.isnan(body_x[i]) or np.isnan(body_y[i]):
                continue
            if np.isnan(forelimb_x[i]) or np.isnan(forelimb_y[i]) or np.isnan(nose_x[i]) or np.isnan(nose_y[i]):
                continue
            if np.isnan(back_x[i]) or np.isnan(back_y[i]):
                axis_xi = forelimb_x[i] - body_x[i]
                axis_yi = forelimb_y[i] - body_y[i]
            else:
                axis_xi = body_x[i] - back_x[i]
                axis_yi = body_y[i] - back_y[i]
            head_vec_xi = nose_x[i] - forelimb_x[i]
            head_vec_yi = nose_y[i] - forelimb_y[i]
            if np.isnan(axis_xi) or np.isnan(axis_yi) or np.isnan(head_vec_xi) or np.isnan(head_vec_yi):
                continue
            angle = signed_angle((axis_xi, axis_yi), (head_vec_xi, head_vec_yi))
            # Check if angle is beyond 90 degrees and correct vector direction if needed
            if np.abs(angle) > 90:
                # Flip the head vector direction and recalculate
                head_vec_xi_flipped = -head_vec_xi
                head_vec_yi_flipped = -head_vec_yi
                angle_flipped = signed_angle((axis_xi, axis_yi), (head_vec_xi_flipped, head_vec_yi_flipped))
                # Use the smaller angle (closer to 0)
                if np.abs(angle_flipped) < np.abs(angle):
                    head_angle[i] = angle_flipped
                else:
                    head_angle[i] = angle
            else:
                head_angle[i] = angle
        
        # Behavior classification
        head_gesture = (np.abs(head_angle) > gesture_angle_threshold) & (np.abs(head_angle) < max_valid_head_turn)
        head_turn_time = np.nansum(head_gesture) / sampling_rate
        
        # Walking/freezing based on body speed
        body_dx = np.diff(body_x)
        body_dy = np.diff(body_y)
        body_speed = np.sqrt(body_dx**2 + body_dy**2) * sampling_rate
        body_speed = np.insert(body_speed, 0, np.nan)
        walk_speed_threshold = 3.0
        walking = body_speed > walk_speed_threshold
        freezing = body_speed <= walk_speed_threshold
        walk_time = np.nansum(walking) / sampling_rate
        freeze_time = np.nansum(freezing) / sampling_rate
        
        # Stand behavior - prioritize head-forelimb proximity (rearing)
        head_body_dist = np.sqrt((nose_x - body_x)**2 + (nose_y - body_y)**2)
        forelimb_body_dist = np.sqrt((forelimb_x - body_x)**2 + (forelimb_y - body_y)**2)
        head_forelimb_dist = np.sqrt((nose_x - forelimb_x)**2 + (nose_y - forelimb_y)**2)
        
        # Primary criteria: head and forelimb are close (rearing behavior)
        rearing_dist_thresh = 20.0  # Tighter threshold for rearing
        stand_rearing = head_forelimb_dist < rearing_dist_thresh
        
        # Secondary criteria: all points close together (original stand criteria)
        stand_dist_thresh = 25.0
        stand_all_close = (head_body_dist < stand_dist_thresh) & (forelimb_body_dist < stand_dist_thresh) & (head_forelimb_dist < stand_dist_thresh)
        
        # Combine: prioritize rearing, then all-close as secondary
        stand_multi = stand_rearing | stand_all_close
        stand_multi_time = np.nansum(stand_multi) / sampling_rate
        
        # Rotation
        body_axis_angle = np.arctan2(body_y - back_y, body_x - back_x)
        body_axis_angle_deg = np.degrees(body_axis_angle)
        body_axis_angle_diff = np.diff(body_axis_angle_deg)
        body_axis_angle_diff = (body_axis_angle_diff + 180) % 360 - 180
        angular_speed = np.abs(body_axis_angle_diff) * sampling_rate
        angular_speed = np.insert(angular_speed, 0, 0)
        rotate_thresh = 30
        rotating = angular_speed > rotate_thresh
        rotate_time = np.nansum(rotating) / sampling_rate
        
        # Save behavior results
        metrics2 = [
            "Freeze time (s)",
            "Stand (all close) time (s)",
            "Walk time (s)",
            "Rotate time (s)",
            "Head turn time (s)"
        ]
        values2 = [
            round(freeze_time, 2),
            round(stand_multi_time, 2),
            round(walk_time, 2),
            round(rotate_time, 2),
            round(head_turn_time, 2)
        ]
        pd.DataFrame([values2], columns=metrics2).to_csv(os.path.join(output_dir, 'behavior_summary.csv'), index=False)
        pd.DataFrame({'head_angle_deg': head_angle}).to_csv(os.path.join(output_dir, 'head_angle_over_time.csv'), index=False)
        
        # Save frame labels
        frame_labels = []
        for i in range(total_frames):
            if stand_multi[i]:
                frame_labels.append('stand')
            elif rotating[i]:
                frame_labels.append('rotate')
            elif head_gesture[i]:
                frame_labels.append('head_turn')
            elif walking[i]:
                frame_labels.append('walk')
            elif freezing[i]:
                frame_labels.append('freeze')
            else:
                frame_labels.append('other')
        pd.DataFrame({'frame_label': frame_labels}).to_csv(os.path.join(output_dir, 'frame_labels.csv'), index=False)
        
        # Save head angle plots
        fig_labeled = plt.figure()
        plt.plot(np.arange(total_frames) / sampling_rate, head_angle, label="Head Angle (deg)")
        plt.axhline(gesture_angle_threshold, color='r', linestyle='--', label="Gesture Threshold")
        plt.axhline(-gesture_angle_threshold, color='r', linestyle='--')
        plt.axhline(max_valid_head_turn, color='g', linestyle='--', label="Max Valid Turn")
        plt.axhline(-max_valid_head_turn, color='g', linestyle='--')
        plt.xlabel("Time (s)")
        plt.ylabel("Head Angle (deg)")
        plt.ylim(-180, 180)  # Set y-axis to full range
        plt.legend()
        plt.title("Head Angle Over Time")
        ylim = plt.ylim()
        plt.text(0, ylim[1]*0.9, 'Right', color='blue', ha='left', va='top', fontsize=12)
        plt.text(0, ylim[0]*0.9, 'Left', color='blue', ha='left', va='bottom', fontsize=12)
        plt.tight_layout()
        fig_labeled.savefig(os.path.join(output_dir, 'head_angle_over_time_labeled.png'))
        plt.close(fig_labeled)
        
        # Save head angle histogram
        fig_hist = plt.figure()
        valid_angles = head_angle[~np.isnan(head_angle)]
        plt.hist(valid_angles, bins=36, weights=np.ones_like(valid_angles) * (1 / sampling_rate), color='purple', alpha=0.7)
        plt.xlabel('Head Angle (deg)')
        plt.ylabel('Time (s)')
        plt.xlim(-90, 90)  # Set x-axis to -90 to 90 degrees for histogram
        plt.title('Distribution of Head Turn Angles by Time')
        plt.tight_layout()
        fig_hist.savefig(os.path.join(output_dir, 'head_angle_histogram.png'))
        plt.close(fig_hist)

    def auto_detect_xgb_models(self):
        """
        Automatically search for and load XGBoost models from the same directory as the script.
        """
        print("=== AUTO-DETECTION STARTED ===")
        print(f"Function called at: {__import__('datetime').datetime.now()}")
        print(f"Current working directory: {os.getcwd()}")
        
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available, skipping auto-detection")
            return
        
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"Script directory: {script_dir}")
            
            # Check if directory exists and is accessible
            if not os.path.exists(script_dir):
                print(f"ERROR: Script directory does not exist: {script_dir}")
                return
            
            print(f"Directory exists: {os.path.exists(script_dir)}")
            print(f"Directory accessible: {os.access(script_dir, os.R_OK)}")
            
            # List all files in directory
            try:
                all_files = os.listdir(script_dir)
                print(f"All files in directory: {len(all_files)} files")
                print(f"First 10 files: {all_files[:10]}")
            except Exception as e:
                print(f"ERROR listing directory: {e}")
                return
            
            # Search for .joblib files
            joblib_files = []
            for file in all_files:
                if file.endswith('.joblib'):
                    joblib_files.append(file)
                    print(f"Found joblib file: {file}")
            
            print(f"Total joblib files found: {len(joblib_files)}")
            print(f"Joblib files: {joblib_files}")
            
            if joblib_files:
                # Try to load the first available model
                first_model = joblib_files[0]
                model_path = os.path.join(script_dir, first_model)
                
                print(f"Attempting to auto-load: {first_model}")
                print(f"Full model path: {model_path}")
                print(f"Model file exists: {os.path.exists(model_path)}")
                print(f"Model file size: {os.path.getsize(model_path)} bytes")
                
                # Try to load the model
                try:
                    print("Loading model with joblib...")
                    model_data = joblib.load(model_path)
                    print(f"Model loaded successfully, type: {type(model_data)}")
                    
                    # Extract components based on the structure from the training script
                    if isinstance(model_data, dict):
                        print("Model is dictionary format, extracting components...")
                        print(f"Dictionary keys: {list(model_data.keys())}")
                        
                        # New format from training script
                        self.xgb_model = model_data.get('model')
                        self.xgb_scaler = model_data.get('scaler')
                        self.xgb_label_encoder = model_data.get('label_encoder')
                        self.xgb_feature_names = model_data.get('feature_names')
                        
                        print(f"Model extracted: {self.xgb_model is not None}")
                        print(f"Scaler extracted: {self.xgb_scaler is not None}")
                        print(f"Label encoder extracted: {self.xgb_label_encoder is not None}")
                        print(f"Feature names extracted: {self.xgb_feature_names is not None}")
                    else:
                        print("Model is not dictionary format, using as-is")
                        # Old format or different structure
                        self.xgb_model = model_data
                        self.xgb_scaler = None
                        self.xgb_label_encoder = None
                        self.xgb_feature_names = None
                    
                    # Update GUI
                    self.xgb_model_path.set(model_path)
                    self.xgb_status_label.config(text=f'Auto-loaded: {first_model}', fg='green')
                    
                    # Update info text
                    self.xgb_info_text.config(state=tk.NORMAL)
                    self.xgb_info_text.delete(1.0, tk.END)
                    self.xgb_info_text.insert(tk.END, f"Auto-detected and loaded: {first_model}\n")
                    self.xgb_info_text.insert(tk.END, f"Model path: {model_path}\n\n")
                    
                    if self.xgb_feature_names:
                        self.xgb_info_text.insert(tk.END, f"Number of features: {len(self.xgb_feature_names)}\n")
                        self.xgb_info_text.insert(tk.END, f"Feature names: {', '.join(self.xgb_feature_names[:5])}...\n")
                    else:
                        self.xgb_info_text.insert(tk.END, "Feature names: Not available\n")
                    
                    if self.xgb_label_encoder:
                        self.xgb_info_text.insert(tk.END, f"Behavior classes: {', '.join(self.xgb_label_encoder.classes_)}\n")
                    else:
                        self.xgb_info_text.insert(tk.END, "Behavior classes: Not available\n")
                    
                    self.xgb_info_text.insert(tk.END, f"Confidence threshold: {self.xgb_confidence_threshold.get()}\n\n")
                    
                    if self.xgb_scaler and self.xgb_label_encoder:
                        self.xgb_info_text.insert(tk.END, "Model is ready for behavior classification.\n")
                        self.xgb_info_text.insert(tk.END, "You can now enable XGBoost classification.\n")
                    else:
                        self.xgb_info_text.insert(tk.END, "Model loaded but some components missing.\n")
                        self.xgb_info_text.insert(tk.END, "Original logic will be used as fallback.\n")
                    
                    self.xgb_info_text.config(state=tk.DISABLED)
                    
                    print(f"Successfully auto-loaded XGBoost model: {first_model}")
                    print("=== AUTO-DETECTION SUCCESSFUL ===")
                    
                except Exception as e:
                    print(f"Failed to auto-load {first_model}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    self.xgb_status_label.config(text=f'Auto-load failed: {first_model}', fg='orange')
                    self.xgb_info_text.config(state=tk.NORMAL)
                    self.xgb_info_text.delete(1.0, tk.END)
                    self.xgb_info_text.insert(tk.END, f"Auto-detection found {first_model} but failed to load it.\n")
                    self.xgb_info_text.insert(tk.END, f"Error: {e}\n\n")
                    self.xgb_info_text.insert(tk.END, "Please try loading the model manually or check if it's corrupted.")
                    self.xgb_info_text.config(state=tk.DISABLED)
            else:
                print("No XGBoost models found in script directory")
                self.xgb_status_label.config(text='No models found - auto-load disabled', fg='blue')
                self.xgb_info_text.config(state=tk.NORMAL)
                self.xgb_info_text.delete(1.0, tk.END)
                self.xgb_info_text.insert(tk.END, "No XGBoost models (.joblib files) found in the script directory.\n\n")
                self.xgb_info_text.insert(tk.END, "To use XGBoost classification:\n")
                self.xgb_info_text.insert(tk.END, "1. Place your trained .joblib model in this directory\n")
                self.xgb_info_text.insert(tk.END, "2. Restart the application for auto-detection\n")
                self.xgb_info_text.insert(tk.END, "3. Or manually browse and load a model\n\n")
                self.xgb_info_text.insert(tk.END, "Original logic will be used for behavior classification.")
                self.xgb_info_text.config(state=tk.DISABLED)
                print("=== AUTO-DETECTION COMPLETED - NO MODELS FOUND ===")
                
        except Exception as e:
            print(f"Error during auto-detection: {e}")
            import traceback
            traceback.print_exc()
            
            self.xgb_status_label.config(text='Auto-detection error', fg='red')
            self.xgb_info_text.config(state=tk.NORMAL)
            self.xgb_info_text.delete(1.0, tk.END)
            self.xgb_info_text.insert(tk.END, f"Error during auto-detection: {e}\n\n")
            self.xgb_info_text.insert(tk.END, "Please try loading a model manually.")
            self.xgb_info_text.config(state=tk.DISABLED)
            print("=== AUTO-DETECTION FAILED ===")

def main():
    root = tk.Tk()
    app = OpenFieldApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 