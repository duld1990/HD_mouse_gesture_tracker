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

def remove_tracking_outliers(df, max_jump_distance=50.0, window_size=5):
    """
    Remove tracking outliers where points jump to faraway locations.
    
    Args:
        df: DataFrame with tracking coordinates
        max_jump_distance: Maximum allowed distance between consecutive frames (pixels)
        window_size: Number of frames to use for median filtering
    
    Returns:
        DataFrame with outliers replaced by NaN
    """
    df_clean = df.copy()
    
    # Columns to check for outliers
    coord_columns = [col for col in df.columns if col.endswith(('_x', '_y'))]
    
    for col in coord_columns:
        if col in df.columns:
            # Calculate distances between consecutive frames
            coords = df[col].values
            distances = np.abs(np.diff(coords))
            
            # Find jumps that exceed threshold
            outlier_mask = np.zeros_like(coords, dtype=bool)
            outlier_mask[1:] = distances > max_jump_distance
            
            # Also check for isolated valid points surrounded by NaN
            for i in range(1, len(coords) - 1):
                if not np.isnan(coords[i]) and (np.isnan(coords[i-1]) or np.isnan(coords[i+1])):
                    # Check if this point is far from its neighbors
                    if not np.isnan(coords[i-1]) and abs(coords[i] - coords[i-1]) > max_jump_distance:
                        outlier_mask[i] = True
                    if not np.isnan(coords[i+1]) and abs(coords[i] - coords[i+1]) > max_jump_distance:
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
        self.heatmap_cmap = tk.StringVar(value='hot')
        self.heatmap_vmin = tk.StringVar(value='')
        self.heatmap_vmax = tk.StringVar(value='')
        self.max_jump_distance = tk.DoubleVar(value=50.0)

        # GUI setup
        self.notebook = ttk.Notebook(root)
        self.frame_analysis = tk.Frame(self.notebook)
        self.frame_help = tk.Frame(self.notebook)
        self.notebook.add(self.frame_analysis, text='Analysis')
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

        # Help Tab
        help_text = (
            'Required CSV columns:\n'
            'body_center_x, body_center_y, tail_base_x, tail_base_y, forelimbs_x, forelimbs_y, head_x, head_y\n\n'
            'Each row should represent a video frame.\n'
            'Missing or bad data should be left blank or as NaN.\n'
            'Sampling rate is the video frame rate (e.g., 30 for 30 fps).\n'
            'Arena width/height should be in centimeters.\n'
            'Max jump distance: Maximum allowed distance between consecutive frames (pixels).\n'
            '  Points that jump further than this will be marked as NaN.\n\n'
            'Heatmap options:\n'
            '- Color scheme: Choose from hot, viridis, plasma, inferno, magma, cividis, jet. Default is "hot".\n'
            '- Colorbar min/max: Leave blank to auto-scale (default behavior).\n'
            '  To reset to default, simply clear these fields.\n\n'
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

    def run_analysis(self):
        csv_path = self.csv_path.get()
        output_dir = self.output_dir.get()
        if not csv_path or not output_dir:
            messagebox.showerror('Error', 'Please select both a CSV file and an output folder.')
            return
        try:
            max_jump_distance = self.max_jump_distance.get()
            df_full, df_valid, valid_mask = load_and_clean_csv(csv_path, max_jump_distance)
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

    def save_analysis_outputs(self, df_full, df_valid, valid_mask, sampling_rate, arena_width, arena_height, output_dir):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        import os
        
        total_frames = len(df_full)
        
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
            if np.abs(angle) > 90:
                head_angle[i] = np.nan
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
        
        # Stand behavior
        head_body_dist = np.sqrt((nose_x - body_x)**2 + (nose_y - body_y)**2)
        forelimb_body_dist = np.sqrt((forelimb_x - body_x)**2 + (forelimb_y - body_y)**2)
        head_forelimb_dist = np.sqrt((nose_x - forelimb_x)**2 + (nose_y - forelimb_y)**2)
        stand_dist_thresh = 25.0
        stand_multi = (head_body_dist < stand_dist_thresh) & (forelimb_body_dist < stand_dist_thresh) & (head_forelimb_dist < stand_dist_thresh)
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
        plt.title('Distribution of Head Turn Angles by Time')
        plt.tight_layout()
        fig_hist.savefig(os.path.join(output_dir, 'head_angle_histogram.png'))
        plt.close(fig_hist)

def main():
    root = tk.Tk()
    app = OpenFieldApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 