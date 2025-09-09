# -*- coding: utf-8 -*-
"""
Animal tracking based on YOLOv8+
Tracks 4 body landmarks (body_center, head, fore_limbs, tail_base) and exports CSV and .npz for transformer training.
UI: Import Video → Select ROI → Select Output Directory → Analyze
by Lida Du
ldu13@jh.edu
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO
import PIL.Image, PIL.ImageTk
import tkinter.ttk as ttk
import seaborn as sns
import importlib.util
import types
import torch
import torch.nn as nn
import datetime

def print_imported_modules():
    print("\nImported modules and their file paths:")
    for name, module in sorted(sys.modules.items()):
        if isinstance(module, types.ModuleType):
            try:
                file = getattr(module, '__file__', None)
                if file:
                    print(f"{name}: {file}")
            except Exception:
                pass
    print("\n--- End of imported modules list ---\n")

print_imported_modules()

def select_roi_opencv(frame_gray):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(frame_gray, cmap='gray')
    ax.set_title('Select 4 ROI corners (clockwise or counterclockwise)')
    points = plt.ginput(4, timeout=0)  # Wait for 4 clicks
    plt.close(fig)
    if len(points) < 4:
        raise RuntimeError("ROI selection cancelled or not enough points selected.")
    xs, ys = zip(*points)
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

class App:
    def __init__(self, root):
        self.root = root
        root.title('YOLO Mouse Tracker')
        root.geometry('900x700')
        self.video_path = None
        self.roi = None
        self.output_dir = None
        self.live = tk.BooleanVar(value=False)
        self.abort_flag = threading.Event()
        self.user_fps_var = tk.DoubleVar(value=30)
        self.user_fps = 30  # Default FPS
        # Frame skip and batch size
        self.frame_skip = tk.IntVar(value=1)
        self.batch_size = tk.IntVar(value=1)
        self.selected_behavior = tk.StringVar(value='Not selected')
        self.behavior_classes = [
            'freeze', 'walk', 'rotate', 'jump', 'head_turn',
            'stand_rearing', 'stand_lean', 'stand_grooming'
        ]
        # Custom behavior map: stand_rearing=1, stand_lean=2, stand_grooming=3, others get unique values
        self.behavior_map = {
            'freeze': 0,
            'stand_rearing': 1,
            'stand_lean': 2,
            'stand_grooming': 3,
            'walk': 4,
            'rotate': 5,
            'jump': 6,
            'head_turn': 7
        }
        # Use ttk.Notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.main_frame = tk.Frame(self.notebook)
        self.behav_frame = tk.Frame(self.notebook)
        self.advanced_frame = tk.Frame(self.notebook)
        self.help_frame = tk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text='Animal Tracking')
        self.notebook.add(self.behav_frame, text='Behavior Analysis')
        self.notebook.add(self.advanced_frame, text='Advanced')
        self.notebook.add(self.help_frame, text='Help')
        self.notebook.pack(fill='both', expand=True)
        # Move all main UI to self.main_frame instead of root
        main_parent = self.main_frame
        # Backend dropdown
        backend_frame = tk.Frame(self.advanced_frame)
        tk.Label(backend_frame, text='Backend:').pack(side='left')
        self.backend = tk.StringVar(value='PyTorch')
        self.backend_dropdown = tk.OptionMenu(backend_frame, self.backend, 'PyTorch', 'ONNX', 'OpenVINO', command=self.on_backend_change)
        self.backend_dropdown.pack(side='left')
        self.model_path = tk.StringVar(value='')
        self.btn_select_model = tk.Button(backend_frame, text='Select Model File/Dir', command=self.select_model_file, state='disabled')
        self.btn_select_model.pack(side='left', padx=5)
        backend_frame.pack(fill='x')
        # Model size dropdown and imgsz entry
        model_frame = tk.Frame(self.advanced_frame)
        tk.Label(model_frame, text='YOLO Model:').pack(side='left')
        self.model_size = tk.StringVar(value='Default')
        self.model_dropdown = tk.OptionMenu(model_frame, self.model_size, 'Default', 'nano', 'small', 'medium', 'Custom', command=self.on_model_change)
        self.model_dropdown.pack(side='left')
        self.custom_model_path = tk.StringVar(value='')
        # On startup, search for best_yolov11n.pt in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(script_dir, 'best_yolov11n.pt')
        if os.path.exists(default_model_path):
            self.default_model_path = default_model_path
        else:
            # fallback to best.pt if best_yolov11n.pt does not exist
            fallback_path = os.path.join(script_dir, 'best.pt')
            if os.path.exists(fallback_path):
                self.default_model_path = fallback_path
            else:
                self.default_model_path = ''
        self.btn_select_custom_model = tk.Button(model_frame, text='Select .pt File', command=self.select_custom_model, state='disabled')
        self.btn_select_custom_model.pack(side='left', padx=5)
        tk.Label(model_frame, text='Input Size (imgsz):').pack(side='left', padx=(10,0))
        self.imgsz = tk.IntVar(value=640)
        tk.Entry(model_frame, textvariable=self.imgsz, width=6).pack(side='left')
        # Add FPS entry to Advanced tab
        tk.Label(model_frame, text='Video FPS:').pack(side='left', padx=(10,0))
        fps_entry = tk.Entry(model_frame, textvariable=self.user_fps_var, width=6)
        fps_entry.pack(side='left')
        def on_fps_change(*args):
            try:
                val = float(self.user_fps_var.get())
                if val <= 0:
                    raise ValueError
                self.user_fps = val
            except Exception:
                self.user_fps = 30
        self.user_fps_var.trace_add('write', on_fps_change)
        model_frame.pack(fill='x')
        # imgsz recommendation note
        imgsz_note = (
            'Recommended imgsz (input size): 320 (fastest), 416 (fast), 512 (good detail), 640 (default, slowest).\n'
            'Lower values = faster, higher values = more detail. For CPU, try 320 or 416 for most animal tracking.'
        )
        tk.Label(self.advanced_frame, text=imgsz_note, justify='left', fg='blue', font=('Arial', 10, 'italic')).pack(anchor='w', padx=10, pady=(0,10))
        # Add Export Model button
        export_frame = tk.Frame(self.advanced_frame)
        self.btn_export_model = tk.Button(export_frame, text='Export Model', command=self.export_model)
        self.btn_export_model.pack(side='left', padx=5)
        # Add Export Model Help button
        self.btn_export_model_help = tk.Button(export_frame, text='Export Model Help', command=self.export_model_help)
        self.btn_export_model_help.pack(side='left', padx=5)
        export_frame.pack(fill='x')
        # Main tab UI (basic workflow)
        tk.Label(main_parent, text='Tip: Disable live preview for maximum speed.', fg='red', font=('Arial', 10, 'italic')).pack(fill='x')
        self.btn_import = tk.Button(main_parent, text='Import Video', command=self.import_video)
        self.btn_import.pack(fill='x')
        self.btn_roi = tk.Button(main_parent, text='Select ROI', command=self.crop_roi, state='disabled')
        self.btn_roi.pack(fill='x')
        self.btn_output = tk.Button(main_parent, text='Select Output Directory', command=self.select_output_dir)
        self.btn_output.pack(fill='x')
        self.chk_live = tk.Checkbutton(main_parent, text='Enable Live Preview', variable=self.live)
        self.chk_live.pack(fill='x')
        # Frame skip and batch size controls
        frame_opts = tk.Frame(main_parent)
        tk.Label(frame_opts, text='Frame Skip:').pack(side='left')
        tk.Entry(frame_opts, textvariable=self.frame_skip, width=5).pack(side='left')
        tk.Label(frame_opts, text='Batch Size:').pack(side='left')
        tk.Entry(frame_opts, textvariable=self.batch_size, width=5).pack(side='left')
        frame_opts.pack(fill='x')
        # Time range selection (scales and entries)
        self.scale_frame = tk.Frame(main_parent)
        self.start_frame = tk.IntVar(value=0)
        self.end_frame = tk.IntVar(value=0)
        # --- New: Mode selection for end frame ---
        self.end_mode = tk.StringVar(value='frame')  # 'frame' or 'duration'
        mode_frame = tk.Frame(self.scale_frame)
        tk.Label(mode_frame, text='End selection:').pack(side='left')
        tk.Radiobutton(mode_frame, text='End Frame', variable=self.end_mode, value='frame', command=self.on_end_mode_change).pack(side='left')
        tk.Radiobutton(mode_frame, text='Duration (min:sec)', variable=self.end_mode, value='duration', command=self.on_end_mode_change).pack(side='left')
        mode_frame.pack(side='top', fill='x', padx=5, pady=(0,2))
        # --- End mode selection ---
        # Duration input fields (hidden by default)
        self.duration_min = tk.IntVar(value=0)
        self.duration_sec = tk.IntVar(value=0)
        self.duration_frame = tk.Frame(self.scale_frame)
        tk.Label(self.duration_frame, text='Duration:').pack(side='left')
        tk.Entry(self.duration_frame, textvariable=self.duration_min, width=4).pack(side='left')
        tk.Label(self.duration_frame, text='min').pack(side='left')
        tk.Entry(self.duration_frame, textvariable=self.duration_sec, width=4).pack(side='left')
        tk.Label(self.duration_frame, text='sec').pack(side='left')
        # Hide by default
        self.duration_frame.pack_forget()
        # --- End duration input fields ---
        self.scale_start = tk.Scale(self.scale_frame, from_=0, to=0, orient='horizontal', label='Start Frame', variable=self.start_frame, command=self.update_time_labels)
        self.scale_end = tk.Scale(self.scale_frame, from_=0, to=0, orient='horizontal', label='End Frame', variable=self.end_frame, command=self.update_time_labels)
        self.scale_start.pack(side='left', fill='x', expand=True, padx=5)
        self.scale_end.pack(side='left', fill='x', expand=True, padx=5)
        # Frame entry boxes
        entry_frame = tk.Frame(self.scale_frame)
        tk.Label(entry_frame, text='Start Frame:').pack(side='left')
        self.entry_start = tk.Entry(entry_frame, textvariable=self.start_frame, width=7)
        self.entry_start.pack(side='left')
        tk.Label(entry_frame, text='End Frame:').pack(side='left')
        self.entry_end = tk.Entry(entry_frame, textvariable=self.end_frame, width=7)
        self.entry_end.pack(side='left')
        entry_frame.pack(side='left', padx=10)
        # Bind update_time_labels to manual entry changes
        def on_entry_change(*args):
            self.update_time_labels()
        self.start_frame.trace_add('write', on_entry_change)
        self.end_frame.trace_add('write', on_entry_change)
        self.duration_min.trace_add('write', self.on_duration_change)
        self.duration_sec.trace_add('write', self.on_duration_change)
        self.scale_frame.pack(fill='x')
        self.time_label = tk.Label(main_parent, text='')
        self.time_label.pack(fill='x')
        self.progress_track = ttk.Progressbar(main_parent, orient='horizontal', length=400, mode='determinate', style='Track.Horizontal.TProgressbar')
        tk.Label(main_parent, text='Analysis Progress').pack(fill='x')
        self.progress_track.pack(fill='x', padx=5, pady=2)
        # Add behavior dropdown to Main tab (above Start Analysis)
        behavior_frame = tk.Frame(self.main_frame)
        tk.Label(behavior_frame, text='Behavior Class:').pack(side='left')
        self.behavior_menu = tk.OptionMenu(behavior_frame, self.selected_behavior, 'Not selected', *self.behavior_classes)
        self.behavior_menu.pack(side='left')
        behavior_frame.pack(fill='x', pady=(5,0))
        # Add entry and buttons for add/remove
        add_remove_frame = tk.Frame(self.main_frame)
        self.new_behavior_var = tk.StringVar()
        tk.Entry(add_remove_frame, textvariable=self.new_behavior_var, width=12).pack(side='left', padx=(0,5))
        def add_behavior():
            new_b = self.new_behavior_var.get().strip()
            if new_b and new_b not in self.behavior_classes and new_b != 'Not selected':
                self.behavior_classes.append(new_b)
                menu = self.behavior_menu['menu']
                menu.add_command(label=new_b, command=tk._setit(self.selected_behavior, new_b))
                self.new_behavior_var.set('')
        def remove_behavior():
            sel = self.selected_behavior.get()
            if sel != 'Not selected' and sel in self.behavior_classes:
                self.behavior_classes.remove(sel)
                # Rebuild menu
                menu = self.behavior_menu['menu']
                menu.delete(0, 'end')
                menu.add_command(label='Not selected', command=tk._setit(self.selected_behavior, 'Not selected'))
                for b in self.behavior_classes:
                    menu.add_command(label=b, command=tk._setit(self.selected_behavior, b))
                self.selected_behavior.set('Not selected')
        add_btn = tk.Button(add_remove_frame, text='Add', command=add_behavior)
        add_btn.pack(side='left', padx=(0,5))
        remove_btn = tk.Button(add_remove_frame, text='Remove', command=remove_behavior)
        remove_btn.pack(side='left')
        add_remove_frame.pack(fill='x', pady=(0,5))
        btn_frame = tk.Frame(main_parent)
        self.btn_start = tk.Button(btn_frame, text='Start Analysis', command=self.start_analysis, state='disabled')
        self.btn_start.pack(side='left', fill='x', expand=True)
        self.btn_abort = tk.Button(btn_frame, text='Abort', command=self.abort_analysis, state='disabled')
        self.btn_abort.pack(side='left', fill='x', expand=True)
        btn_frame.pack(fill='x')
        self.status = tk.Label(main_parent, text='No video loaded.')
        self.status.pack(fill='x')
        self.preview_label = tk.Label(main_parent)
        self.preview_label.pack(fill='both', expand=True)
        # Add legend label below preview
        legend_text = (
            'body_center: ',
            'tail_base: ',
            'forelimbs: ',
            'head: '
        )
        legend_colors = ['#0033cc', '#ff7f7f', '#ffb6c1', '#ff6666']
        self.legend_frame = tk.Frame(main_parent)
        for i, (txt, color) in enumerate(zip(legend_text, legend_colors)):
            tk.Label(self.legend_frame, text=txt, fg=color, font=('Arial', 12, 'bold')).pack(side='left', padx=5)
        self.legend_frame.pack(fill='x')
        self.btn_restart = tk.Button(main_parent, text='Restart App', command=self.restart_app)
        self.btn_restart.pack(fill='x')
        # Help tab content
        # Step-by-step workflow for main tracking function
        help_text = (
            'Step-by-step Workflow for Mouse Tracking:\n'
            '1. Import your video using the "Import Video" button.\n'
            '2. Select the region of interest (ROI) using "Select ROI".\n'
            '3. Choose the output directory for results.\n'
            '4. (Optional) Adjust frame range, frame skip, batch size, and other advanced settings.\n'
            '5. (Optional) Check the "Live preview" box if a preview video is needed.\n'
            '6. Click "Start Analysis" to begin tracking.\n'
            '7. Wait for analysis to complete. Results will be saved in the output directory.\n'
            '8. Review the output files.\n'
            '\n'
            'This app is developed and optimized by Lida Du, Wendy Shi, Pennie Lu\n'
            'Johns Hopkins Medical Institutes 2025'
        )
        tk.Label(self.help_frame, text=help_text, justify='left', fg='black', font=('Arial', 12), anchor='nw').pack(anchor='nw', padx=10, pady=10, fill='x')

        # Advanced Help button and info
        def show_advanced_help():
            adv_text = (
                'This tracker is based on the YOLO v11 nano model, suitable for integrated GPUs.\n'
                'For more information about YOLO v11, visit: https://docs.ultralytics.com/models/yolo11/\n'
                '\n'
                'How to increase analysis speed:\n'
                '(1) Use a smaller YOLO model (nano or small)\n'
                '(2) Increase Frame Skip (process every Nth frame)\n'
                '(3) Lower Input Size (imgsz)\n'
                '(4) Increase Batch Size (if RAM allows)\n'
                '(5) Disable Live Preview\n'
                '(6) Close other applications to free up CPU/RAM\n'
                '(7) For advanced users: try ONNX or OpenVINO for CPU acceleration (select backend and model in Advanced tab)\n'
                '\n'
                'Tip: The above options are available in the Advanced tab.'
            )
            from tkinter import messagebox
            messagebox.showinfo('Advanced Help', adv_text)

        adv_btn = tk.Button(self.help_frame, text='Advanced Help', command=show_advanced_help)
        adv_btn.pack(anchor='nw', padx=10, pady=10)
        
        # --- Behavior Analysis Tab ---
        self.behav_model_path = tk.StringVar(value='')
        # Create and pack behav_status label before using it
        self.behav_status = tk.Label(self.behav_frame, text='', fg='blue', font=('Arial', 12, 'bold'))
        self.behav_status.pack(anchor='w', padx=10, pady=(10,0))
        # Add checkbox to enable/disable transformer model
        self.use_transformer_model = tk.BooleanVar(value=True)
        self.chk_use_transformer = tk.Checkbutton(self.behav_frame, text='Enable Transformer Model for Behavior Detection', variable=self.use_transformer_model)
        self.chk_use_transformer.pack(anchor='w', padx=10, pady=(0,5))
        # --- Auto-load transformer model if present in script directory ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        auto_model = None
        for f in os.listdir(script_dir):
            if f.endswith('.pth'):
                auto_model = os.path.join(script_dir, f)
                break
        if auto_model:
            self.behav_model_path.set(auto_model)
            self.behav_status.config(text=f'Auto-loaded: {os.path.basename(auto_model)} (from script directory)')
            tk.messagebox.showinfo('Transformer Model Loaded', f'Automatically loaded transformer model: {os.path.basename(auto_model)}\nThis model will be used for behavior detection unless you select another .pth file.')
        else:
            self.behav_status.config(text='No behavior model loaded.')
        tk.Label(self.behav_frame, text='Select a Transformer Model (*.pth) for behavior class detection').pack(anchor='w', padx=10, pady=(10,0))
        behav_model_entry = tk.Entry(self.behav_frame, textvariable=self.behav_model_path, width=60)
        behav_model_entry.pack(anchor='w', padx=10)
        tk.Button(self.behav_frame, text='Select Model', command=self.select_behav_model).pack(anchor='w', padx=10, pady=(0,10))
        self.behav_result = tk.Label(self.behav_frame, text='', fg='blue', font=('Arial', 12, 'bold'))
        self.behav_result.pack(anchor='w', padx=10, pady=(10,10))
        
    def import_video(self):
        path = filedialog.askopenfilename(filetypes=[('Video','*.mp4;*.avi;*.mov;*.mkv')])
        if path:
            self.video_path = path
            self.status.config(text=f'Loaded video: {os.path.basename(path)}')
            self.btn_roi.config(state='normal')  # enable ROI select
            # Set up time range sliders
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.scale_start.config(from_=0, to=total_frames-1)
            self.scale_end.config(from_=0, to=total_frames-1)
            self.start_frame.set(0)
            self.end_frame.set(total_frames-1)
            self.update_time_labels()

    def select_roi_canvas(self, frame_gray):
        import PIL.Image, PIL.ImageTk
        roi_points = []
        roi_window = tk.Toplevel(self.root)
        roi_window.title('Select ROI - click 4 corners')
        h, w = frame_gray.shape
        scale = min(800/w, 600/h, 1.0)
        disp_w, disp_h = int(w*scale), int(h*scale)
        img = PIL.Image.fromarray(frame_gray)
        img = img.resize((disp_w, disp_h), PIL.Image.LANCZOS)
        tk_img = PIL.ImageTk.PhotoImage(img)
        canvas = tk.Canvas(roi_window, width=disp_w, height=disp_h)
        canvas.pack()
        canvas.create_image(0, 0, anchor='nw', image=tk_img)
        marker_ids = []
        poly_id = None
        def on_click(event):
            nonlocal poly_id
            if len(roi_points) < 4:
                x, y = int(event.x/scale), int(event.y/scale)
                roi_points.append((x, y))
                marker_ids.append(canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill='red'))
                if len(roi_points) == 4:
                    # Draw polygon
                    pts = [((x*scale), (y*scale)) for (x, y) in roi_points]
                    poly_id = canvas.create_polygon(pts, outline='yellow', fill='', width=2)
        canvas.bind('<Button-1>', on_click)
        def on_done():
            if len(roi_points) == 4:
                roi_window.destroy()
            else:
                messagebox.showwarning('ROI', 'Please select 4 points.')
        btn = tk.Button(roi_window, text='Done', command=on_done)
        btn.pack()
        roi_window.transient(self.root)
        roi_window.grab_set()
        self.root.wait_window(roi_window)
        if len(roi_points) < 4:
            raise RuntimeError('ROI selection cancelled or not enough points selected.')
        xs, ys = zip(*roi_points)
        return min(xs), min(ys), max(xs), max(ys)

    def crop_roi(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror('Error', 'Cannot read video frame.')
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("Frame shape:", gray.shape, "dtype:", gray.dtype, "min:", gray.min(), "max:", gray.max())
        try:
            self.roi = self.select_roi_canvas(gray)
        except Exception as e:
            messagebox.showerror('ROI Error', str(e))
            return
        self.status.config(text=f'ROI set to {self.roi}')
        self.update_start_button()

    def select_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir = path
            self.status.config(text=f'Output directory: {path}')
            self.update_start_button()

    def update_start_button(self):
        # enable Start only if video, ROI, and output_dir are set
        if self.video_path and self.roi and self.output_dir:
            self.btn_start.config(state='normal')
        else:
            self.btn_start.config(state='disabled')

    def start_analysis(self):
        if not (self.video_path and self.roi and self.output_dir):
            messagebox.showwarning('Missing', 'Please set video, ROI, and output directory.')
            return
        self.status.config(text='Analyzing...')
        self.abort_flag.clear()
        self.btn_start.config(state='disabled')
        self.btn_abort.config(state='normal')
        thread = threading.Thread(target=self.run_tracking)
        thread.start()

    def run_tracking(self):
        import time
        import matplotlib.pyplot as plt
        import seaborn as sns
        # Model selection (now from advanced tab)
        backend = self.backend.get()
        model_file = None
        try:
            if backend == 'ONNX':
                model_file = self.model_path.get() or 'yolov8n.onnx'
            elif backend == 'OpenVINO':
                model_file = self.model_path.get() or 'yolov8n_openvino_model/'
                # Check for .xml and .bin files
                if not (os.path.isdir(model_file) and any(f.endswith('.xml') for f in os.listdir(model_file)) and any(f.endswith('.bin') for f in os.listdir(model_file))):
                    tk.messagebox.showerror('OpenVINO Model Error', 'Selected directory does not contain .xml and .bin files required for OpenVINO model.')
                    self.root.after(0, lambda: self.status.config(text='Error'))
                    self.root.after(0, lambda: self.btn_abort.config(state='disabled'))
                    self.root.after(0, lambda: self.btn_start.config(state='normal'))
                    self.root.after(0, lambda: self.btn_import.config(state='normal'))
                    self.root.after(0, lambda: self.btn_roi.config(state='normal'))
                    self.root.after(0, lambda: self.btn_output.config(state='normal'))
                    self.abort_flag.clear()
                    return
            else:
                if self.model_size.get() == 'Default':
                    model_file = self.default_model_path or 'best_yolov11n.pt'
                elif self.model_size.get() == 'Custom':
                    model_file = self.custom_model_path.get()
                    if not model_file:
                        tk.messagebox.showerror('Custom Model Error', 'No custom .pt file selected.')
                        self.root.after(0, lambda: self.status.config(text='Error'))
                        self.root.after(0, lambda: self.btn_abort.config(state='disabled'))
                        self.root.after(0, lambda: self.btn_start.config(state='normal'))
                        self.root.after(0, lambda: self.btn_import.config(state='normal'))
                        self.root.after(0, lambda: self.btn_roi.config(state='normal'))
                        self.root.after(0, lambda: self.btn_output.config(state='normal'))
                        self.abort_flag.clear()
                        return
                else:
                    model_map = {'nano': 'yolov8n.pt', 'small': 'yolov8s.pt', 'medium': 'yolov8m.pt'}
                    model_file = model_map.get(self.model_size.get(), 'yolov8n.pt')
            imgsz_val = int(self.imgsz.get())
            print(f'Loading model: {model_file} (imgsz={imgsz_val}, backend={backend})')
            model = YOLO(model_file)
            # Use GPU if available and backend is PyTorch
            if backend == 'PyTorch' and hasattr(model, 'to'):
                try:
                    model.to('cuda')
                except Exception:
                    pass
        except Exception as e:
            import traceback
            tk.messagebox.showerror('Model Load Error', f'Error loading model: {e}\n{traceback.format_exc()}')
            self.root.after(0, lambda: self.status.config(text='Error'))
            self.root.after(0, lambda: self.btn_abort.config(state='disabled'))
            self.root.after(0, lambda: self.btn_start.config(state='normal'))
            self.root.after(0, lambda: self.btn_import.config(state='normal'))
            self.root.after(0, lambda: self.btn_roi.config(state='normal'))
            self.root.after(0, lambda: self.btn_output.config(state='normal'))
            self.abort_flag.clear()
            return
        self.cap = cv2.VideoCapture(self.video_path)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.user_fps if self.user_fps else 30
        interval = max(1, int(self.frame_skip.get()))
        batch_size = max(1, int(self.batch_size.get()))
        x0, y0, x1, y1 = self.roi
        # Get selected range and folder name early
        start_idx = self.start_frame.get()
        end_idx = self.end_frame.get()
        t1 = start_idx / fps if fps else 0
        t2 = end_idx / fps if fps else 0
        def mmss(t):
            mm = int(t // 60)
            ss = int(t % 60)
            return f"{mm:02d}{ss:02d}"
        start_str = mmss(t1)
        end_str = mmss(t2)
        import re
        tracks_folder = os.path.join(self.output_dir, f"tracks_{start_str}_{end_str}")
        os.makedirs(tracks_folder, exist_ok=True)
        # Update tracked parts and correct order
        part_names = ['body_center', 'tail_base', 'forelimbs', 'head']
        tracks = {name: [] for name in part_names}
        self.writer = None
        show_preview = self.live.get()
        if show_preview:
            os.makedirs(tracks_folder, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            w, h = x1-x0, y1-y0
            preview_path = os.path.join(tracks_folder, 'preview.mp4')
            self.writer = cv2.VideoWriter(preview_path, fourcc, 60, (w, h))
            print(f'Recording live preview to {preview_path}')
        # Get selected range
        start_idx = self.start_frame.get()
        end_idx = self.end_frame.get()
        num_frames = end_idx - start_idx + 1
        self.root.after(0, lambda: self.progress_track.config(value=0, maximum=num_frames//interval))
        idx = 0
        batch = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        pose_window = []
        walk_predictions = []
        walk_track = []
        behavior_predictions = []  # Store predicted behavior index for each frame
        behavior_names = ['freeze', 'walk', 'rotate', 'jump', 'head_turn', 'stand_rearing', 'stand_lean', 'stand_grooming']
        for i in range(start_idx, end_idx+1):
            if self.abort_flag.is_set():
                self.cleanup_resources()
                self.root.after(0, lambda: self.status.config(text='Aborted'))
                self.root.after(0, lambda: self.btn_abort.config(state='disabled'))
                self.root.after(0, lambda: self.btn_start.config(state='normal'))
                self.root.after(0, lambda: self.btn_import.config(state='normal'))
                self.root.after(0, lambda: self.btn_roi.config(state='normal'))
                self.root.after(0, lambda: self.btn_output.config(state='normal'))
                self.root.after(0, lambda: self.btn_restart.config(state='disabled'))
                self.abort_flag.clear()
                return
            ret, frame = self.cap.read()
            if not ret:
                break
            if (i-start_idx) % interval != 0:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crop = gray[y0:y1, x0:x1]
            feed = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            batch.append(feed)
            if len(batch) == batch_size or i == end_idx:
                # Batch inference if possible
                try:
                    results = model(batch, imgsz=imgsz_val, conf=0.25, verbose=False)
                except Exception:
                    results = [model(feed, imgsz=imgsz_val, conf=0.25, verbose=False)[0] for feed in batch]
                for bidx, res in enumerate(results):
                    best = {0:None,1:None,2:None,3:None}
                    for box in res.boxes:
                        cls = int(box.cls)
                        if cls in best:
                            c = float(box.conf)
                            x1b,y1b,x2b,y2b = box.xyxy[0].cpu().numpy()
                            cx = (x1b + x2b) / 2
                            cy = (y1b + y2b) / 2
                            if best[cls] is None or c > best[cls][0]:
                                best[cls] = (c, (cx, cy))
                    # Save tracks relative to cropped ROI (no offset needed)
                    for i, name in enumerate(part_names):
                        tracks[name].append(best[i][1] if best[i] else (np.nan, np.nan))
                    # --- Behavior analysis: build pose vector for this frame ---
                    pose_vec = []
                    for name in part_names:
                        pt = tracks[name][-1]
                        pose_vec.extend(pt if pt is not None else (np.nan, np.nan))
                    # Compute kinematic features for this frame (use previous frames if available)
                    # Only if at least 2 body_center points are available
                    if len(tracks['body_center']) > 1:
                        idx = len(tracks['body_center']) - 1
                        # Speed
                        prev = np.array(tracks['body_center'][idx-1])
                        curr = np.array(tracks['body_center'][idx])
                        speed = np.linalg.norm(curr - prev)
                        # Acceleration
                        if len(tracks['body_center']) > 2:
                            prev2 = np.array(tracks['body_center'][idx-2])
                            prev_speed = np.linalg.norm(prev - prev2)
                            accel = speed - prev_speed
                        else:
                            accel = 0.0
                        # Head vs. body angle
                        curr_head = np.array(tracks['head'][idx])
                        curr_tail = np.array(tracks['tail_base'][idx])
                        main_axis = curr_tail - curr
                        head_vec = curr_head - curr
                        norm1 = np.linalg.norm(main_axis)
                        norm2 = np.linalg.norm(head_vec)
                        if norm1 > 0 and norm2 > 0:
                            dot = np.dot(main_axis, head_vec)
                            cross = main_axis[0]*head_vec[1] - main_axis[1]*head_vec[0]
                            head_body_angle = np.arctan2(cross, dot)
                        else:
                            head_body_angle = np.nan
                        # Angular speed
                        if len(tracks['body_center']) > 2:
                            prev_head = np.array(tracks['head'][idx-1])
                            prev_tail = np.array(tracks['tail_base'][idx-1])
                            prev_main_axis = prev_tail - prev
                            prev_head_vec = prev_head - prev
                            norm1p = np.linalg.norm(prev_main_axis)
                            norm2p = np.linalg.norm(prev_head_vec)
                            if norm1p > 0 and norm2p > 0:
                                dotp = np.dot(prev_main_axis, prev_head_vec)
                                crossp = prev_main_axis[0]*prev_head_vec[1] - prev_main_axis[1]*prev_head_vec[0]
                                prev_angle = np.arctan2(crossp, dotp)
                                angular_speed = head_body_angle - prev_angle
                            else:
                                angular_speed = 0.0
                        else:
                            angular_speed = 0.0
                    else:
                        speed = 0.0
                        accel = 0.0
                        angular_speed = 0.0
                        head_body_angle = 0.0
                    pose_vec.extend([speed, accel, angular_speed, head_body_angle])
                    pose_window.append(pose_vec)
                    if len(pose_window) > 32:
                        pose_window.pop(0)
                    # Run transformer inference if window is full and no NaNs
                    if self.use_transformer_model.get() and self.behav_model_path.get() and os.path.exists(self.behav_model_path.get()):
                        class SimpleTransformerClassifier(nn.Module):
                            def __init__(self, input_dim, num_classes, window_size, d_model=64, nhead=4, num_layers=2):
                                super().__init__()
                                self.input_proj = nn.Linear(input_dim, d_model)
                                encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
                                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                                self.classifier = nn.Linear(d_model, num_classes)
                            def forward(self, x):
                                x = self.input_proj(x)
                                x = self.transformer(x)
                                x = x[:, -1, :]
                                return self.classifier(x)
                        transformer = SimpleTransformerClassifier(12, 8, 32)
                        transformer.load_state_dict(torch.load(self.behav_model_path.get(), map_location='cpu'))
                        transformer.eval()
                        if len(pose_window) == 32 and np.isfinite(pose_window).all():
                            pose_arr = np.array(pose_window, dtype=np.float32).reshape(1, 32, 12)
                            with torch.no_grad():
                                logits = transformer(torch.tensor(pose_arr))
                                pred = logits.argmax(dim=1).item()
                            walk_predictions.append(pred == 4)
                            behavior_predictions.append(pred)
                            if pred == 4:
                                walk_track.append(tracks['body_center'][-1])
                        else:
                            behavior_predictions.append(None)
                    else:
                        behavior_predictions.append(None)
                    if show_preview:
                        vis = feed.copy()
                        # Draw all tracked spots with different colors
                        for i, name in enumerate(part_names):
                            val = best[i]
                            if val:
                                _, (cx, cy) = val
                                # Use the correct BGR for the legend colors (swapped for head and forelimbs)
                                if name == 'body_center':
                                    color = (204,51,51) # deep blue (BGR)
                                elif name == 'tail_base':
                                    color = (127,127,255) # light coral (BGR)
                                elif name == 'forelimbs':
                                    color = (182,182,255) # light pink (BGR)
                                elif name == 'head':
                                    color = (102,102,255) # light red (BGR)
                                else:
                                    color = (0,0,0)
                                cv2.circle(vis, (int(cx), int(cy)), 5, color, -1)
                        # --- Overlay detected behavior label in small font ---
                        if self.use_transformer_model.get() and len(behavior_predictions) > 0 and behavior_predictions[-1] is not None:
                            behav_idx = behavior_predictions[-1]
                            # Display 'stand' for any stand subclass
                            if behav_idx in [1, 2, 3]:
                                behav_label = 'stand'
                            else:
                                behav_label = behavior_names[behav_idx] if behav_idx < len(behavior_names) else str(behav_idx)
                            cv2.putText(vis, behav_label, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
                        self.root.after(0, self.update_preview_img, vis)
                        if self.writer:
                            self.writer.write(vis)
                    elif self.writer:
                        self.writer.write(feed)
                    idx += 1
                    self.root.after(0, lambda v=idx: self.progress_track.config(value=v))
                batch = []
        self.cleanup_resources()
        # Save tracks in .csv and .npz for transformer
        os.makedirs(self.output_dir, exist_ok=True)
        data_arrays = [np.array(tracks[n]) for n in part_names]
        pose_seq = np.hstack(data_arrays)
        # --- Custom output file naming ---
        video_base = os.path.splitext(os.path.basename(self.video_path))[0]
        # Use user-set FPS for time calculation
        fps = self.user_fps if hasattr(self, 'user_fps') and self.user_fps else 30
        start_idx = self.start_frame.get()
        end_idx = self.end_frame.get()
        t1 = start_idx / fps if fps else 0
        t2 = end_idx / fps if fps else 0
        def mmss(t):
            mm = int(t // 60)
            ss = int(t % 60)
            return f"{mm:02d}{ss:02d}"
        start_str = mmss(t1)
        end_str = mmss(t2)
        # Remove any non-alphanumeric from video_base for file safety
        video_base_safe = re.sub(r'[^A-Za-z0-9_\-]', '_', video_base)
        # Use behavior class if selected, else video name
        behavior = self.selected_behavior.get()
        if behavior and behavior != 'Not selected':
            base_name = f"{behavior}_{start_str}_{end_str}"
        else:
            base_name = f"{video_base_safe}_{start_str}_{end_str}"
        npz_name = f"{base_name}.npz"
        csv_name = f"{base_name}.csv"
        npz_path = os.path.join(self.output_dir, npz_name)
        csv_path = os.path.join(self.output_dir, csv_name)
        # Create subfolder for tracks, images, and preview
        tracks_folder = os.path.join(self.output_dir, f"tracks_{start_str}_{end_str}")
        os.makedirs(tracks_folder, exist_ok=True)
        # Save npz and csv at top level
        if behavior and behavior != 'Not selected':
            np.savez(npz_path, pose=pose_seq, behavior=behavior)
        else:
            np.savez(npz_path, pose=pose_seq)
        header = ','.join([f'{n}_x,{n}_y' for n in part_names])
        np.savetxt(csv_path, pose_seq, delimiter=',', header=header, comments='')
        # Save .txt and plot for each part in subfolder
        colors = {'body_center':'#0033cc', 'tail_base':'#ff7f7f', 'forelimbs':'#ffb6c1', 'head':'#ff6666'}
        plt.figure(figsize=(8,8))
        for name in part_names:
            arr = np.array(tracks[name])
            txt_path = os.path.join(tracks_folder, f'{name}_track.txt')
            np.savetxt(txt_path, arr, fmt='%.5f')
            if arr.size:
                # Individual plot
                plt.figure()
                plt.scatter(arr[:,0], arr[:,1], c=colors.get(name,'k'), s=5, label=name)
                plt.title(f'{name} track')
                plt.gca().invert_yaxis()
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(0, w)
                plt.ylim(h, 0)
                plt.legend()
                plt.savefig(os.path.join(tracks_folder, f'{name}_track_img.png'))
                plt.close()
                # Combined
                plt.figure(1)
                plt.scatter(arr[:,0], arr[:,1], c=colors.get(name,'k'), s=5, label=name)
        plt.title('Combined tracks')
        plt.gca().invert_yaxis()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(0, w)
        plt.ylim(h, 0)
        plt.legend()
        plt.savefig(os.path.join(tracks_folder, 'combined_tracks.png'))
        plt.close()
        # Heatmap for body_center
        arr = np.array(tracks['body_center'])
        arr = arr[~np.isnan(arr).any(axis=1)]
        if arr.size:
            plt.figure(figsize=(8,8))
            try:
                sns.kdeplot(x=arr[:,0], y=arr[:,1], fill=True, cmap='hot', bw_adjust=0.5, thresh=0.05)
            except Exception:
                plt.hist2d(arr[:,0], arr[:,1], bins=50, cmap='hot')
            plt.gca().invert_yaxis()
            plt.title('Body Center Heatmap')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(0, w)
            plt.ylim(h, 0)
            plt.savefig(os.path.join(tracks_folder, 'body_center_heatmap.png'))
            plt.close()
        # Move preview video to subfolder if it exists
        if self.writer:
            self.writer.release()
            self.writer = None
        # Use root.after to update UI from thread
        self.root.after(0, lambda: messagebox.showinfo('Done', 'Analysis completed.'))
        self.root.after(0, lambda: self.status.config(text='Finished'))
        self.root.after(0, lambda: self.btn_abort.config(state='disabled'))
        self.root.after(0, lambda: self.btn_start.config(state='normal'))
        self.root.after(0, lambda: self.btn_import.config(state='normal'))
        self.root.after(0, lambda: self.btn_roi.config(state='normal'))
        self.root.after(0, lambda: self.btn_output.config(state='normal'))
        self.root.after(0, lambda: self.btn_restart.config(state='disabled'))
        self.abort_flag.clear()
        # --- After analysis: output walk track and total walk time ---
        if self.use_transformer_model.get() and self.behav_model_path.get() and os.path.exists(self.behav_model_path.get()):
            # Save per-behavior tracks and plot
            behavior_tracks = {name: [] for name in behavior_names}
            for idx, behav in enumerate(behavior_predictions):
                if behav is not None and behav < len(behavior_names):
                    bc = tracks['body_center'][idx] if idx < len(tracks['body_center']) else (np.nan, np.nan)
                    behavior_tracks[behavior_names[behav]].append(bc)
            # Plot per-behavior tracks
            plt.figure(figsize=(8,8))
            colors_map = {'freeze':'#1f77b4', 'stand_rearing':'#2ca02c', 'walk':'#d62728'}
            for behav, arr in behavior_tracks.items():
                arr = np.array(arr)
                if arr.size and not np.isnan(arr).all():
                    plt.scatter(arr[:,0], arr[:,1], s=5, label=behav, c=colors_map.get(behav, None))
            plt.title('Body Center Tracks by Behavior')
            plt.gca().invert_yaxis()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(0, w)
            plt.ylim(h, 0)
            plt.legend()
            plt.savefig(os.path.join(tracks_folder, 'behavior_tracks.png'))
            plt.close()
            # Output time spent in each behavior
            behav_counts = {name:0 for name in behavior_names}
            for behav in behavior_predictions:
                if behav is not None and behav < len(behavior_names):
                    behav_counts[behavior_names[behav]] += 1
            behav_times = {k: v/fps for k,v in behav_counts.items()}
            behav_result_str = '\n'.join([f"{k}: {behav_times[k]:.2f} sec" for k in behavior_names])
            self.behav_result.config(text=f"Time spent in each behavior:\n{behav_result_str}\nTrack plot saved to: behavior_tracks.png")
        else:
            self.behav_result.config(text='No behavior model loaded or used.')

    def abort_analysis(self):
        self.abort_flag.set()
        self.status.config(text='Aborting...')

    def update_preview_img(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im_pil = PIL.Image.fromarray(img_rgb)
        imgtk = PIL.ImageTk.PhotoImage(image=im_pil)
        self.preview_label.imgtk = imgtk
        self.preview_label.config(image=imgtk)

    def update_time_labels(self, event=None):
        # Show current start/end in frames and hh:mm:ss.ff
        try:
            fps = self.user_fps if self.user_fps else 30
            s = self.start_frame.get()
            if self.end_mode.get() == 'duration':
                mins = self.duration_min.get()
                secs = self.duration_sec.get()
                total_sec = mins * 60 + secs
                e = s + int(total_sec * fps)
            else:
                e = self.end_frame.get()
            t1 = s / fps if fps else 0
            t2 = e / fps if fps else 0
            t1_str = self._format_time(t1)
            t2_str = self._format_time(t2)
            self.time_label.config(text=f'Selected range: Frame {s} ({t1_str}) to Frame {e} ({t2_str})')
        except Exception:
            self.time_label.config(text='Invalid frame input')

    def cleanup_resources(self):
        # Release video and writer, clear preview
        try:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                self.cap = None
        except Exception:
            pass
        try:
            if hasattr(self, 'writer') and self.writer:
                self.writer.release()
                self.writer = None
        except Exception:
            pass
        self.preview_label.config(image='')

    def restart_app(self):
        # Destroy and recreate the main window for a fresh start
        self.user_fps_var.set(30)
        self.root.destroy()
        main()

    def on_backend_change(self, value=None):
        backend = self.backend.get()
        if backend in ['ONNX', 'OpenVINO']:
            self.btn_select_model.config(state='normal')
        else:
            self.btn_select_model.config(state='disabled')
            self.model_path.set('')

    def select_model_file(self):
        backend = self.backend.get()
        if backend == 'ONNX':
            path = filedialog.askopenfilename(filetypes=[('ONNX Model', '*.onnx')])
        elif backend == 'OpenVINO':
            path = filedialog.askdirectory()
        else:
            path = ''
        if path:
            self.model_path.set(path)

    def export_model(self):
        from ultralytics import YOLO
        pt_path = filedialog.askopenfilename(filetypes=[('YOLOv8 PyTorch Model', '*.pt')], title='Select YOLOv8 .pt model to export')
        if not pt_path:
            return
        # Ask user for export format
        format_win = tk.Toplevel(self.root)
        format_win.title('Select Export Format')
        tk.Label(format_win, text='Export format:').pack()
        export_format = tk.StringVar(value='onnx')
        def do_export():
            fmt = export_format.get()
            format_win.destroy()
            model = YOLO(pt_path)
            out = model.export(format=fmt)
            if fmt == 'onnx':
                out_path = pt_path.replace('.pt', '.onnx')
            else:
                out_path = pt_path.replace('.pt', '_openvino_model/')
            tk.messagebox.showinfo('Export Complete', f'Model exported to: {out_path}')
        tk.Radiobutton(format_win, text='ONNX', variable=export_format, value='onnx').pack(anchor='w')
        tk.Radiobutton(format_win, text='OpenVINO', variable=export_format, value='openvino').pack(anchor='w')
        tk.Button(format_win, text='Export', command=do_export).pack(pady=5)

    def export_model_help(self):
        msg = (
            'Click Export Model in the Advanced tab.\n'
            'Select your .pt YOLOv8 model file.\n'
            'Choose either ONNX or OpenVINO as the export format.\n'
            "The model will be exported using Ultralytics' built-in export, and you'll get a message with the export location.\n"
            'This makes it easy to prepare ONNX or OpenVINO models for fast CPU inference, directly from your GUI—no command line needed!'
        )
        tk.messagebox.showinfo('Export Model Help', msg)

    def on_model_change(self, value=None):
        if self.model_size.get() == 'Custom':
            self.btn_select_custom_model.config(state='normal')
        else:
            self.btn_select_custom_model.config(state='disabled')

    def select_custom_model(self):
        path = filedialog.askopenfilename(filetypes=[('YOLOv8 PyTorch Model', '*.pt')], title='Select custom YOLOv8 .pt model')
        if path:
            self.custom_model_path.set(path)

    def _format_time(self, seconds):
        # Helper to format seconds as hh:mm:ss.ff
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        f = int((seconds - int(seconds)) * 100)
        return f"{h:02d}:{m:02d}:{s:02d}.{f:02d}"

    def on_end_mode_change(self):
        mode = self.end_mode.get()
        if mode == 'duration':
            self.scale_end.pack_forget()
            self.entry_end.pack_forget()
            self.duration_frame.pack(side='left', padx=10)
            self.update_time_labels()
        else:
            self.duration_frame.pack_forget()
            self.scale_end.pack(side='left', fill='x', expand=True, padx=5)
            self.entry_end.pack(side='left')
            self.update_time_labels()

    def on_duration_change(self, *args):
        if self.end_mode.get() == 'duration':
            try:
                fps = self.user_fps if self.user_fps else 30
                s = self.start_frame.get()
                mins = self.duration_min.get()
                secs = self.duration_sec.get()
                total_sec = mins * 60 + secs
                frames = int(total_sec * fps)
                e = s + frames
                self.end_frame.set(e)
            except Exception:
                pass
            self.update_time_labels()

    def select_behav_model(self):
        path = filedialog.askopenfilename(filetypes=[('PyTorch Model', '*.pth')], title='Select transformer .pth model')
        if path:
            self.behav_model_path.set(path)
            self.behav_status.config(text=f'Loaded: {os.path.basename(path)}')
        else:
            # If user cancels, keep auto-loaded model if present
            if self.behav_model_path.get():
                self.behav_status.config(text=f'Auto-loaded: {os.path.basename(self.behav_model_path.get())} (from script directory)')
            else:
                self.behav_status.config(text='No behavior model loaded.')

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__=='__main__':
    
    main()