"""
Video perspective correction: select a quadrilateral ROI, warp it to a chosen aspect ratio,
optionally rotate, and export the processed video.
"""
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


def rotate_image(image, angle):
    """Rotate image around its center; output size matches input."""
    if angle == 0:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, m, (w, h))


def get_perspective_matrix(pts, out_w, out_h, pad):
    """
    Map 4 source points to a padded rectangle of size (out_w, out_h).
    Returns perspective matrix M and output canvas size (total_w, total_h).
    """
    pts = np.array(pts, dtype=np.float32)
    dst = np.array(
        [
            [pad, pad],
            [pad + out_w, pad],
            [pad + out_w, pad + out_h],
            [pad, pad + out_h],
        ],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(pts, dst)
    total_w = int(pad * 2 + out_w)
    total_h = int(pad * 2 + out_h)
    return m, total_w, total_h


def compute_output_dims(base_size, aspect_w, aspect_h):
    """
    Given a base edge length and aspect ratio aspect_w:aspect_h,
    return (out_w, out_h) with integer dimensions preserving the ratio.
    Uses base_size as the width of the warped content region (height scales).
    """
    if aspect_w <= 0 or aspect_h <= 0:
        raise ValueError("Aspect ratio parts must be positive.")
    aw, ah = float(aspect_w), float(aspect_h)
    out_w = int(round(base_size))
    out_h = int(round(base_size * ah / aw))
    if out_w < 1:
        out_w = 1
    if out_h < 1:
        out_h = 1
    return out_w, out_h


def process_video(video_path, output_path, angle, m, total_w, total_h):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_w, total_h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rotated = rotate_image(frame, angle)
        warped = cv2.warpPerspective(rotated, m, (total_w, total_h))
        out.write(warped)
        frame_count += 1
        if total_frames > 0 and frame_count % 50 == 0:
            print(f"Progress: {frame_count}/{total_frames}")

    cap.release()
    out.release()
    print("Video saved successfully.")


def draw_numbered_points(bgr, points, scale=1.0):
    """Draw corner markers on the image (in-place)."""
    for i, p in enumerate(points):
        x, y = int(p[0] * scale), int(p[1] * scale)
        cv2.circle(bgr, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            bgr,
            str(i + 1),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
        )


def hstack_match_height(left, right):
    """Horizontally stack two BGR images, padding the shorter one to match height."""
    if left.shape[0] != right.shape[0]:
        mh = max(left.shape[0], right.shape[0])
        if left.shape[0] < mh:
            left = cv2.copyMakeBorder(left, 0, mh - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if right.shape[0] < mh:
            right = cv2.copyMakeBorder(right, 0, mh - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return np.hstack((left, right))


def make_side_by_side_warp_preview(rotated, warped, points):
    """Small preview: left = rotated with points, right = warped patch."""
    h0, w0 = rotated.shape[:2]
    h1, w1 = warped.shape[:2]
    scale_l = min(1.0, 400 / max(h0, 1))
    scale_r = min(1.0, 400 / max(h1, 1))
    left = cv2.resize(rotated, (int(w0 * scale_l), int(h0 * scale_l)))
    right = cv2.resize(warped, (int(w1 * scale_r), int(h1 * scale_r)))
    draw_numbered_points(left, points, scale=scale_l)
    return hstack_match_height(left, right)


class VideoStretchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video ROI Stretch & Rotate")
        self.geometry("1100x720")
        self.minsize(900, 600)

        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.angle = tk.StringVar(value="0")
        self.pad = tk.IntVar(value=80)
        self.base_size = tk.IntVar(value=500)
        self.aspect_w = tk.DoubleVar(value=1.0)
        self.aspect_h = tk.DoubleVar(value=1.0)

        self.cap = None
        self.frame_bgr = None
        self.points = []
        self.preview_job = None
        self._imgtk = None
        self._processing = False

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Input video:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        ttk.Entry(top, textvariable=self.video_path, width=70).grid(row=0, column=1, sticky=tk.EW, padx=4)
        ttk.Button(top, text="Browse…", command=self._browse_input).grid(row=0, column=2)

        ttk.Label(top, text="Output file:").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Entry(top, textvariable=self.output_path, width=70).grid(row=1, column=1, sticky=tk.EW, padx=4, pady=(6, 0))
        ttk.Button(top, text="Save as…", command=self._browse_output).grid(row=1, column=2, pady=(6, 0))

        top.columnconfigure(1, weight=1)

        params = ttk.LabelFrame(self, text="Transform", padding=8)
        params.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(params, text="Rotation (°):").grid(row=0, column=0, sticky=tk.W)
        rot_frame = ttk.Frame(params)
        rot_frame.grid(row=0, column=1, sticky=tk.W, padx=6)
        self.rot_entry = ttk.Entry(rot_frame, textvariable=self.angle, width=12)
        self.rot_entry.pack(side=tk.LEFT)
        ttk.Label(rot_frame, text="(type any angle, e.g. -15 or 90.5)", font=("Segoe UI", 8), foreground="gray").pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(params, text="Padding (px):").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        ttk.Spinbox(params, from_=0, to=500, textvariable=self.pad, width=8).grid(row=0, column=3, sticky=tk.W, padx=6)

        ttk.Label(params, text="Warp width (px):").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Spinbox(params, from_=50, to=4000, textvariable=self.base_size, width=10).grid(row=1, column=1, sticky=tk.W, padx=6, pady=(8, 0))

        ttk.Label(params, text="Output aspect W:H:").grid(row=1, column=2, sticky=tk.W, pady=(8, 0), padx=(20, 0))
        ar = ttk.Frame(params)
        ar.grid(row=1, column=3, sticky=tk.W, pady=(8, 0), padx=6)
        ttk.Spinbox(ar, from_=0.01, to=100, increment=0.1, textvariable=self.aspect_w, width=6).pack(side=tk.LEFT)
        ttk.Label(ar, text=":").pack(side=tk.LEFT, padx=4)
        ttk.Spinbox(ar, from_=0.01, to=100, increment=0.1, textvariable=self.aspect_h, width=6).pack(side=tk.LEFT)
        ttk.Button(ar, text="1:1 (square)", command=lambda: self._preset_aspect(1, 1)).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(ar, text="16:9", command=lambda: self._preset_aspect(16, 9)).pack(side=tk.LEFT, padx=6)
        ttk.Button(ar, text="4:3", command=lambda: self._preset_aspect(4, 3)).pack(side=tk.LEFT, padx=0)

        hint = ttk.Label(
            params,
            text="The ROI quadrilateral is warped to this aspect ratio. “Warp width” sets the content width; height follows W:H.",
            font=("Segoe UI", 8),
            foreground="gray",
        )
        hint.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))

        preview_frame = ttk.LabelFrame(self, text="Preview — click 4 corners on the rotated frame (order: TL → TR → BR → BL)", padding=8)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.canvas = tk.Canvas(preview_frame, bg="#222", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        btns = ttk.Frame(self, padding=8)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Open video", command=self._open_video).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Clear points (R)", command=self._clear_points).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Export full video", command=self._export_video).pack(side=tk.LEFT, padx=4)
        ttk.Label(btns, text="Shortcuts: R = clear points").pack(side=tk.LEFT, padx=16)

        self.bind("<Key-r>", lambda e: self._clear_points())
        self.bind("<Key-R>", lambda e: self._clear_points())

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _parse_rotation_angle(self):
        """Degrees; negative allowed. Returns None if empty/invalid."""
        s = self.angle.get().strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def _resolve_output_dims(self, *, strict=False):
        """Returns (pad, out_w, out_h). If strict, invalid aspect raises ValueError."""
        pad = int(self.pad.get())
        base_size = int(self.base_size.get())
        aw, ah = float(self.aspect_w.get()), float(self.aspect_h.get())
        try:
            out_w, out_h = compute_output_dims(base_size, aw, ah)
        except ValueError:
            if strict:
                raise
            out_w, out_h = base_size, base_size
        return pad, out_w, out_h

    def _preset_aspect(self, w, h):
        self.aspect_w.set(float(w))
        self.aspect_h.set(float(h))

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")],
        )
        if path:
            self.video_path.set(path)
            base, _ = os.path.splitext(os.path.basename(path))
            out_dir = os.path.join(os.path.dirname(path), "video_modified")
            os.makedirs(out_dir, exist_ok=True)
            self.output_path.set(os.path.join(out_dir, f"{base}_corrected.mp4"))
            self.after(0, self._open_video)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save processed video as",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("All", "*.*")],
        )
        if path:
            self.output_path.set(path)

    def _on_canvas_configure(self, _event=None):
        self._schedule_preview_refresh()

    def _open_video(self):
        path = self.video_path.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showwarning("No file", "Choose a valid input video.")
            return
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open the video.")
            self.cap = None
            return
        self._clear_points()

    def _clear_points(self):
        self.points = []
        self._schedule_preview_refresh()

    def _schedule_preview_refresh(self):
        if self.preview_job is not None:
            self.after_cancel(self.preview_job)
        self.preview_job = self.after(16, self._tick_preview)

    def _tick_preview(self):
        self.preview_job = None
        if self.cap is None or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return

        angle = self._parse_rotation_angle() or 0.0
        rotated = rotate_image(frame, angle)
        self.frame_bgr = rotated.copy()

        pad, out_w, out_h = self._resolve_output_dims(strict=False)

        display = rotated
        if len(self.points) == 4:
            try:
                m, tw, th = get_perspective_matrix(self.points, out_w, out_h, pad)
                warped = cv2.warpPerspective(rotated, m, (tw, th))
                display = make_side_by_side_warp_preview(rotated, warped, self.points)
            except Exception as ex:
                print("Preview warp error:", ex)
                display = rotated
        else:
            draw_numbered_points(display, self.points, scale=1.0)

        self._draw_frame_on_canvas(display, allow_clicks=len(self.points) < 4)
        self.preview_job = self.after(33, self._tick_preview)

    def _draw_frame_on_canvas(self, bgr, allow_clicks=True):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        h, w = bgr.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        if nw < 1 or nh < 1:
            return
        resized = cv2.resize(bgr, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        self._imgtk = ImageTk.PhotoImage(im)
        self.canvas.delete("all")
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self._imgtk)
        self._scale_xy = (scale, scale)
        self._offset_xy = (ox, oy)
        self._click_ok = allow_clicks

    def _on_canvas_click(self, event):
        if not getattr(self, "_click_ok", True) or self.frame_bgr is None:
            return
        sx, sy = getattr(self, "_scale_xy", (1.0, 1.0))
        ox, oy = getattr(self, "_offset_xy", (0, 0))
        lx = event.x - ox
        ly = event.y - oy
        ix = lx / sx
        iy = ly / sy
        w, h = self.frame_bgr.shape[1], self.frame_bgr.shape[0]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return
        self.points.append([float(ix), float(iy)])
        self._schedule_preview_refresh()

    def _export_video(self):
        path_in = self.video_path.get().strip()
        path_out = self.output_path.get().strip()
        if not path_in or not os.path.isfile(path_in):
            messagebox.showwarning("No file", "Select a valid input video.")
            return
        if not path_out:
            messagebox.showwarning("No output", "Set an output file path.")
            return
        if len(self.points) != 4:
            messagebox.showwarning("Points", "Click exactly 4 corners on the rotated frame (TL, TR, BR, BL).")
            return
        if self._processing:
            return

        try:
            pad, out_w, out_h = self._resolve_output_dims(strict=True)
        except ValueError as e:
            messagebox.showerror("Invalid aspect", str(e))
            return

        angle = self._parse_rotation_angle()
        if angle is None:
            messagebox.showerror("Invalid angle", "Enter a valid rotation in degrees (numbers only, negative allowed).")
            return
        try:
            m, total_w, total_h = get_perspective_matrix(self.points, out_w, out_h, pad)
        except Exception as e:
            messagebox.showerror("Transform error", str(e))
            return

        self._processing = True

        def run():
            try:
                process_video(path_in, path_out, angle, m, total_w, total_h)
                self.after(0, lambda: messagebox.showinfo("Done", f"Saved:\n{path_out}"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Export failed", str(e)))
            finally:
                self.after(0, self._export_done)

        threading.Thread(target=run, daemon=True).start()

    def _export_done(self):
        self._processing = False

    def _on_close(self):
        if self.preview_job is not None:
            self.after_cancel(self.preview_job)
        if self.cap is not None:
            self.cap.release()
        self.destroy()


def main():
    app = VideoStretchApp()
    app.mainloop()


if __name__ == "__main__":
    main()
