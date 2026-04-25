"""
Eye-Gaze Estimation — GUI interface.

Full-featured graphical interface for real-time gaze estimation using
GazeDINO model, MediaPipe FaceMesh, calibration, distance and pupillometry.

Usage:
    python gui.py
    python gui.py --model checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import math
import threading
import time
from pathlib import Path

import cv2
import customtkinter as ctk
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Imports from project
# ---------------------------------------------------------------------------
from infer_gaze import (
    FaceMeshAnalyzer,
    EMA,
    load_model,
    predict,
    _get_device,
    _draw_eye_arrows,
    _draw_gaze_arrow,
    _draw_gaze_dot,
    _draw_distance,
    _draw_pupil,
)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class GazeApp(ctk.CTk):
    CAMERA_W = 800
    CAMERA_H = 600

    def __init__(self, model_path: str, camera_id: int = 0,
                 video_path: str | None = None, loop_video: bool = True):
        super().__init__()
        self.title("Eye-Gaze Estimation")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._running = True
        self._camera_id = camera_id
        self._video_path = video_path
        self._loop_video = loop_video
        self._video_mode = video_path is not None

        # -- load model
        self._device = _get_device()
        self._model = load_model(model_path, self._device)
        self._mesh = FaceMeshAnalyzer()
        self._ema = EMA(alpha=0.3)

        # calibrator
        self._calibrator = None
        self._calib_path = "calibration.pkl"
        if Path(self._calib_path).exists():
            from calibration import GazeCalibrator
            self._calibrator = GazeCalibrator.load(self._calib_path)

        # state
        self._show_distance = True
        self._show_pupil = True
        self._show_arrows = True
        self._yaw = 0.0
        self._pitch = 0.0
        self._dist_cm: float | None = None
        self._pupil_rel: float | None = None
        self._face_detected = False
        self._fps = 0.0

        # video playback state
        self._paused = False
        self._video_pos = 0
        self._video_total = 0
        self._video_fps = 30.0
        self._seeking = False

        # -- build UI
        self._build_ui()

        # -- open video source
        if self._video_mode:
            self._cap = cv2.VideoCapture(video_path)
            self._video_total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._video_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            src = Path(video_path).name
            print(f"[gui] Video mode: {src}  "
                  f"{self._video_total} frames, {self._video_fps:.1f} fps")
        else:
            self._cap = cv2.VideoCapture(camera_id)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_W)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_H)

        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

        # UI update timer
        self._update_ui()

    # ------------------------------------------------------------------ #
    # UI layout
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        # Main container
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # -- Left: camera feed
        self._canvas = ctk.CTkLabel(self, text="", fg_color="black",
                                    corner_radius=8)
        self._canvas.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nsew")
        self._photo = None

        # -- Right: sidebar
        sidebar = ctk.CTkFrame(self, width=280, corner_radius=10)
        sidebar.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="ns")
        sidebar.grid_propagate(False)

        # Title
        ctk.CTkLabel(sidebar, text="Eye-Gaze", font=("", 22, "bold")
                     ).pack(pady=(18, 2))
        ctk.CTkLabel(sidebar, text="Real-time gaze estimation",
                     font=("", 12), text_color="gray"
                     ).pack(pady=(0, 14))

        # -- Status section
        status_frame = ctk.CTkFrame(sidebar, corner_radius=8)
        status_frame.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(status_frame, text="STATUS",
                     font=("", 11, "bold"), text_color="gray"
                     ).pack(anchor="w", padx=10, pady=(8, 2))

        self._lbl_face = ctk.CTkLabel(status_frame, text="Face: --",
                                       font=("", 13))
        self._lbl_face.pack(anchor="w", padx=10, pady=1)

        self._lbl_fps = ctk.CTkLabel(status_frame, text="FPS: --",
                                      font=("", 13))
        self._lbl_fps.pack(anchor="w", padx=10, pady=1)

        self._lbl_device = ctk.CTkLabel(status_frame,
                                         text=f"Device: {self._device}",
                                         font=("", 13))
        self._lbl_device.pack(anchor="w", padx=10, pady=(1, 8))

        # -- Gaze values section
        gaze_frame = ctk.CTkFrame(sidebar, corner_radius=8)
        gaze_frame.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(gaze_frame, text="GAZE",
                     font=("", 11, "bold"), text_color="gray"
                     ).pack(anchor="w", padx=10, pady=(8, 2))

        self._lbl_yaw = ctk.CTkLabel(gaze_frame, text="Yaw:   0.0\u00b0",
                                      font=("Menlo", 16))
        self._lbl_yaw.pack(anchor="w", padx=10, pady=2)

        self._lbl_pitch = ctk.CTkLabel(gaze_frame, text="Pitch: 0.0\u00b0",
                                        font=("Menlo", 16))
        self._lbl_pitch.pack(anchor="w", padx=10, pady=(2, 8))

        # -- Metrics section
        metrics_frame = ctk.CTkFrame(sidebar, corner_radius=8)
        metrics_frame.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(metrics_frame, text="METRICS",
                     font=("", 11, "bold"), text_color="gray"
                     ).pack(anchor="w", padx=10, pady=(8, 2))

        self._lbl_dist = ctk.CTkLabel(metrics_frame, text="Distance: --",
                                       font=("", 13))
        self._lbl_dist.pack(anchor="w", padx=10, pady=1)

        self._lbl_pupil = ctk.CTkLabel(metrics_frame, text="Iris: --",
                                        font=("", 13))
        self._lbl_pupil.pack(anchor="w", padx=10, pady=(1, 8))

        # -- Toggles section
        toggle_frame = ctk.CTkFrame(sidebar, corner_radius=8)
        toggle_frame.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(toggle_frame, text="OVERLAY",
                     font=("", 11, "bold"), text_color="gray"
                     ).pack(anchor="w", padx=10, pady=(8, 4))

        self._sw_arrows = ctk.CTkSwitch(toggle_frame, text="Gaze arrows",
                                         command=self._toggle_arrows)
        self._sw_arrows.select()
        self._sw_arrows.pack(anchor="w", padx=10, pady=2)

        self._sw_dist = ctk.CTkSwitch(toggle_frame, text="Distance",
                                       command=self._toggle_distance)
        self._sw_dist.select()
        self._sw_dist.pack(anchor="w", padx=10, pady=2)

        self._sw_pupil = ctk.CTkSwitch(toggle_frame, text="Pupillometry",
                                        command=self._toggle_pupil)
        self._sw_pupil.select()
        self._sw_pupil.pack(anchor="w", padx=10, pady=(2, 10))

        # -- Buttons
        btn_frame = ctk.CTkFrame(sidebar, corner_radius=8, fg_color="transparent")
        btn_frame.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkButton(btn_frame, text="Calibrate",
                      command=self._run_calibration, height=36
                      ).pack(fill="x", pady=(0, 6))

        ctk.CTkButton(btn_frame, text="Reset",
                      command=self._reset, height=36,
                      fg_color="#555", hover_color="#666"
                      ).pack(fill="x", pady=(0, 6))

        ctk.CTkButton(btn_frame, text="Quit",
                      command=self._on_close, height=36,
                      fg_color="#a03030", hover_color="#c04040"
                      ).pack(fill="x")

        # -- Calibration status
        calib_txt = "Active" if self._calibrator else "None"
        self._lbl_calib = ctk.CTkLabel(sidebar, text=f"Calibration: {calib_txt}",
                                        font=("", 11), text_color="gray")
        self._lbl_calib.pack(pady=(4, 10))

        # -- Video playback controls (only in video mode)
        if self._video_mode:
            vid_frame = ctk.CTkFrame(sidebar, corner_radius=8)
            vid_frame.pack(fill="x", padx=12, pady=(0, 10))

            ctk.CTkLabel(vid_frame, text="VIDEO",
                         font=("", 11, "bold"), text_color="gray"
                         ).pack(anchor="w", padx=10, pady=(8, 4))

            # source label
            src_name = Path(self._video_path).name
            ctk.CTkLabel(vid_frame, text=src_name,
                         font=("", 11), text_color="#aaa"
                         ).pack(anchor="w", padx=10, pady=(0, 4))

            # play / pause button
            self._btn_playpause = ctk.CTkButton(
                vid_frame, text="Pause", height=30, width=80,
                command=self._toggle_pause)
            self._btn_playpause.pack(pady=(0, 6))

            # position slider
            self._slider = ctk.CTkSlider(
                vid_frame, from_=0, to=max(1, self._video_total - 1),
                command=self._on_slider, number_of_steps=self._video_total or 1)
            self._slider.set(0)
            self._slider.pack(fill="x", padx=10, pady=(0, 2))
            self._slider.bind("<ButtonPress-1>", lambda e: setattr(self, '_seeking', True))
            self._slider.bind("<ButtonRelease-1>", self._on_slider_release)

            # time label
            self._lbl_time = ctk.CTkLabel(vid_frame, text="0:00 / 0:00",
                                           font=("Menlo", 12))
            self._lbl_time.pack(pady=(0, 4))

            # loop toggle
            self._sw_loop = ctk.CTkSwitch(vid_frame, text="Loop",
                                           command=self._toggle_loop)
            if self._loop_video:
                self._sw_loop.select()
            self._sw_loop.pack(anchor="w", padx=10, pady=(2, 10))

    # ------------------------------------------------------------------ #
    # Toggle callbacks
    # ------------------------------------------------------------------ #
    def _toggle_arrows(self):
        self._show_arrows = self._sw_arrows.get()

    def _toggle_distance(self):
        self._show_distance = self._sw_dist.get()

    def _toggle_pupil(self):
        self._show_pupil = self._sw_pupil.get()

    def _reset(self):
        self._ema.reset()
        self._mesh.reset()
        self._yaw = 0.0
        self._pitch = 0.0

    # -- Video playback callbacks
    def _toggle_pause(self):
        self._paused = not self._paused
        self._btn_playpause.configure(text="Play" if self._paused else "Pause")

    def _toggle_loop(self):
        self._loop_video = bool(self._sw_loop.get())

    def _on_slider(self, value):
        """Called while dragging the slider."""
        pass  # actual seek happens on release

    def _on_slider_release(self, event):
        """Seek to slider position when released."""
        pos = int(self._slider.get())
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        self._video_pos = pos
        self._ema.reset()
        self._mesh.reset()
        self._seeking = False

    def _run_calibration(self):
        """Run calibration in a separate thread (uses OpenCV windows)."""
        def _calib():
            from calibration import run_calibration

            def _model_fn(face_bgr):
                return predict(self._model, face_bgr, self._device)

            def _detector_fn(frame_rgb):
                info = self._mesh.process(frame_rgb)
                return info["face_bbox"]

            try:
                self._calibrator = run_calibration(
                    _model_fn, _detector_fn,
                    camera_id=self._camera_id,
                    save_path=self._calib_path,
                )
                self._lbl_calib.configure(text="Calibration: Active")
            except Exception as e:
                print(f"[gui] Calibration error: {e}")

        threading.Thread(target=_calib, daemon=True).start()

    # ------------------------------------------------------------------ #
    # Camera loop (background thread)
    # ------------------------------------------------------------------ #
    def _camera_loop(self):
        frame_time = 0.0
        target_dt = 1.0 / self._video_fps if self._video_mode else 0.0

        while self._running:
            t0 = time.perf_counter()

            # Video mode: handle pause and loop
            if self._video_mode:
                if self._paused or self._seeking:
                    time.sleep(0.02)
                    continue

            ret, frame = self._cap.read()
            if not ret:
                if self._video_mode and self._loop_video:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._video_pos = 0
                    self._ema.reset()
                    self._mesh.reset()
                    continue
                elif self._video_mode:
                    self._paused = True
                    continue
                time.sleep(0.01)
                continue

            if self._video_mode:
                self._video_pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self._mesh.process(frame_rgb)
            info = faces[0] if faces else {}

            face_bbox = info.get("face_bbox")
            left_eye = info.get("left_eye")
            right_eye = info.get("right_eye")

            self._face_detected = face_bbox is not None

            if face_bbox is not None:
                x, y, w, h = face_bbox
                fh, fw = frame.shape[:2]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(fw, x + w), min(fh, y + h)

                if x2 > x1 and y2 > y1:
                    face_bgr = frame[y1:y2, x1:x2]
                    raw_yaw, raw_pitch = predict(self._model, face_bgr,
                                                 self._device)
                    smoothed = self._ema.update(
                        np.array([raw_yaw, raw_pitch]))
                    yaw, pitch = float(smoothed[0]), float(smoothed[1])
                    self._yaw = yaw
                    self._pitch = pitch

                    # Draw on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (200, 200, 200), 1)

                    if self._show_arrows:
                        if left_eye or right_eye:
                            _draw_eye_arrows(frame, left_eye, right_eye,
                                             yaw, pitch, 120)
                        else:
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            _draw_gaze_arrow(frame, cx, cy, yaw, pitch, 120)

                    if self._calibrator:
                        sx, sy = self._calibrator.predict(yaw, pitch)
                        _draw_gaze_dot(frame, sx, sy)

                    # HUD text on frame
                    yaw_d = math.degrees(yaw)
                    pitch_d = math.degrees(pitch)
                    cv2.putText(frame, f"Yaw {yaw_d:+.1f}  Pitch {pitch_d:+.1f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 230, 60), 2)
            else:
                cv2.putText(frame, "No face", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)
                self._ema.reset()

            # Distance & pupil overlays
            if self._show_distance:
                _draw_distance(frame, info.get("distance", {}))
            if self._show_pupil:
                _draw_pupil(frame, info.get("pupil"))

            self._dist_cm = info.get("distance", {}).get("cm")
            pupil_info = info.get("pupil")
            self._pupil_rel = pupil_info["mean_rel"] if pupil_info else None

            # Convert to display
            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._latest_frame = Image.fromarray(display)

            dt = time.perf_counter() - t0
            frame_time = 0.9 * frame_time + 0.1 * dt
            self._fps = 1.0 / max(frame_time, 1e-6)

            # In video mode, pace to original FPS
            if self._video_mode and target_dt > dt:
                time.sleep(target_dt - dt)

        self._cap.release()

    # ------------------------------------------------------------------ #
    # UI update (main thread)
    # ------------------------------------------------------------------ #
    def _update_ui(self):
        if not self._running:
            return

        # Update camera image
        if hasattr(self, "_latest_frame"):
            frame = self._latest_frame
            # Resize to fit canvas
            cw = self._canvas.winfo_width() or self.CAMERA_W
            ch = self._canvas.winfo_height() or self.CAMERA_H
            if cw > 10 and ch > 10:
                frame = frame.resize((cw, ch), Image.LANCZOS)
            self._photo = ctk.CTkImage(light_image=frame, size=(cw, ch))
            self._canvas.configure(image=self._photo)

        # Update labels
        face_color = "#00e060" if self._face_detected else "#e04040"
        face_text = "Detected" if self._face_detected else "Not found"
        self._lbl_face.configure(text=f"Face: {face_text}",
                                  text_color=face_color)

        self._lbl_fps.configure(text=f"FPS: {self._fps:.0f}")

        yaw_d = math.degrees(self._yaw)
        pitch_d = math.degrees(self._pitch)
        self._lbl_yaw.configure(text=f"Yaw:   {yaw_d:+6.1f}\u00b0")
        self._lbl_pitch.configure(text=f"Pitch: {pitch_d:+6.1f}\u00b0")

        if self._dist_cm is not None:
            self._lbl_dist.configure(text=f"Distance: {self._dist_cm:.0f} cm")
        else:
            self._lbl_dist.configure(text="Distance: --")

        if self._pupil_rel is not None:
            self._lbl_pupil.configure(text=f"Iris: {self._pupil_rel:.3f}")
        else:
            self._lbl_pupil.configure(text="Iris: --")

        # Video playback UI
        if self._video_mode and hasattr(self, '_slider'):
            if not self._seeking:
                self._slider.set(self._video_pos)
            cur_s = self._video_pos / max(self._video_fps, 1)
            tot_s = self._video_total / max(self._video_fps, 1)
            self._lbl_time.configure(
                text=f"{int(cur_s)//60}:{int(cur_s)%60:02d} / "
                     f"{int(tot_s)//60}:{int(tot_s)%60:02d}")
            if self._paused:
                self._btn_playpause.configure(text="Play")

        self.after(33, self._update_ui)  # ~30 fps UI refresh

    # ------------------------------------------------------------------ #
    def _on_close(self):
        self._running = False
        time.sleep(0.1)
        self.destroy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Eye-Gaze Estimation GUI")
    p.add_argument("--model", default=None,
                   help="Path to .pt checkpoint (auto-finds latest if omitted)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", default=None,
                   help="Path to video file (test mode — replaces camera)")
    p.add_argument("--no-loop", action="store_true",
                   help="Do not loop video at end")
    args = p.parse_args()

    model_path = args.model
    if model_path is None:
        ckpts = sorted(Path("checkpoints").glob("**/best*.pt"))
        if not ckpts:
            print("[gui] No checkpoint found. Provide --model path.")
            raise SystemExit(1)
        model_path = str(ckpts[-1])
        print(f"[gui] Auto-selected: {model_path}")

    if args.video and not Path(args.video).exists():
        print(f"[gui] Video not found: {args.video}")
        raise SystemExit(1)

    app = GazeApp(
        model_path=model_path,
        camera_id=args.camera,
        video_path=args.video,
        loop_video=not args.no_loop,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
