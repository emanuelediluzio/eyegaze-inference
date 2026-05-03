"""
Eye-Gaze Estimation — GUI.

Usage:
    python gui.py
    python gui.py --video path/to/file.mp4
"""
from __future__ import annotations

import argparse
import math
import threading
import time
import tkinter.filedialog as fd
from pathlib import Path

import cv2
import customtkinter as ctk
import numpy as np
import torch
from PIL import Image

from infer_gaze import (
    FaceMeshAnalyzer,
    EMA,
    load_model,
    predict,
    _get_device,
    _draw_eye_arrows,
    _draw_gaze_arrow,
    _draw_face_box,
    _draw_iris_dots,
)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
BG        = "#000000"
SURFACE   = "#111111"
BORDER    = "#222222"
TEXT      = "#e0e0e0"
TEXT_DIM  = "#555555"
ACCENT    = "#ffffff"
GREEN     = "#34c759"
YELLOW    = "#ffcc00"
RED       = "#ff3b30"


def _make_switch(parent, text, command, default_on=True):
    sw = ctk.CTkSwitch(
        parent, text=text, font=("SF Pro Text", 12),
        text_color=TEXT, fg_color=BORDER, progress_color="#444",
        button_color=ACCENT, button_hover_color="#ccc",
        command=command)
    if default_on:
        sw.select()
    return sw


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class GazeApp(ctk.CTk):
    CAMERA_W = 800
    CAMERA_H = 600

    def __init__(self, model_path: str, camera_id: int = 0,
                 video_path: str | None = None):
        super().__init__()
        self.title("EyeGaze")
        self.configure(fg_color=BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        ctk.set_appearance_mode("dark")

        self._running = True
        self._camera_id = camera_id
        self._model_path = model_path

        # model
        self._device = _get_device()
        self._model = load_model(model_path, self._device)
        self._mesh = FaceMeshAnalyzer()
        self._emas: dict[int, EMA] = {}

        # load val_err from checkpoint for display
        self._val_err: float = float('nan')
        try:
            _ckpt = torch.load(model_path, map_location=self._device, weights_only=False)
            self._val_err = float(_ckpt.get('val_err', _ckpt.get('val_err_deg', float('nan'))))
        except Exception:
            pass

        # calibration state
        self._calibrator = None
        self._calibrating = False

        # overlay state
        self._show_face_box = True
        self._show_arrows = True
        self._show_iris = True

        # data
        self._faces_data: list[dict] = []
        self._face_count = 0
        self._fps = 0.0
        self._selected_face = 0
        self._prev_centroids: list[tuple[float, float]] = []

        # video
        self._video_mode = video_path is not None
        self._video_path = video_path
        self._paused = False
        self._video_pos = 0
        self._video_total = 0
        self._video_fps = 30.0
        self._seeking = False
        self._loop_video = True

        # open source before UI (macOS camera permission)
        self._cap = None
        self._open_source(video_path)

        # UI
        self._build_ui()

        # processing thread
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        self._update_ui()

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------
    def _open_source(self, video_path: str | None = None):
        if self._cap is not None:
            self._cap.release()
        self._video_mode = video_path is not None
        self._video_path = video_path
        self._paused = False
        self._video_pos = 0
        self._emas.clear()

        if self._video_mode:
            self._cap = cv2.VideoCapture(video_path)
            self._video_total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._video_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            self._cap = cv2.VideoCapture(self._camera_id)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_W)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_H)

    def _open_video_file(self):
        path = fd.askopenfilename(
            title="Open Video",
            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.webm"), ("All", "*.*")],
        )
        if not path:
            return
        self._open_source(path)
        self._rebuild_video_controls()

    def _switch_to_camera(self):
        self._open_source(None)
        self._rebuild_video_controls()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # -- feed
        self._canvas = ctk.CTkLabel(self, text="", fg_color=BG, corner_radius=0)
        self._canvas.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")
        self._photo = None

        # -- sidebar
        self._sidebar = ctk.CTkFrame(self, width=240, corner_radius=0,
                                      fg_color=SURFACE, border_width=0)
        self._sidebar.grid(row=0, column=1, padx=0, pady=0, sticky="ns")
        self._sidebar.grid_propagate(False)

        self._build_sidebar()

        # keyboard shortcuts
        self.bind("<q>", lambda e: self._on_close())
        self.bind("<Q>", lambda e: self._on_close())
        self.bind("<c>", lambda e: threading.Thread(
            target=self._run_calibration_thread, daemon=True).start())
        self.bind("<C>", lambda e: threading.Thread(
            target=self._run_calibration_thread, daemon=True).start())
        self.bind("<space>", lambda e: self._toggle_pause() if self._video_mode else None)
        self.bind("<Left>", lambda e: self._prev_face())
        self.bind("<Right>", lambda e: self._next_face())

    def _sep(self, parent, pad_x=16):
        ctk.CTkFrame(parent, height=1, fg_color=BORDER).pack(
            fill="x", padx=pad_x, pady=4)

    def _section_label(self, parent, text, pad_x=16, top=12):
        ctk.CTkLabel(parent, text=text, font=("SF Pro Text", 10),
                     text_color=TEXT_DIM).pack(
            anchor="w", padx=pad_x, pady=(top, 4))

    def _build_sidebar(self):
        for w in self._sidebar.winfo_children():
            w.destroy()

        sb = self._sidebar
        px = 16

        # -- title
        ctk.CTkLabel(sb, text="EyeGaze", font=("SF Pro Display", 20, "bold"),
                     text_color=ACCENT).pack(anchor="w", padx=px, pady=(20, 0))
        ctk.CTkLabel(sb, text="GazeDINO \u00b7 ViT-B/14",
                     font=("SF Pro Text", 10), text_color=TEXT_DIM
                     ).pack(anchor="w", padx=px, pady=(0, 12))

        # -- status
        self._lbl_face = ctk.CTkLabel(sb, text="No face",
                                       font=("SF Mono", 12), text_color=TEXT_DIM)
        self._lbl_face.pack(anchor="w", padx=px)

        self._lbl_fps = ctk.CTkLabel(sb, text="-- fps",
                                      font=("SF Mono", 12), text_color=TEXT_DIM)
        self._lbl_fps.pack(anchor="w", padx=px, pady=(0, 12))

        self._sep(sb, px)

        # -- model info
        self._section_label(sb, "MODEL", px)
        val_err_str = f"{self._val_err:.2f}\u00b0" if not math.isnan(self._val_err) else "--"
        ctk.CTkLabel(sb, text=f"Val err  {val_err_str}",
                     font=("SF Mono", 12), text_color=TEXT
                     ).pack(anchor="w", padx=px, pady=(0, 8))

        self._sep(sb, px)

        # -- face selector
        self._section_label(sb, "FACE", px)

        sel_frame = ctk.CTkFrame(sb, fg_color="transparent")
        sel_frame.pack(fill="x", padx=px, pady=(0, 6))

        ctk.CTkButton(sel_frame, text="\u25C0", width=28, height=24,
                      font=("SF Mono", 12), corner_radius=4,
                      fg_color=BORDER, hover_color="#333",
                      text_color=TEXT, command=self._prev_face
                      ).pack(side="left")

        self._lbl_face_sel = ctk.CTkLabel(sel_frame, text="Face 1/1",
                                           font=("SF Mono", 12), text_color=TEXT)
        self._lbl_face_sel.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(sel_frame, text="\u25B6", width=28, height=24,
                      font=("SF Mono", 12), corner_radius=4,
                      fg_color=BORDER, hover_color="#333",
                      text_color=TEXT, command=self._next_face
                      ).pack(side="left")

        # -- gaze
        self._section_label(sb, "GAZE", px)

        self._lbl_yaw = ctk.CTkLabel(sb, text="Yaw    --",
                                      font=("SF Mono", 14), text_color=TEXT)
        self._lbl_yaw.pack(anchor="w", padx=px, pady=1)

        self._lbl_pitch = ctk.CTkLabel(sb, text="Pitch  --",
                                        font=("SF Mono", 14), text_color=TEXT)
        self._lbl_pitch.pack(anchor="w", padx=px, pady=1)

        self._lbl_screen = ctk.CTkLabel(sb, text="",
                                         font=("SF Mono", 12), text_color=TEXT_DIM)
        self._lbl_screen.pack(anchor="w", padx=px, pady=(1, 12))

        self._sep(sb, px)

        # -- metrics
        self._section_label(sb, "METRICS", px)

        self._lbl_dist = ctk.CTkLabel(sb, text="Distance  --",
                                       font=("SF Mono", 12), text_color=TEXT)
        self._lbl_dist.pack(anchor="w", padx=px, pady=1)

        self._lbl_pupil = ctk.CTkLabel(sb, text="Iris      --",
                                        font=("SF Mono", 12), text_color=TEXT)
        self._lbl_pupil.pack(anchor="w", padx=px, pady=(1, 12))

        self._sep(sb, px)

        # -- calibration
        self._section_label(sb, "CALIBRATION", px)

        self._lbl_calib_status = ctk.CTkLabel(
            sb, text="Not calibrated", font=("SF Mono", 12), text_color=RED)
        self._lbl_calib_status.pack(anchor="w", padx=px, pady=(0, 6))

        calib_btn_frame = ctk.CTkFrame(sb, fg_color="transparent")
        calib_btn_frame.pack(fill="x", padx=px, pady=(0, 4))

        ctk.CTkButton(calib_btn_frame, text="Calibrate", height=28,
                      font=("SF Pro Text", 11), corner_radius=6,
                      fg_color=BORDER, hover_color="#333",
                      text_color=TEXT, border_width=1, border_color="#333",
                      command=lambda: threading.Thread(
                          target=self._run_calibration_thread, daemon=True).start()
                      ).pack(side="left", fill="x", expand=True, padx=(0, 3))

        ctk.CTkButton(calib_btn_frame, text="Load", height=28,
                      font=("SF Pro Text", 11), corner_radius=6,
                      fg_color=BORDER, hover_color="#333",
                      text_color=TEXT, border_width=1, border_color="#333",
                      command=self._load_calibration
                      ).pack(side="left", fill="x", expand=True, padx=3)

        ctk.CTkButton(calib_btn_frame, text="Clear", height=28,
                      font=("SF Pro Text", 11), corner_radius=6,
                      fg_color=BORDER, hover_color="#333",
                      text_color=TEXT, border_width=1, border_color="#333",
                      command=self._clear_calibration
                      ).pack(side="left", fill="x", expand=True, padx=(3, 0))

        self._sep(sb, px)

        # -- overlay toggles
        self._section_label(sb, "OVERLAY", px)

        self._sw_face_box = _make_switch(sb, "Face box",
                                          lambda: setattr(self, '_show_face_box',
                                                          bool(self._sw_face_box.get())))
        self._sw_face_box.pack(anchor="w", padx=px, pady=2)

        self._sw_arrows = _make_switch(sb, "Gaze arrows",
                                        lambda: setattr(self, '_show_arrows',
                                                        bool(self._sw_arrows.get())))
        self._sw_arrows.pack(anchor="w", padx=px, pady=2)

        self._sw_iris = _make_switch(sb, "Iris dots",
                                      lambda: setattr(self, '_show_iris',
                                                      bool(self._sw_iris.get())))
        self._sw_iris.pack(anchor="w", padx=px, pady=(2, 16))

        self._sep(sb, px)

        # -- source
        self._section_label(sb, "SOURCE", px)

        btn_frame = ctk.CTkFrame(sb, fg_color="transparent")
        btn_frame.pack(fill="x", padx=px)

        ctk.CTkButton(btn_frame, text="Open Video", height=32,
                      font=("SF Pro Text", 12), corner_radius=6,
                      fg_color=BORDER, hover_color="#333",
                      text_color=TEXT, border_width=1, border_color="#333",
                      command=self._open_video_file
                      ).pack(fill="x", pady=(0, 6))

        ctk.CTkButton(btn_frame, text="Camera", height=32,
                      font=("SF Pro Text", 12), corner_radius=6,
                      fg_color=BORDER, hover_color="#333",
                      text_color=TEXT, border_width=1, border_color="#333",
                      command=self._switch_to_camera
                      ).pack(fill="x", pady=(0, 6))

        # video controls placeholder
        self._vid_frame = ctk.CTkFrame(sb, fg_color="transparent")
        self._vid_frame.pack(fill="x", padx=0, pady=0)
        self._rebuild_video_controls()

        # spacer
        ctk.CTkFrame(sb, fg_color="transparent").pack(fill="both", expand=True)

        # shortcuts hint
        ctk.CTkLabel(sb, text="C=Calibrate  Q=Quit  \u2190\u2192=Face",
                     font=("SF Mono", 9), text_color="#333"
                     ).pack(anchor="w", padx=px, pady=(0, 2))

        # device label at bottom
        ctk.CTkLabel(sb, text=f"{self._device}",
                     font=("SF Mono", 10), text_color="#333"
                     ).pack(anchor="w", padx=px, pady=(0, 12))

        # sync calibration label state
        self._update_calib_status()

    def _rebuild_video_controls(self):
        for w in self._vid_frame.winfo_children():
            w.destroy()

        px = 16

        if not self._video_mode:
            ctk.CTkLabel(self._vid_frame, text="Live camera",
                         font=("SF Mono", 11), text_color=TEXT_DIM
                         ).pack(anchor="w", padx=px, pady=(8, 0))
            return

        vf = self._vid_frame

        self._sep(vf, px)
        self._section_label(vf, "VIDEO", px, top=8)

        src_name = Path(self._video_path).name
        if len(src_name) > 24:
            src_name = src_name[:24] + "\u2026"
        ctk.CTkLabel(vf, text=src_name, font=("SF Mono", 11),
                     text_color=TEXT_DIM).pack(anchor="w", padx=px, pady=(0, 6))

        self._btn_playpause = ctk.CTkButton(
            vf, text="Pause", height=28, width=70,
            font=("SF Pro Text", 11), corner_radius=6,
            fg_color=BORDER, hover_color="#333",
            text_color=TEXT, border_width=1, border_color="#333",
            command=self._toggle_pause)
        self._btn_playpause.pack(pady=(0, 6))

        self._slider = ctk.CTkSlider(
            vf, from_=0, to=max(1, self._video_total - 1),
            number_of_steps=max(1, self._video_total),
            fg_color=BORDER, progress_color="#444",
            button_color=ACCENT, button_hover_color="#ccc",
            command=lambda v: None)
        self._slider.set(0)
        self._slider.pack(fill="x", padx=px, pady=(0, 2))
        self._slider.bind("<ButtonPress-1>",
                          lambda e: setattr(self, '_seeking', True))
        self._slider.bind("<ButtonRelease-1>", self._on_slider_release)

        self._lbl_time = ctk.CTkLabel(vf, text="0:00 / 0:00",
                                       font=("SF Mono", 11), text_color=TEXT_DIM)
        self._lbl_time.pack(pady=(0, 8))

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _prev_face(self):
        if self._faces_data:
            self._selected_face = (self._selected_face - 1) % len(self._faces_data)

    def _next_face(self):
        if self._faces_data:
            self._selected_face = (self._selected_face + 1) % len(self._faces_data)

    def _toggle_pause(self):
        self._paused = not self._paused
        if hasattr(self, '_btn_playpause'):
            self._btn_playpause.configure(
                text="Play" if self._paused else "Pause")

    def _on_slider_release(self, event):
        pos = int(self._slider.get())
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        self._video_pos = pos
        self._emas.clear()
        self._mesh.reset()
        self._seeking = False

    # ------------------------------------------------------------------
    # Processing loop (background thread)
    # ------------------------------------------------------------------
    def _stable_sort_faces(self, faces: list[dict]) -> list[dict]:
        """Reorder faces to match previous frame's ordering by centroid proximity."""
        if not faces:
            self._prev_centroids = []
            return faces

        # compute centroids for current frame
        centroids = []
        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            centroids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        if not self._prev_centroids:
            self._prev_centroids = centroids
            return faces

        # match current faces to previous by nearest centroid
        n_prev = len(self._prev_centroids)
        n_curr = len(centroids)
        used = [False] * n_curr
        ordered: list[dict | None] = [None] * max(n_prev, n_curr)

        for pi, (px, py) in enumerate(self._prev_centroids):
            best_j, best_d = -1, float("inf")
            for ci, (cx, cy) in enumerate(centroids):
                if used[ci]:
                    continue
                d = (px - cx) ** 2 + (py - cy) ** 2
                if d < best_d:
                    best_d, best_j = d, ci
            if best_j >= 0 and best_d < 200_000:  # ~450px max shift
                ordered[pi] = faces[best_j]
                used[best_j] = True

        # append unmatched new faces
        slot = 0
        for ci in range(n_curr):
            if not used[ci]:
                while slot < len(ordered) and ordered[slot] is not None:
                    slot += 1
                if slot < len(ordered):
                    ordered[slot] = faces[ci]
                else:
                    ordered.append(faces[ci])

        result = [f for f in ordered if f is not None]
        self._prev_centroids = [
            ((f["bbox"][0] + f["bbox"][2]) / 2.0,
             (f["bbox"][1] + f["bbox"][3]) / 2.0)
            for f in result
        ]
        return result

    def _get_ema(self, face_idx: int) -> EMA:
        if face_idx not in self._emas:
            self._emas[face_idx] = EMA(alpha=0.3)
        return self._emas[face_idx]

    def _process_loop(self):
        frame_time = 0.0

        while self._running:
            t0 = time.perf_counter()

            # pause loop while calibration is running
            if self._calibrating:
                time.sleep(0.02)
                continue

            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.05)
                continue

            target_dt = 1.0 / self._video_fps if self._video_mode else 0.0

            if self._video_mode and (self._paused or self._seeking):
                time.sleep(0.02)
                continue

            ret, frame = self._cap.read()
            if not ret:
                if self._video_mode and self._loop_video:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._video_pos = 0
                    self._emas.clear()
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

            self._face_count = len(faces)
            faces_out = []

            for fi, info in enumerate(faces):
                face_bbox = info.get("face_bbox")
                left_eye = info.get("left_eye")
                right_eye = info.get("right_eye")
                dist_info = info.get("distance", {})
                pupil_info = info.get("pupil")

                if face_bbox is None:
                    continue

                x, y, w, h = face_bbox
                fh, fw = frame.shape[:2]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(fw, x + w), min(fh, y + h)

                if x2 <= x1 or y2 <= y1:
                    continue

                face_bgr = frame[y1:y2, x1:x2]
                raw_yaw, raw_pitch = predict(self._model, face_bgr, self._device)

                ema = self._get_ema(fi)
                smoothed = ema.update(np.array([raw_yaw, raw_pitch]))
                yaw, pitch = float(smoothed[0]), float(smoothed[1])

                face_entry = {
                    "yaw": yaw, "pitch": pitch,
                    "bbox": (x1, y1, x2, y2),
                    "left_eye": left_eye, "right_eye": right_eye,
                    "dist_cm": dist_info.get("cm"),
                    "dist_status": dist_info.get("status", "ok"),
                    "pupil_rel": pupil_info["mean_rel"] if pupil_info else None,
                    "screen_xy": None,
                }

                # apply calibration if fitted
                if self._calibrator is not None and self._calibrator.is_fitted:
                    try:
                        sx, sy = self._calibrator.predict(yaw, pitch)
                        face_entry["screen_xy"] = (sx, sy)
                    except Exception:
                        pass

                faces_out.append(face_entry)

                # overlays — scale with face size
                face_w = x2 - x1
                arrow_len = max(50, face_w * 2 // 3)
                iris_r = max(5, face_w // 30)

                if self._show_face_box:
                    _draw_face_box(frame, x1, y1, x2, y2)
                if self._show_iris:
                    _draw_iris_dots(frame, left_eye, right_eye, radius=iris_r)
                if self._show_arrows:
                    if left_eye or right_eye:
                        _draw_eye_arrows(frame, left_eye, right_eye,
                                         yaw, pitch, arrow_len)
                    else:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        _draw_gaze_arrow(frame, cx, cy, yaw, pitch, arrow_len)

            # stable face ordering via centroid matching
            faces_out = self._stable_sort_faces(faces_out)
            self._faces_data = faces_out

            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._latest_frame = Image.fromarray(display)

            dt = time.perf_counter() - t0
            frame_time = 0.9 * frame_time + 0.1 * dt
            self._fps = 1.0 / max(frame_time, 1e-6)

            # video pacing: sleep if ahead, skip frames if behind
            if self._video_mode and target_dt > 0:
                if dt < target_dt:
                    time.sleep(target_dt - dt)
                else:
                    frames_behind = int(dt / target_dt) - 1
                    if frames_behind > 0 and self._cap is not None:
                        new_pos = self._video_pos + frames_behind
                        if new_pos < self._video_total:
                            self._cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)

        if self._cap is not None:
            self._cap.release()

    # ------------------------------------------------------------------
    # UI update (main thread)
    # ------------------------------------------------------------------
    def _update_ui(self):
        if not self._running:
            return

        # frame
        if hasattr(self, "_latest_frame"):
            frame = self._latest_frame
            fw, fh = frame.size
            cw = self._canvas.winfo_width() or self.CAMERA_W
            ch = self._canvas.winfo_height() or self.CAMERA_H
            if cw > 10 and ch > 10 and fw > 0 and fh > 0:
                scale = min(cw / fw, ch / fh)
                nw, nh = int(fw * scale), int(fh * scale)
                frame = frame.resize((nw, nh), Image.LANCZOS)
                self._photo = ctk.CTkImage(light_image=frame, size=(nw, nh))
                self._canvas.configure(image=self._photo)

        # status
        n = self._face_count
        if n == 0:
            self._lbl_face.configure(text="No face", text_color=RED)
        elif n == 1:
            self._lbl_face.configure(text="1 face", text_color=GREEN)
        else:
            self._lbl_face.configure(text=f"{n} faces", text_color=GREEN)

        self._lbl_fps.configure(text=f"{self._fps:.0f} fps")

        # face selector
        n_faces = len(self._faces_data)
        if n_faces > 0:
            self._selected_face = min(self._selected_face, n_faces - 1)
        else:
            self._selected_face = 0
        if hasattr(self, '_lbl_face_sel'):
            self._lbl_face_sel.configure(
                text=f"Face {self._selected_face + 1}/{max(n_faces, 1)}")

        # gaze + metrics (selected face)
        if self._faces_data:
            f = self._faces_data[self._selected_face]
            yd = math.degrees(f["yaw"])
            pd = math.degrees(f["pitch"])
            self._lbl_yaw.configure(text=f"Yaw   {yd:+6.1f}\u00b0")
            self._lbl_pitch.configure(text=f"Pitch {pd:+6.1f}\u00b0")

            sxy = f.get("screen_xy")
            if sxy is not None:
                self._lbl_screen.configure(
                    text=f"Screen  {sxy[0]}, {sxy[1]}", text_color=TEXT_DIM)
            else:
                self._lbl_screen.configure(text="")

            dcm = f.get("dist_cm")
            dist_status = f.get("dist_status", "ok")
            if dcm:
                dist_color = {"ok": GREEN, "too_close": RED, "too_far": YELLOW}.get(
                    dist_status, TEXT)
                self._lbl_dist.configure(
                    text=f"Distance  {dcm:.0f} cm", text_color=dist_color)
            else:
                self._lbl_dist.configure(text="Distance  --", text_color=TEXT)

            pr = f.get("pupil_rel")
            self._lbl_pupil.configure(
                text=f"Iris      {pr:.3f}" if pr else "Iris      --")
        else:
            self._lbl_yaw.configure(text="Yaw    --")
            self._lbl_pitch.configure(text="Pitch  --")
            self._lbl_screen.configure(text="")
            self._lbl_dist.configure(text="Distance  --", text_color=TEXT)
            self._lbl_pupil.configure(text="Iris      --")

        # video
        if self._video_mode and hasattr(self, '_slider'):
            if not self._seeking:
                self._slider.set(self._video_pos)
            cur_s = self._video_pos / max(self._video_fps, 1)
            tot_s = self._video_total / max(self._video_fps, 1)
            if hasattr(self, '_lbl_time'):
                self._lbl_time.configure(
                    text=f"{int(cur_s)//60}:{int(cur_s)%60:02d} / "
                         f"{int(tot_s)//60}:{int(tot_s)%60:02d}")

        self.after(33, self._update_ui)

    # ------------------------------------------------------------------
    # Calibration methods
    # ------------------------------------------------------------------
    def _run_calibration_thread(self):
        """Executed in a separate thread so it doesn't block the UI."""
        import mediapipe as mp
        import torchvision.transforms as T
        from calibration import run_calibration

        # 1. Stop the processing loop
        self._calibrating = True
        time.sleep(0.15)  # wait for loop to pause

        # 2. Release camera so run_calibration can open it
        if self._cap:
            self._cap.release()
            self._cap = None

        fd_mp = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

        try:
            # 3. Build model_fn and detector_fn
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            def model_fn(face_bgr):
                rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                inp = transform(rgb).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    out = self._model(inp)[0].cpu().numpy()
                return float(out[0]), float(out[1])

            def detector_fn(frame_rgb):
                h, w = frame_rgb.shape[:2]
                res = fd_mp.process(frame_rgb)
                if not res.detections:
                    return None
                best = max(res.detections, key=lambda d: d.score[0])
                bb = best.location_data.relative_bounding_box
                return (max(0, int(bb.xmin * w)), max(0, int(bb.ymin * h)),
                        int(bb.width * w), int(bb.height * h))

            # 4. Run calibration (opens its own VideoCapture)
            self._calibrator = run_calibration(
                model_fn, detector_fn, camera_id=self._camera_id)

            # 5. Update UI on success
            self.after(0, self._on_calibration_done, True)

        except KeyboardInterrupt:
            self.after(0, self._on_calibration_done, False)
        except Exception as e:
            print(f"[calib] Error: {e}")
            self.after(0, self._on_calibration_done, False)
        finally:
            fd_mp.close()
            # 6. Reopen camera for normal processing loop
            self._open_source(self._video_path if self._video_mode else None)
            self._calibrating = False

    def _on_calibration_done(self, success: bool):
        """Called on the main thread after calibration completes."""
        self._update_calib_status()

    def _update_calib_status(self):
        """Update the calibration status label in the sidebar."""
        if not hasattr(self, '_lbl_calib_status'):
            return
        if self._calibrator is not None and self._calibrator.is_fitted:
            self._lbl_calib_status.configure(
                text="Calibrated \u2713", text_color=GREEN)
        else:
            self._lbl_calib_status.configure(
                text="Not calibrated", text_color=RED)

    def _load_calibration(self):
        path = fd.askopenfilename(
            filetypes=[("Calibration", "*.pkl"), ("All", "*.*")])
        if path:
            try:
                from calibration import GazeCalibrator
                self._calibrator = GazeCalibrator.load(path)
                self._update_calib_status()
            except Exception as e:
                print(f"[calib] Load error: {e}")

    def _clear_calibration(self):
        self._calibrator = None
        self._update_calib_status()

    # ------------------------------------------------------------------
    def _on_close(self):
        self._running = False
        time.sleep(0.1)
        self.destroy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="EyeGaze")
    p.add_argument("--model", default=None)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", default=None)
    args = p.parse_args()

    model_path = args.model
    if model_path is None:
        ckpts = sorted(Path("checkpoints").glob("**/best*.pt"))
        if not ckpts:
            print("[gui] No checkpoint found. Provide --model path.")
            raise SystemExit(1)
        model_path = str(ckpts[-1])

    app = GazeApp(
        model_path=model_path,
        camera_id=args.camera,
        video_path=args.video,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
