
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

import numpy as np
import soundfile as sf
from scipy.signal import spectrogram as sp_spectrogram, firwin, hilbert, savgol_filter

from PyQt5 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

# --- GPU: PyTorch (opcjonalnie) ---
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    TORCH_DEV = torch.device("mps" if HAS_MPS else "cpu")
except Exception:
    HAS_TORCH = False
    HAS_MPS = False
    TORCH_DEV = None

APP_TITLE = "SpectroBand (krótkie zdarzenia: STFT->CPU/fallback Hilbert; GPU MPS)"
DEFAULT_WINDOW_SEC = 60.0
DEFAULT_STEP_SEC = 60.0
DEFAULT_LO_HZ = 49000.0
DEFAULT_HI_HZ = 52000.0
DEFAULT_ORDER = 4
DEFAULT_FS_TARGET = 192000
DEFAULT_NPERSEG = 8192
DEFAULT_OVERLAP = 0.75
DEFAULT_FMAX_PLOT = 55000.0
DEFAULT_WAVE_MAXPOINTS = 40000
DEFAULT_ROW_HEIGHT_PX = 140
DEFAULT_MIN_EVENT_MS = 5.0
DEFAULT_FREQ_SMOOTH_MS = 3.0
DEFAULT_FREQ_STEP_MS = 2.0
DEFAULT_EVENT_BAND_ALPHA = 0.18
EPS = 1e-12


def human_time(s: float) -> str:
    if s < 0: s = 0.0
    h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
    if h > 0: return f"{h:d}:{m:02d}:{sec:05.2f}"
    return f"{m:d}:{sec:05.2f}"


def decimate_for_plot(y: np.ndarray, fs: int, start_s: float, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    n = y.size
    if n <= 2 or n <= max_points:
        t = start_s + np.arange(n, dtype=np.float32) / float(fs)
        return t, y.astype(np.float32, copy=False)
    step = int(np.ceil(n / float(max_points)))
    idx = np.arange(0, n, step, dtype=int)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    t = start_s + idx.astype(np.float32) / float(fs)
    y_dec = y[idx].astype(np.float32, copy=False)
    return t, y_dec


def decimate_visible(y: np.ndarray, fs: int, start_s: float, x0: float, x1: float, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if x1 <= x0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    n = y.size
    i0 = max(0, int(np.floor((x0 - start_s) * fs)))
    i1 = min(n, int(np.ceil((x1 - start_s) * fs)))
    if i1 - i0 <= 1:
        t = start_s + (np.arange(i0, i1, dtype=np.float32) / float(fs))
        return t, y[i0:i1].astype(np.float32, copy=False)
    y_sub = y[i0:i1]
    t0 = start_s + i0 / float(fs)
    return decimate_for_plot(y_sub, fs, t0, max_points)


def normalize_signal(x: np.ndarray, mode: str = "peak", target_rms: float = 1.0) -> np.ndarray:
    if x.size == 0: return x
    if mode == "rms":
        rms = float(np.sqrt(np.mean(x.astype(np.float64)**2)))
        return (x * (target_rms / rms)).astype(np.float32, copy=False) if rms > 0 else x.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(x)))
    return (x / peak).astype(np.float32, copy=False) if peak > 0 else x.astype(np.float32, copy=False)


def detect_events(y: np.ndarray, fs: int, thr: float, mode: str, min_event_s: float) -> List[Dict]:
    if y.size == 0: return []
    mask = np.abs(y) >= thr if mode == "abs" else (y >= thr)
    events = []; inside = False; start_idx = 0
    for i, m in enumerate(mask):
        if m and not inside:
            inside = True; start_idx = i
        elif not m and inside:
            inside = False; end_idx = i - 1
            dur = (end_idx - start_idx + 1) / float(fs)
            if dur >= min_event_s:
                seg = y[start_idx:end_idx+1]
                peak = float(np.max(np.abs(seg)) if mode == "abs" else np.max(seg))
                events.append({"start_idx": int(start_idx), "end_idx": int(end_idx),
                               "start_s": float(start_idx / float(fs)), "end_s": float((end_idx + 1) / float(fs)),
                               "duration_s": float(dur), "peak": peak})
    if inside:
        end_idx = y.size - 1; dur = (end_idx - start_idx + 1) / float(fs)
        if dur >= min_event_s:
            seg = y[start_idx:end_idx+1]
            peak = float(np.max(np.abs(seg)) if mode == "abs" else np.max(seg))
            events.append({"start_idx": int(start_idx), "end_idx": int(end_idx),
                           "start_s": float(start_idx / float(fs)), "end_s": float((end_idx + 1) / float(fs)),
                           "duration_s": float(dur), "peak": peak})
    return events


# ---------- GPU utils ----------
def torch_window(window_spec: Union[str, tuple], nperseg: int) -> Optional['torch.Tensor']:
    if not HAS_TORCH:
        return None
    ident = window_spec if isinstance(window_spec, str) else window_spec[0]
    param = None if isinstance(window_spec, str) else window_spec[1]
    if ident == "hann":
        w = torch.hann_window(nperseg, periodic=True, dtype=torch.float32, device=TORCH_DEV)
    elif ident == "hamming":
        w = torch.hamming_window(nperseg, periodic=True, dtype=torch.float32, device=TORCH_DEV)
    elif ident == "blackman":
        w = torch.blackman_window(nperseg, periodic=True, dtype=torch.float32, device=TORCH_DEV)
    elif ident == "bartlett":
        n = torch.arange(nperseg, device=TORCH_DEV, dtype=torch.float32)
        w = 1.0 - torch.abs((n - (nperseg - 1)/2.0) / ((nperseg - 1)/2.0))
    elif ident == "kaiser":
        beta = float(param) if param is not None else 8.6
        w = torch.kaiser_window(nperseg, beta=beta, periodic=True, dtype=torch.float32, device=TORCH_DEV)
    elif ident == "boxcar":
        w = torch.ones(nperseg, dtype=torch.float32, device=TORCH_DEV)
    else:
        try:
            import scipy.signal as sig
            if ident == "blackmanharris":
                w_np = sig.get_window("blackmanharris", nperseg, fftbins=True).astype(np.float32)
            elif ident == "flattop":
                w_np = sig.get_window("flattop", nperseg, fftbins=True).astype(np.float32)
            elif ident == "tukey":
                alpha = float(param) if param is not None else 0.5
                w_np = sig.get_window(("tukey", alpha), nperseg, fftbins=True).astype(np.float32)
            else:
                return None
            w = torch.from_numpy(w_np).to(TORCH_DEV)
        except Exception:
            return None
    return w


def stft_gpu_mps(x: np.ndarray, fs: int, nperseg: int, noverlap: int, window_spec: Union[str, tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (HAS_TORCH and HAS_MPS):
        raise RuntimeError("MPS niedostępny")
    hop = nperseg - noverlap
    if hop <= 0: hop = max(1, nperseg // 4)
    x_t = torch.from_numpy(np.ascontiguousarray(x)).to(TORCH_DEV, dtype=torch.float32)
    w_t = torch_window(window_spec, nperseg)
    if w_t is None:
        w_t = torch.hann_window(nperseg, periodic=True, dtype=torch.float32, device=TORCH_DEV)
    Z = torch.stft(x_t, n_fft=nperseg, hop_length=hop, win_length=nperseg, window=w_t, center=False, return_complex=True)
    S = torch.abs(Z).clamp_min_(1e-12)
    Sdb = (20.0 * torch.log10(S)).to("cpu").numpy()
    f = np.fft.rfftfreq(nperseg, d=1.0/float(fs)).astype(np.float32)
    n_frames = Sdb.shape[1]; t = (np.arange(n_frames, dtype=np.float32) * hop) / float(fs)
    return f, t, Sdb


def fir_bandpass_gpu(x: np.ndarray, fs: int, f_lo: float, f_hi: float, numtaps: int, window_spec: Union[str, tuple]) -> np.ndarray:
    if not (HAS_TORCH and HAS_MPS):
        raise RuntimeError("MPS niedostępny")
    nyq = fs * 0.5
    f_lo = max(0.0, min(f_lo, nyq * 0.999))
    f_hi = max(0.0, min(f_hi, nyq * 0.999))
    if f_hi <= f_lo or f_lo <= 0.0:
        return x.astype(np.float32, copy=False)
    ident = window_spec if isinstance(window_spec, str) else window_spec[0]
    param = None if isinstance(window_spec, str) else window_spec[1]
    if ident == "kaiser":
        beta = float(param) if param is not None else 8.6
        win_np = ("kaiser", beta)
    elif ident == "tukey":
        alpha = float(param) if param is not None else 0.5
        win_np = ("tukey", alpha)
    else:
        win_np = ident
    h = firwin(numtaps, [f_lo, f_hi], pass_zero=False, fs=fs, window=win_np).astype(np.float32)
    h_t = torch.from_numpy(h[::-1].copy()).to(TORCH_DEV, dtype=torch.float32).view(1, 1, -1)
    x_t = torch.from_numpy(np.ascontiguousarray(x)).to(TORCH_DEV, dtype=torch.float32).view(1, 1, -1)
    pad = (h_t.shape[-1] // 2)
    y_t = F.conv1d(x_t, h_t, padding=pad)
    y = y_t.view(-1).to("cpu").numpy().astype(np.float32, copy=False)
    return y


# ---------- Canvas'y ----------
class MplCanvasPairsN(FigureCanvas):
    def __init__(self, parent=None, dpi=100):
        fig = Figure(figsize=(12, 7), dpi=dpi)
        super().__init__(fig); self.setParent(parent)
        self.axes: List = []; self.n_pairs = 0; self.row_height_px = DEFAULT_ROW_HEIGHT_PX
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.rebuild_pairs(1, self.row_height_px)

    def rebuild_pairs(self, n_pairs: int, row_height_px: int):
        n_pairs = max(1, int(n_pairs)); self.n_pairs = n_pairs
        self.row_height_px = int(max(80, min(400, row_height_px)))
        self.figure.clf(); axes = []
        for i in range(n_pairs * 2):
            ax = self.figure.add_subplot(n_pairs * 2, 1, i + 1)
            axes.append(ax)
        self.axes = axes
        dpi = self.figure.get_dpi()
        total_height_px = int(self.row_height_px * (2 * n_pairs))
        total_height_in = max(4.0, min(30.0, total_height_px / float(dpi)))
        w_in, _ = self.figure.get_size_inches()
        self.figure.set_size_inches(w_in, total_height_in, forward=True)
        self.setMinimumHeight(int(total_height_px * 1.02))
        self.updateGeometry()


class MplCanvasFreqN(FigureCanvas):
    def __init__(self, parent=None, dpi=100):
        fig = Figure(figsize=(12, 6), dpi=dpi)
        super().__init__(fig); self.setParent(parent)
        self.axes: List = []; self.n_axes = 0; self.row_height_px = DEFAULT_ROW_HEIGHT_PX
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.rebuild_axes(1, self.row_height_px)

    def rebuild_axes(self, n_axes: int, row_height_px: int):
        n_axes = max(1, int(n_axes)); self.n_axes = n_axes
        self.row_height_px = int(max(80, min(400, row_height_px)))
        self.figure.clf(); axes = []
        for i in range(n_axes):
            ax = self.figure.add_subplot(n_axes, 1, i + 1)
            axes.append(ax)
        self.axes = axes
        dpi = self.figure.get_dpi()
        total_height_px = int(self.row_height_px * n_axes)
        total_height_in = max(3.0, min(30.0, total_height_px / float(dpi)))
        w_in, _ = self.figure.get_size_inches()
        self.figure.set_size_inches(w_in, total_height_in, forward=True)
        self.setMinimumHeight(int(total_height_px * 1.02))
        self.updateGeometry()


# ---------- GUI ----------
class AudioSlot(QtCore.QObject):
    changed = QtCore.pyqtSignal()
    def __init__(self, slot_index: int):
        super().__init__()
        self.index = slot_index
        self.path: Optional[Path] = None
        self.samplerate: Optional[int] = None
        self.frames: Optional[int] = None
        self.channels: Optional[int] = None
        self.duration: Optional[float] = None
        self.channel_mode: str = "mix"
        self.display_name: str = f"Slot {slot_index+1}"

    def load(self, path: Path):
        info = sf.info(str(path))
        self.path = path
        self.samplerate = int(info.samplerate)
        self.frames = int(info.frames)
        self.channels = int(info.channels)
        self.duration = self.frames / float(self.samplerate) if self.samplerate else 0.0
        self.channel_mode = "mix" if self.channels and self.channels > 1 else "0"
        self.display_name = path.name
        self.changed.emit()

    def clear(self):
        self.path = None; self.samplerate = None; self.frames = None; self.channels = None; self.duration = None
        self.channel_mode = "mix"; self.display_name = f"Slot {self.index+1}"; self.changed.emit()

    def read_segment(self, start_s: float, length_s: float) -> Tuple[Optional[np.ndarray], Optional[int]]:
        if not self.path or self.samplerate is None or self.frames is None: return None, None
        fs = self.samplerate; n_total = self.frames
        start_s = max(0.0, start_s); length_s = max(0.01, length_s)
        start_frame = int(round(start_s * fs)); stop_frame = start_frame + int(round(length_s * fs))
        start_frame = max(0, min(start_frame, n_total)); stop_frame = max(start_frame, min(stop_frame, n_total))
        if stop_frame - start_frame <= 0: return None, fs
        data, _ = sf.read(str(self.path), start=start_frame, stop=stop_frame, dtype="float32", always_2d=True)
        if data.shape[1] == 1: x = data[:, 0]
        else:
            if self.channel_mode == "mix": x = np.mean(data, axis=1)
            else:
                try:
                    ch = int(self.channel_mode); ch = max(0, min(ch, data.shape[1]-1)); x = data[:, ch]
                except Exception: x = np.mean(data, axis=1)
        return x.astype(np.float32, copy=False), fs


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(640, 420); self.resize(1280, 900)
        self.slots = [AudioSlot(i) for i in range(4)]
        for s in self.slots: s.changed.connect(self._on_any_file_changed)

        # --- central: TabWidget ---
        self.tabs = QtWidgets.QTabWidget(self)

        # Tab 1: sygnały + spektrogramy
        self.canvas_pairs = MplCanvasPairsN(self)
        self.navbar_pairs = NavToolbar(self.canvas_pairs, self)
        self.scroll_pairs = QtWidgets.QScrollArea(); self.scroll_pairs.setWidgetResizable(True)
        plot_host1 = QtWidgets.QWidget(); plot_layout1 = QtWidgets.QVBoxLayout(plot_host1); plot_layout1.setContentsMargins(0,0,0,0)
        plot_layout1.addWidget(self.navbar_pairs); plot_layout1.addWidget(self.canvas_pairs)
        self.scroll_pairs.setWidget(plot_host1)
        tab1 = QtWidgets.QWidget(); v1 = QtWidgets.QVBoxLayout(tab1); v1.setContentsMargins(0,0,0,0); v1.addWidget(self.scroll_pairs)
        self.tabs.addTab(tab1, "Sygnały + spektrogramy")

        # Tab 2: tory częstotliwości (zdarzenia)
        self.canvas_freq = MplCanvasFreqN(self)
        self.navbar_freq = NavToolbar(self.canvas_freq, self)
        self.scroll_freq = QtWidgets.QScrollArea(); self.scroll_freq.setWidgetResizable(True)
        plot_host2 = QtWidgets.QWidget(); plot_layout2 = QtWidgets.QVBoxLayout(plot_host2); plot_layout2.setContentsMargins(0,0,0,0)
        plot_layout2.addWidget(self.navbar_freq); plot_layout2.addWidget(self.canvas_freq)
        self.scroll_freq.setWidget(plot_host2)
        tab2 = QtWidgets.QWidget(); v2 = QtWidgets.QVBoxLayout(tab2); v2.setContentsMargins(0,0,0,0); v2.addWidget(self.scroll_freq)
        self.tabs.addTab(tab2, "Tory częstotliwości (zdarzenia)")

        self.setCentralWidget(self.tabs)

        # --- panel sterowania (scroll + toolbox) ---
        self.control_panel = self._build_controls()
        self.control_scroll = QtWidgets.QScrollArea(); self.control_scroll.setWidgetResizable(True); self.control_scroll.setWidget(self.control_panel)
        self.dock = QtWidgets.QDockWidget("Sterowanie", self); self.dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.dock.setWidget(self.control_scroll); self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock)
        self.dock.setMinimumWidth(320); self.dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)

        # skróty
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self, self.on_prev)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, self.on_next)
        QtWidgets.QShortcut(QtGui.QKeySequence("R"), self, self.redraw_all)

        # colorbar (dotyczy tylko Tab 1)
        self._cbar = None; self._cbar_ax = None

        # cache do zoom‑redecymacji
        self._axis_slot_map: Dict[object, int] = {}
        self._line_handles: Dict[int, Dict[str, object]] = {}
        self._cache_pairs_data: Dict[int, Dict[str, object]] = {}

        self._last_events_by_slot: Dict[int, List[Dict]] = {}
        self._freq_tracks_by_slot: Dict[int, List[Dict]] = {}

        self.redraw_all()

    # ----- panel sterowania -----
    def _build_controls(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self); vbox = QtWidgets.QVBoxLayout(panel)
        toolbox = QtWidgets.QToolBox(); vbox.addWidget(toolbox)

        # Pliki
        page_files = QtWidgets.QWidget(); files_group = QtWidgets.QGridLayout(page_files)
        self.file_labels = []; self.fs_labels = []; self.chan_boxes = []
        for i in range(4):
            btn = QtWidgets.QPushButton(f"Wybierz plik {i+1}…"); btn.clicked.connect(lambda _, idx=i: self.choose_file(idx))
            lbl = QtWidgets.QLabel("— brak —"); lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse); lbl.setMinimumWidth(220)
            fs_lbl = QtWidgets.QLabel("fs: — / kanały: — / długość: —")
            chan_box = QtWidgets.QComboBox(); chan_box.addItem("Mono (mix)", "mix"); chan_box.setEnabled(False); chan_box.currentIndexChanged.connect(self._on_channel_changed)
            clr = QtWidgets.QPushButton("Wyczyść"); clr.clicked.connect(lambda _, idx=i: self.clear_file(idx))
            row = i
            files_group.addWidget(btn, row, 0); files_group.addWidget(lbl, row, 1, 1, 2); files_group.addWidget(fs_lbl, row, 3)
            files_group.addWidget(QtWidgets.QLabel("Kanał:"), row, 4); files_group.addWidget(chan_box, row, 5); files_group.addWidget(clr, row, 6)
            self.file_labels.append(lbl); self.fs_labels.append(fs_lbl); self.chan_boxes.append(chan_box)
        toolbox.addItem(page_files, "Pliki WAV")

        # Czas
        page_time = QtWidgets.QWidget(); tg = QtWidgets.QGridLayout(page_time)
        self.spin_start = QtWidgets.QDoubleSpinBox(); self.spin_start.setDecimals(3); self.spin_start.setRange(0, 999999); self.spin_start.setSingleStep(1.0); self.spin_start.setValue(0.0); self.spin_start.valueChanged.connect(self._on_start_changed)
        self.spin_window = QtWidgets.QDoubleSpinBox(); self.spin_window.setDecimals(3); self.spin_window.setRange(0.1, 3600*24); self.spin_window.setSingleStep(1.0); self.spin_window.setValue(DEFAULT_WINDOW_SEC); self.spin_window.valueChanged.connect(self._on_window_changed)
        self.spin_step = QtWidgets.QDoubleSpinBox(); self.spin_step.setDecimals(3); self.spin_step.setRange(0.01, 3600*24); self.spin_step.setSingleStep(1.0); self.spin_step.setValue(DEFAULT_STEP_SEC)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(0, 1000); self.slider.valueChanged.connect(self._on_slider_changed)
        self.lbl_pos = QtWidgets.QLabel("Start: 0.00 s"); self.lbl_total = QtWidgets.QLabel("Dostępna oś czasu: —")
        btn_prev = QtWidgets.QPushButton("← Wstecz"); btn_prev.clicked.connect(self.on_prev); btn_next = QtWidgets.QPushButton("Naprzód →"); btn_next.clicked.connect(self.on_next)
        tg.addWidget(QtWidgets.QLabel("Start [s]:"), 0, 0); tg.addWidget(self.spin_start, 0, 1)
        tg.addWidget(QtWidgets.QLabel("Okno [s]:"), 0, 2); tg.addWidget(self.spin_window, 0, 3)
        tg.addWidget(QtWidgets.QLabel("Krok [s]:"), 0, 4); tg.addWidget(self.spin_step, 0, 5)
        tg.addWidget(btn_prev, 1, 0, 1, 3); tg.addWidget(btn_next, 1, 3, 1, 3)
        tg.addWidget(self.slider, 2, 0, 1, 6); tg.addWidget(self.lbl_pos, 3, 0, 1, 3); tg.addWidget(self.lbl_total, 3, 3, 1, 3)
        toolbox.addItem(page_time, "Czas i nawigacja")

        # Filtr (CPU)
        page_filt = QtWidgets.QWidget(); fg2 = QtWidgets.QGridLayout(page_filt)
        self.check_filter = QtWidgets.QCheckBox("Zastosuj filtr"); self.check_filter.setChecked(True); self.check_filter.toggled.connect(self.redraw_all)
        self.spin_lo = QtWidgets.QDoubleSpinBox(); self.spin_lo.setRange(0.0, 1_000_000.0); self.spin_lo.setDecimals(1); self.spin_lo.setValue(DEFAULT_LO_HZ); self.spin_lo.valueChanged.connect(self.redraw_all)
        self.spin_hi = QtWidgets.QDoubleSpinBox(); self.spin_hi.setRange(1.0, 1_000_000.0); self.spin_hi.setDecimals(1); self.spin_hi.setValue(DEFAULT_HI_HZ); self.spin_hi.valueChanged.connect(self.redraw_all)
        self.spin_order = QtWidgets.QSpinBox(); self.spin_order.setRange(1, 12); self.spin_order.setValue(DEFAULT_ORDER); self.spin_order.valueChanged.connect(self.redraw_all)
        self.check_zero_phase = QtWidgets.QCheckBox("Zero‑phase (filtfilt)"); self.check_zero_phase.setChecked(True); self.check_zero_phase.toggled.connect(self.redraw_all)
        fg2.addWidget(self.check_filter, 0, 0, 1, 2)
        fg2.addWidget(QtWidgets.QLabel("Dolna [Hz]:"), 1, 0); fg2.addWidget(self.spin_lo, 1, 1)
        fg2.addWidget(QtWidgets.QLabel("Górna [Hz]:"), 2, 0); fg2.addWidget(self.spin_hi, 2, 1)
        fg2.addWidget(QtWidgets.QLabel("Rząd:"), 3, 0); fg2.addWidget(self.spin_order, 3, 1)
        fg2.addWidget(self.check_zero_phase, 4, 0, 1, 2)
        toolbox.addItem(page_filt, "Filtr (CPU, Butterworth)")

        # Sygnał / Progi / Pasma
        page_sig = QtWidgets.QWidget(); sg = QtWidgets.QGridLayout(page_sig)
        self.check_plot_raw = QtWidgets.QCheckBox("Rysuj sygnał surowy"); self.check_plot_raw.setChecked(False); self.check_plot_raw.toggled.connect(self.redraw_all)
        self.check_plot_filt = QtWidgets.QCheckBox("Rysuj sygnał po filtrze"); self.check_plot_filt.setChecked(True); self.check_plot_filt.toggled.connect(self.redraw_all)
        self.spin_wave_maxpts = QtWidgets.QSpinBox(); self.spin_wave_maxpts.setRange(500, 20000); self.spin_wave_maxpts.setSingleStep(500); self.spin_wave_maxpts.setValue(DEFAULT_WAVE_MAXPOINTS); self.spin_wave_maxpts.valueChanged.connect(self.redraw_all)
        self.check_norm = QtWidgets.QCheckBox("Normalizuj wykresy sygnału"); self.check_norm.setChecked(True); self.check_norm.toggled.connect(self.redraw_all)
        self.check_norm_only_filt = QtWidgets.QCheckBox("Normalizuj tylko po filtrze (próg też)"); self.check_norm_only_filt.setChecked(False); self.check_norm_only_filt.toggled.connect(self._on_norm_only_filt_toggled)
        self.combo_norm_mode = QtWidgets.QComboBox(); self.combo_norm_mode.addItem("Szczytowa (|x|→1)", "peak"); self.combo_norm_mode.addItem("RMS = 1.0", "rms"); self.combo_norm_mode.currentIndexChanged.connect(self.redraw_all)
        self.spin_norm_rms = QtWidgets.QDoubleSpinBox(); self.spin_norm_rms.setRange(0.01, 10.0); self.spin_norm_rms.setDecimals(2); self.spin_norm_rms.setValue(1.0); self.spin_norm_rms.valueChanged.connect(self.redraw_all)
        self.check_thr = QtWidgets.QCheckBox("Pokaż próg i przekroczenia"); self.check_thr.setChecked(True); self.check_thr.toggled.connect(self.redraw_all)
        self.spin_thr = QtWidgets.QDoubleSpinBox(); self.spin_thr.setRange(0.0, 5.0); self.spin_thr.setDecimals(3); self.spin_thr.setValue(0.5); self.spin_thr.valueChanged.connect(self.redraw_all)
        self.combo_thr_mode = QtWidgets.QComboBox(); self.combo_thr_mode.addItem("Dwustronny: |x| ≥ T", "abs"); self.combo_thr_mode.addItem("Jednostronny: x ≥ +T", "pos"); self.combo_thr_mode.currentIndexChanged.connect(self.redraw_all)
        self.combo_thr_source = QtWidgets.QComboBox(); self.combo_thr_source.addItem("Po filtrze", "filt"); self.combo_thr_source.addItem("Surowy", "raw"); self.combo_thr_source.currentIndexChanged.connect(self.redraw_all)
        self.spin_min_event = QtWidgets.QDoubleSpinBox(); self.spin_min_event.setRange(0.0, 1000.0); self.spin_min_event.setDecimals(1); self.spin_min_event.setValue(DEFAULT_MIN_EVENT_MS)

        # Pasma zdarzeń (nowe)
        self.check_fill_events = QtWidgets.QCheckBox("Wypełnij pasmami okna > progu (Tab 1)"); self.check_fill_events.setChecked(True); self.check_fill_events.toggled.connect(self.redraw_all)
        self.spin_fill_alpha = QtWidgets.QDoubleSpinBox(); self.spin_fill_alpha.setRange(0.02, 0.7); self.spin_fill_alpha.setDecimals(2); self.spin_fill_alpha.setSingleStep(0.02); self.spin_fill_alpha.setValue(DEFAULT_EVENT_BAND_ALPHA); self.spin_fill_alpha.valueChanged.connect(self.redraw_all)
        self.combo_fill_color = QtWidgets.QComboBox()
        self.combo_fill_color.addItem("Zielony (tab:green)", "tab:green")
        self.combo_fill_color.addItem("Pomarańcz (tab:orange)", "tab:orange")
        self.combo_fill_color.addItem("Niebieski (tab:blue)", "tab:blue")
        self.combo_fill_color.addItem("Czerwony (tab:red)", "tab:red")
        self.combo_fill_color.currentIndexChanged.connect(self.redraw_all)

        row = 0
        sg.addWidget(self.check_plot_raw, row, 0, 1, 2); row += 1
        sg.addWidget(self.check_plot_filt, row, 0, 1, 2); row += 1
        sg.addWidget(QtWidgets.QLabel("Max punktów/wykres:"), row, 0); sg.addWidget(self.spin_wave_maxpts, row, 1); row += 1
        sg.addWidget(self.check_norm, row, 0, 1, 2); row += 1
        sg.addWidget(self.check_norm_only_filt, row, 0, 1, 2); row += 1
        sg.addWidget(QtWidgets.QLabel("Tryb normalizacji:"), row, 0); sg.addWidget(self.combo_norm_mode, row, 1); row += 1
        sg.addWidget(QtWidgets.QLabel("Docelowe RMS:"), row, 0); sg.addWidget(self.spin_norm_rms, row, 1); row += 1
        sg.addWidget(self.check_thr, row, 0, 1, 2); row += 1
        sg.addWidget(QtWidgets.QLabel("Próg T:"), row, 0); sg.addWidget(self.spin_thr, row, 1); row += 1
        sg.addWidget(QtWidgets.QLabel("Tryb progu:"), row, 0); sg.addWidget(self.combo_thr_mode, row, 1); row += 1
        sg.addWidget(QtWidgets.QLabel("Źródło progu:"), row, 0); sg.addWidget(self.combo_thr_source, row, 1); row += 1
        sg.addWidget(QtWidgets.QLabel("Min. czas zdarzenia [ms]:"), row, 0); sg.addWidget(self.spin_min_event, row, 1); row += 1
        sg.addWidget(self.check_fill_events, row, 0, 1, 2); row += 1
        sg.addWidget(QtWidgets.QLabel("Przezroczystość pasma:"), row, 0); sg.addWidget(self.spin_fill_alpha, row, 1); row += 1
        sg.addWidget(QtWidgets.QLabel("Kolor pasma:"), row, 0); sg.addWidget(self.combo_fill_color, row, 1); row += 1
        toolbox.addItem(page_sig, "Sygnał / Progi / Pasma")

        # Spektrogram (Tab 1)
        page_spec = QtWidgets.QWidget(); spg = QtWidgets.QGridLayout(page_spec)
        self.spin_nperseg = QtWidgets.QSpinBox(); self.spin_nperseg.setRange(256, 65536); self.spin_nperseg.setSingleStep(256); self.spin_nperseg.setValue(DEFAULT_NPERSEG); self.spin_nperseg.valueChanged.connect(self.redraw_all)
        self.spin_overlap = QtWidgets.QDoubleSpinBox(); self.spin_overlap.setRange(0.0, 0.95); self.spin_overlap.setSingleStep(0.05); self.spin_overlap.setValue(DEFAULT_OVERLAP); self.spin_overlap.valueChanged.connect(self.redraw_all)
        self.combo_window = QtWidgets.QComboBox()
        self.combo_window.addItem("Hann", ("hann", None)); self.combo_window.addItem("Hamming", ("hamming", None))
        self.combo_window.addItem("Blackman", ("blackman", None)); self.combo_window.addItem("Blackman-Harris", ("blackmanharris", None))
        self.combo_window.addItem("Bartlett", ("bartlett", None)); self.combo_window.addItem("Flattop", ("flattop", None))
        self.combo_window.addItem("Boxcar (prostokątne)", ("boxcar", None)); self.combo_window.addItem("Kaiser (β)", ("kaiser", "beta")); self.combo_window.addItem("Tukey (α)", ("tukey", "alpha"))
        self.combo_window.currentIndexChanged.connect(self._on_window_changed)
        self.spin_beta = QtWidgets.QDoubleSpinBox(); self.spin_beta.setRange(0.0, 20.0); self.spin_beta.setDecimals(2); self.spin_beta.setValue(8.6); self.spin_beta.setEnabled(False); self.spin_beta.valueChanged.connect(self.redraw_all)
        self.spin_alpha = QtWidgets.QDoubleSpinBox(); self.spin_alpha.setRange(0.0, 1.0); self.spin_alpha.setDecimals(2); self.spin_alpha.setValue(0.5); self.spin_alpha.setEnabled(False); self.spin_alpha.valueChanged.connect(self.redraw_all)
        self.check_match_filter = QtWidgets.QCheckBox("Dopasuj zakres Y do filtra"); self.check_match_filter.setChecked(True); self.check_match_filter.toggled.connect(self.redraw_all)
        self.check_show_band = QtWidgets.QCheckBox("Oznacz pasmo filtra (linie + cień)"); self.check_show_band.setChecked(True); self.check_show_band.toggled.connect(self.redraw_all)
        self.spin_fmax = QtWidgets.QDoubleSpinBox(); self.spin_fmax.setRange(100.0, 1_000_000.0); self.spin_fmax.setDecimals(1); self.spin_fmax.setValue(DEFAULT_FMAX_PLOT); self.spin_fmax.valueChanged.connect(self.redraw_all)
        self.combo_cbar_pos = QtWidgets.QComboBox(); self.combo_cbar_pos.addItem("Colorbar: prawa (pionowy)", "right"); self.combo_cbar_pos.addItem("Colorbar: dół (poziomy)", "bottom"); self.combo_cbar_pos.currentIndexChanged.connect(self.redraw_all)
        self.check_auto_db = QtWidgets.QCheckBox("Auto‑skala dB (5–95 %)"); self.check_auto_db.setChecked(True); self.check_auto_db.toggled.connect(self.redraw_all)
        self.spin_vmin = QtWidgets.QDoubleSpinBox(); self.spin_vmin.setRange(-300.0, 100.0); self.spin_vmin.setDecimals(1); self.spin_vmin.setValue(-120.0); self.spin_vmin.valueChanged.connect(self.redraw_all)
        self.spin_vmax = QtWidgets.QDoubleSpinBox(); self.spin_vmax.setRange(-300.0, 100.0); self.spin_vmax.setDecimals(1); self.spin_vmax.setValue(-20.0); self.spin_vmax.valueChanged.connect(self.redraw_all)

        # GPU toggles
        self.lbl_gpu = QtWidgets.QLabel("GPU MPS: " + ("dostępny" if HAS_MPS else "niedostępny"))
        self.check_gpu_stft = QtWidgets.QCheckBox("Użyj GPU do STFT/spektrogramu"); self.check_gpu_stft.setChecked(HAS_MPS); self.check_gpu_stft.setEnabled(HAS_MPS); self.check_gpu_stft.toggled.connect(self.redraw_all)

        spg.addWidget(self.lbl_gpu, 0, 0, 1, 2)
        spg.addWidget(QtWidgets.QLabel("nperseg (FFT):"), 1, 0); spg.addWidget(self.spin_nperseg, 1, 1)
        spg.addWidget(QtWidgets.QLabel("Overlap:"), 2, 0); spg.addWidget(self.spin_overlap, 2, 1)
        spg.addWidget(QtWidgets.QLabel("Okno FFT:"), 3, 0); spg.addWidget(self.combo_window, 3, 1)
        spg.addWidget(QtWidgets.QLabel("β (Kaiser):"), 4, 0); spg.addWidget(self.spin_beta, 4, 1)
        spg.addWidget(QtWidgets.QLabel("α (Tukey):"), 5, 0); spg.addWidget(self.spin_alpha, 5, 1)
        spg.addWidget(self.check_match_filter, 6, 0, 1, 2)
        spg.addWidget(self.check_show_band, 7, 0, 1, 2)
        spg.addWidget(QtWidgets.QLabel("f_max rys. [Hz]:"), 8, 0); spg.addWidget(self.spin_fmax, 8, 1)
        spg.addWidget(self.combo_cbar_pos, 9, 0, 1, 2)
        spg.addWidget(self.check_auto_db, 10, 0, 1, 2)
        spg.addWidget(QtWidgets.QLabel("vmin [dB]:"), 11, 0); spg.addWidget(self.spin_vmin, 11, 1)
        spg.addWidget(QtWidgets.QLabel("vmax [dB]:"), 12, 0); spg.addWidget(self.spin_vmax, 12, 1)
        toolbox.addItem(page_spec, "Spektrogram (Tab 1)")

        # GPU / Akceleracja (FIR)
        page_gpu = QtWidgets.QWidget(); gg = QtWidgets.QGridLayout(page_gpu)
        self.check_gpu_fir = QtWidgets.QCheckBox("Filtruj na GPU (FIR beta) zamiast CPU (Butterworth)")
        self.check_gpu_fir.setChecked(False); self.check_gpu_fir.setEnabled(HAS_MPS); self.check_gpu_fir.toggled.connect(self.redraw_all)
        self.spin_fir_taps = QtWidgets.QSpinBox(); self.spin_fir_taps.setRange(129, 16385); self.spin_fir_taps.setSingleStep(128); self.spin_fir_taps.setValue(2049); self.spin_fir_taps.valueChanged.connect(self.redraw_all)
        gg.addWidget(QtWidgets.QLabel("Status GPU (MPS): " + ("dostępny" if HAS_MPS else "niedostępny")), 0, 0, 1, 2)
        gg.addWidget(self.check_gpu_fir, 1, 0, 1, 2)
        gg.addWidget(QtWidgets.QLabel("Długość FIR [tapy]:"), 2, 0); gg.addWidget(self.spin_fir_taps, 2, 1)
        toolbox.addItem(page_gpu, "GPU / Akceleracja")

        # Częstotliwość zdarzeń (Tab 2)
        page_freq = QtWidgets.QWidget(); fg = QtWidgets.QGridLayout(page_freq)
        self.check_freq_track = QtWidgets.QCheckBox("Wyznacz i rysuj częstotliwość zdarzeń"); self.check_freq_track.setChecked(True); self.check_freq_track.toggled.connect(self.redraw_all)
        self.combo_freq_method = QtWidgets.QComboBox(); self.combo_freq_method.addItem("Hilbert (dokładna, ciągła)", "hilbert"); self.combo_freq_method.addItem("STFT ridge (interpolacja)", "stft"); self.combo_freq_method.currentIndexChanged.connect(self.redraw_all)
        self.spin_freq_smooth = QtWidgets.QDoubleSpinBox(); self.spin_freq_smooth.setRange(0.0, 100.0); self.spin_freq_smooth.setDecimals(1); self.spin_freq_smooth.setValue(DEFAULT_FREQ_SMOOTH_MS); self.spin_freq_smooth.valueChanged.connect(self.redraw_all)
        self.spin_freq_step = QtWidgets.QDoubleSpinBox(); self.spin_freq_step.setRange(0.2, 200.0); self.spin_freq_step.setDecimals(1); self.spin_freq_step.setValue(DEFAULT_FREQ_STEP_MS); self.spin_freq_step.valueChanged.connect(self.redraw_all)
        self.combo_freq_time = QtWidgets.QComboBox(); self.combo_freq_time.addItem("Oś czasu: absolutna", "abs"); self.combo_freq_time.addItem("Oś czasu: względna (od startu zdarzenia)", "rel"); self.combo_freq_time.currentIndexChanged.connect(self.redraw_all)
        self.check_freq_clip_band = QtWidgets.QCheckBox("Ogranicz tor do pasma filtra"); self.check_freq_clip_band.setChecked(True); self.check_freq_clip_band.toggled.connect(self.redraw_all)
        self.btn_export_tracks = QtWidgets.QPushButton("Eksport torów f do CSV…"); self.btn_export_tracks.clicked.connect(self.export_freq_tracks_csv)

        fg.addWidget(self.check_freq_track, 0, 0, 1, 2)
        fg.addWidget(QtWidgets.QLabel("Metoda:"), 1, 0); fg.addWidget(self.combo_freq_method, 1, 1)
        fg.addWidget(QtWidgets.QLabel("Wygładzanie [ms]:"), 2, 0); fg.addWidget(self.spin_freq_smooth, 2, 1)
        fg.addWidget(QtWidgets.QLabel("Próbkowanie toru [ms]:"), 3, 0); fg.addWidget(self.spin_freq_step, 3, 1)
        fg.addWidget(self.combo_freq_time, 4, 0, 1, 2)
        fg.addWidget(self.check_freq_clip_band, 5, 0, 1, 2)
        fg.addWidget(self.btn_export_tracks, 6, 0, 1, 2)
        toolbox.addItem(page_freq, "Tory częstotliwości (Tab 2)")

        # Układ / Zapis
        page_layout = QtWidgets.QWidget(); lg = QtWidgets.QGridLayout(page_layout)
        self.spin_row_height = QtWidgets.QSpinBox(); self.spin_row_height.setRange(80, 400); self.spin_row_height.setSingleStep(10); self.spin_row_height.setValue(DEFAULT_ROW_HEIGHT_PX); self.spin_row_height.valueChanged.connect(self._on_row_height_changed)
        btn_refresh = QtWidgets.QPushButton("Odśwież (R)"); btn_refresh.clicked.connect(self.redraw_all)
        btn_save1 = QtWidgets.QPushButton("Zapisz rysunek Tab 1…"); btn_save1.clicked.connect(self.save_figure_pairs)
        btn_save2 = QtWidgets.QPushButton("Zapisz rysunek Tab 2…"); btn_save2.clicked.connect(self.save_figure_freq)
        lg.addWidget(QtWidgets.QLabel("Wysokość 1 osi [px]:"), 0, 0); lg.addWidget(self.spin_row_height, 0, 1)
        lg.addWidget(btn_refresh, 1, 0); lg.addWidget(btn_save1, 1, 1); lg.addWidget(btn_save2, 1, 2)
        toolbox.addItem(page_layout, "Układ / Zapis")

        vbox.addStretch(1); return panel

    # ----- pliki -----
    def choose_file(self, idx: int):
        dlg = QtWidgets.QFileDialog(self, f"Wybierz plik WAV dla slotu {idx+1}", str(Path.home()), "Pliki WAV (*.wav *.wave)")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            path = Path(dlg.selectedFiles()[0])
            try: self.slots[idx].load(path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Błąd wczytywania", f"Nie udało się wczytać pliku:\n{path}\n\n{e}"); return
            self._populate_slot_controls(idx); s = self.slots[idx]
            if s.samplerate and abs(s.samplerate - DEFAULT_FS_TARGET) > 1:
                QtWidgets.QMessageBox.warning(self, "Inne fs niż 192 kHz", f"Plik ma próbkowanie {s.samplerate} Hz (nie 192000 Hz).\nAplikacja i tak zadziała.")
            self._recalibrate_timeline(); self.redraw_all()

    def clear_file(self, idx: int):
        self.slots[idx].clear(); self._populate_slot_controls(idx); self._recalibrate_timeline(); self.redraw_all()

    def _populate_slot_controls(self, idx: int):
        s = self.slots[idx]; lbl = self.file_labels[idx]; fs_lbl = self.fs_labels[idx]; chan_box = self.chan_boxes[idx]
        if s.path is None:
            lbl.setText("— brak —"); fs_lbl.setText("fs: — / kanały: — / długość: —"); chan_box.clear(); chan_box.addItem("Mono (mix)", "mix"); chan_box.setEnabled(False)
        else:
            elided = lbl.fontMetrics().elidedText(str(s.path), QtCore.Qt.ElideMiddle, 260); lbl.setText(elided)
            dur = f"{s.duration:.3f} s" if s.duration is not None else "—"; fs_lbl.setText(f"fs: {s.samplerate} Hz  |  kanały: {s.channels}  |  długość: {dur}")
            chan_box.blockSignals(True); chan_box.clear()
            if s.channels and s.channels > 1:
                chan_box.addItem("Mono (mix)", "mix");  [chan_box.addItem(f"Ch {ch}", str(ch)) for ch in range(s.channels)]
            else: chan_box.addItem("Mono", "0")
            chan_box.setEnabled(True); idx_to_select = 0
            for i in range(chan_box.count()):
                if chan_box.itemData(i) == s.channel_mode: idx_to_select = i; break
            chan_box.setCurrentIndex(idx_to_select); chan_box.blockSignals(False)

    def _on_any_file_changed(self):
        for i in range(4): self._populate_slot_controls(i)
        self._recalibrate_timeline()

    def _on_channel_changed(self, _):
        for i, box in enumerate(self.chan_boxes):
            s = self.slots[i]
            if s.path is not None: s.channel_mode = box.currentData()
        self.redraw_all()

    # ----- czas -----
    def _recalibrate_timeline(self):
        durations = [s.duration for s in self.slots if s.duration]; total = max(durations) if durations else 0.0
        win = self.spin_window.value(); max_start = max(0.0, total - win)
        self.slider.blockSignals(True); self.slider.setRange(0, 1000)
        start = min(self.spin_start.value(), max_start); self.spin_start.setValue(start)
        val = int(round((start / max_start) * 1000)) if max_start > 0 else 0; self.slider.setValue(val)
        self.slider.blockSignals(False)
        self.lbl_total.setText(f"Dostępna oś czasu: 0 – {total:.2f} s  (okno {win:.2f} s)") if total > 0 else self.lbl_total.setText("Dostępna oś czasu: — (najpierw wczytaj pliki)")
        self.lbl_pos.setText(f"Start: {self.spin_start.value():.3f} s  ({human_time(self.spin_start.value())})")

    def _on_start_changed(self, _val: float):
        self.lbl_pos.setText(f"Start: {self.spin_start.value():.3f} s  ({human_time(self.spin_start.value())})"); self._recalibrate_timeline(); self.redraw_all()

    def _on_window_changed(self, _val: float):
        self._recalibrate_timeline(); self.redraw_all()

    def _on_slider_changed(self, val: int):
        durations = [s.duration for s in self.slots if s.duration]; total = max(durations) if durations else 0.0
        win = self.spin_window.value(); max_start = max(0.0, total - win)
        start = (val / 1000.0) * max_start if max_start > 0 else 0.0
        self.spin_start.blockSignals(True); self.spin_start.setValue(start); self.spin_start.blockSignals(False)
        self.lbl_pos.setText(f"Start: {start:.3f} s  ({human_time(start)})"); self.redraw_all()

    def on_prev(self): self.spin_start.setValue(max(0.0, self.spin_start.value() - self.spin_step.value()))
    def on_next(self):
        durations = [s.duration for s in self.slots if s.duration]; total = max(durations) if durations else 0.0
        win = self.spin_window.value(); max_start = max(0.0, total - win)
        self.spin_start.setValue(min(max_start, self.spin_start.value() + self.spin_step.value()))

    # ----- DSP CPU/GPU -----
    def _maybe_filter_cpu(self, x: np.ndarray, fs: int) -> np.ndarray:
        if not self.check_filter.isChecked(): return x
        from scipy.signal import sosfiltfilt, sosfilt, butter as _butter
        lo = self.spin_lo.value(); hi = self.spin_hi.value(); order = self.spin_order.value()
        nyq = fs * 0.5; lo = max(0.0, min(lo, nyq * 0.999)); hi = max(0.0, min(hi, nyq * 0.999))
        if hi <= lo or lo <= 0.0: return x
        wn = [lo / nyq, hi / nyq]; sos = _butter(order, wn, btype="bandpass", output="sos")
        y = sosfiltfilt(sos, x, axis=0) if self.check_zero_phase.isChecked() else sosfilt(sos, x, axis=0)
        return y.astype(np.float32, copy=False)

    def _maybe_filter(self, x: np.ndarray, fs: int) -> np.ndarray:
        if self.check_gpu_fir.isChecked() and HAS_MPS and self.check_filter.isChecked():
            try:
                taps = int(self.spin_fir_taps.value()); ident, param = self.combo_window.currentData()
                return fir_bandpass_gpu(x, fs, float(self.spin_lo.value()), float(self.spin_hi.value()), taps, (ident, param))
            except Exception:
                return self._maybe_filter_cpu(x, fs)
        else:
            return self._maybe_filter_cpu(x, fs)

    def _get_window_param(self):
        ident, param = self.combo_window.currentData()
        if ident == "kaiser": return ("kaiser", float(self.spin_beta.value()))
        if ident == "tukey":  return ("tukey", float(self.spin_alpha.value()))
        return ident

    def _compute_spec_db_cpu(self, x: np.ndarray, fs: int):
        nperseg = int(self.spin_nperseg.value()); overlap_ratio = float(self.spin_overlap.value())
        noverlap = int(max(0, min(nperseg-1, round(overlap_ratio * nperseg)))); win = self._get_window_param()
        f, t, Sxx = sp_spectrogram(x, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap, scaling="density", mode="magnitude")
        Sxx_db = 20.0 * np.log10(np.maximum(Sxx, EPS)); return f, t, Sxx_db

    def _compute_spec_db(self, x: np.ndarray, fs: int):
        if self.check_gpu_stft.isChecked() and HAS_MPS:
            try:
                nperseg = int(self.spin_nperseg.value()); overlap_ratio = float(self.spin_overlap.value())
                noverlap = int(max(0, min(nperseg-1, round(overlap_ratio * nperseg)))); win = self._get_window_param()
                return stft_gpu_mps(x, fs, nperseg, noverlap, win)
            except Exception:
                return self._compute_spec_db_cpu(x, fs)
        else:
            return self._compute_spec_db_cpu(x, fs)

    # ----- colorbar (Tab 1) -----
    def _clear_cbar(self):
        if self._cbar is not None:
            try: self._cbar.remove()
            except Exception: pass
            self._cbar = None
        if self._cbar_ax is not None:
            try: self._cbar_ax.remove()
            except Exception: pass
            self._cbar_ax = None

    def _create_cbar(self, mappable):
        pos = self.combo_cbar_pos.currentData()
        fig = self.canvas_pairs.figure
        self._clear_cbar()
        sp = fig.subplotpars
        if pos == "right":
            w = 0.02; x0 = min(0.98 - w, sp.right + 0.01); y0 = sp.bottom; h = sp.top - sp.bottom
            cax = fig.add_axes([x0, y0, w, h]); self._cbar = fig.colorbar(mappable, cax=cax, orientation="vertical")
        else:
            h = 0.03; y0 = max(0.02, sp.bottom - h - 0.01); x0 = sp.left; w = sp.right - sp.left
            cax = fig.add_axes([x0, y0, w, h]); self._cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        self._cbar.set_label("Amplituda [dB]"); self._cbar_ax = cax

    # ----- częstotliwość: metody -----
    def _inst_freq_hilbert(self, y: np.ndarray, fs: int, smooth_ms: float) -> np.ndarray:
        if y.size == 0: return np.array([], dtype=np.float32)
        z = hilbert(y.astype(np.float64, copy=False))
        phi = np.unwrap(np.angle(z)); dphi = np.gradient(phi)
        fi = (fs / (2.0 * np.pi)) * dphi
        if smooth_ms > 0.0:
            w = int(round((smooth_ms / 1000.0) * fs))
            if w % 2 == 0: w += 1
            if w >= 5:
                try: fi = savgol_filter(fi, window_length=w, polyorder=2, mode="interp")
                except Exception: pass
        return fi.astype(np.float32, copy=False)

    def _freq_track_stft_local(self, y: np.ndarray, fs: int, smooth_ms: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        STFT ridge on *event segment*.
        Robust for very short segments:
        - uses CPU spectrogram with nperseg <= len(y) (avoids GPU no-frame issue),
        - if <2 frames => AUTO-fallback to Hilbert instantaneous frequency.
        """
        n_full = y.size
        if n_full == 0:
            return np.array([]), np.array([])

        nperseg_global = int(self.spin_nperseg.value())
        nloc = min(max(64, n_full), nperseg_global)  # at least 64 samples
        win = self._get_window_param()
        noverlap = int(max(0, min(nloc - 1, round(float(self.spin_overlap.value()) * nloc))))

        f, t, Sxx = sp_spectrogram(y, fs=fs, window=win, nperseg=nloc, noverlap=noverlap,
                                   scaling="density", mode="magnitude")
        if f.size == 0 or t.size == 0:
            fi = self._inst_freq_hilbert(y, fs, smooth_ms)
            tt = np.arange(fi.size, dtype=np.float64) / float(fs)
            return tt.astype(np.float32, copy=False), fi.astype(np.float32, copy=False)

        S = np.maximum(Sxx, 1e-24)

        nyq = fs * 0.5
        lo = float(self.spin_lo.value()); hi = float(self.spin_hi.value())
        use_band = self.check_filter.isChecked() and self.check_freq_clip_band.isChecked()
        if use_band:
            lo = max(0.0, min(lo, nyq * 0.999)); hi = max(0.0, min(hi, nyq * 0.999))
            mask_f = (f >= lo) & (f <= hi)
        else:
            fmax_plot = float(self.spin_fmax.value()); mask_f = f <= min(fmax_plot, nyq)
        f_use = f[mask_f]; S_use = S[mask_f, :]
        if f_use.size == 0:
            fi = self._inst_freq_hilbert(y, fs, smooth_ms)
            tt = np.arange(fi.size, dtype=np.float64) / float(fs)
            return tt.astype(np.float32, copy=False), fi.astype(np.float32, copy=False)

        def parabolic_interp(log_mag: np.ndarray, k: int) -> float:
            if k <= 0 or k >= log_mag.size - 1: return 0.0
            a = log_mag[k-1]; b = log_mag[k]; c = log_mag[k+1]
            denom = (a - 2*b + c)
            if abs(denom) < 1e-12: return 0.0
            return 0.5 * (a - c) / denom

        t_rel = t.astype(np.float64, copy=False)
        if t_rel.size < 2:
            fi = self._inst_freq_hilbert(y, fs, smooth_ms)
            tt = np.arange(fi.size, dtype=np.float64) / float(fs)
            return tt.astype(np.float32, copy=False), fi.astype(np.float32, copy=False)

        f_track = np.zeros(t_rel.size, dtype=np.float32)
        for j in range(t_rel.size):
            col = S_use[:, j]
            if col.size == 0:
                f_track[j] = np.nan; continue
            k = int(np.argmax(col))
            log_mag = np.log(np.maximum(col, 1e-24))
            delta = parabolic_interp(log_mag, k)
            df = f_use[1] - f_use[0] if f_use.size > 1 else 0.0
            f_est = f_use[0] + (k + delta) * df
            f_track[j] = float(f_est)

        if smooth_ms > 0.0 and t_rel.size > 3:
            dt = float(np.median(np.diff(t_rel))) if t_rel.size > 1 else 0.0
            if dt > 0:
                win_pts = int(round((smooth_ms/1000.0) / dt))
                if win_pts % 2 == 0: win_pts += 1
                if win_pts >= 5:
                    try:
                        f_track = savgol_filter(f_track, window_length=win_pts, polyorder=2, mode="interp").astype(np.float32, copy=False)
                    except Exception:
                        pass

        return t_rel.astype(np.float32, copy=False), f_track

    # ----- rysowanie Tab 1 + przygotowanie torów -----
    def redraw_pairs(self):
        loaded = self._loaded_slots(); n_pairs = max(1, len(loaded))
        if self.canvas_pairs.n_pairs != n_pairs or self.canvas_pairs.row_height_px != self.spin_row_height.value():
            self.canvas_pairs.rebuild_pairs(n_pairs, self.spin_row_height.value())

        axes = self.canvas_pairs.axes
        for ax in axes: ax.clear()
        self._clear_cbar()
        self._last_events_by_slot = {}
        self._freq_tracks_by_slot = {}
        self._axis_slot_map.clear(); self._line_handles.clear(); self._cache_pairs_data.clear()

        start = float(self.spin_start.value()); win = float(self.spin_window.value())
        plot_raw = self.check_plot_raw.isChecked(); plot_filt = self.check_plot_filt.isChecked()
        max_pts = int(self.spin_wave_maxpts.value())
        do_norm = self.check_norm.isChecked(); norm_mode = self.combo_norm_mode.currentData(); target_rms = float(self.spin_norm_rms.value())
        norm_only_filt = self.check_norm_only_filt.isChecked()
        thr_on = self.check_thr.isChecked(); thr = float(self.spin_thr.value()); thr_mode = self.combo_thr_mode.currentData()
        thr_src = "filt" if norm_only_filt else self.combo_thr_source.currentData()
        min_event_s = float(self.spin_min_event.value()) / 1000.0

        auto_db = self.check_auto_db.isChecked(); vmin = float(self.spin_vmin.value()); vmax = float(self.spin_vmax.value())
        use_band_for_plot = self.check_filter.isChecked() and self.check_match_filter.isChecked(); show_band = self.check_show_band.isChecked()

        do_freq = self.check_freq_track.isChecked()
        freq_smooth_ms = float(self.spin_freq_smooth.value())
        freq_step_ms = float(self.spin_freq_step.value())

        fill_on = self.check_fill_events.isChecked()
        fill_alpha = float(self.spin_fill_alpha.value())
        fill_color = self.combo_fill_color.currentData()

        if len(loaded) == 0:
            ax = axes[0]; ax.text(0.5, 0.5, "Brak plików — wczytaj w panelu", ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_xlabel("Czas [s]"); ax.set_ylabel("Amplituda"); ax.grid(True, alpha=0.2); self.canvas_pairs.draw_idle(); return

        spec_mappable = None

        for i, slot in enumerate(loaded):
            ax_sig = axes[2*i]; ax_spc = axes[2*i+1]
            ax_sig.set_title(slot.display_name, fontsize=9, loc='left')

            x, fs = slot.read_segment(start, win)
            if x is None or fs is None or x.size == 0:
                ax_sig.text(0.5, 0.5, "Poza zakresem", ha="center", va="center", transform=ax_sig.transAxes, fontsize=11)
                ax_spc.text(0.5, 0.5, "Poza zakresem", ha="center", va="center", transform=ax_spc.transAxes, fontsize=11)
                continue

            x_raw = x
            y_filt = self._maybe_filter(x, fs) if (plot_filt or thr_src == "filt") else None

            # normalizacja (do progów/wykresów)
            x_raw_norm = x_raw; y_filt_norm = y_filt
            if do_norm:
                if norm_only_filt:
                    if y_filt is not None: y_filt_norm = normalize_signal(y_filt, norm_mode, target_rms)
                else:
                    x_raw_norm = normalize_signal(x_raw, norm_mode, target_rms)
                    if y_filt is not None: y_filt_norm = normalize_signal(y_filt, norm_mode, target_rms)

            # detekcja zdarzeń (referencja do progu + baza analizy)
            events = []; ref_full = None; analysis_base = None
            if thr_on:
                if thr_src == "filt":
                    analysis_base = y_filt if y_filt is not None else self._maybe_filter(x, fs)
                    ref_full = y_filt_norm if (y_filt_norm is not None) else (normalize_signal(analysis_base, norm_mode, target_rms) if do_norm else analysis_base)
                else:
                    analysis_base = x_raw
                    if do_norm and not norm_only_filt: ref_full = x_raw_norm
                    elif do_norm and norm_only_filt:  ref_full = normalize_signal(x_raw, norm_mode, target_rms)
                    else: ref_full = x_raw
                events = detect_events(ref_full, fs, thr, thr_mode, min_event_s)
                self._last_events_by_slot[slot.index] = events

            # cache do zoom‑redecymacji
            self._cache_pairs_data[slot.index] = {
                "fs": fs,
                "start": start,
                "x_raw_norm": x_raw_norm,
                "y_plot": (y_filt_norm if y_filt_norm is not None else x_raw_norm),
                "plot_raw": plot_raw,
                "plot_filt": plot_filt,
            }

            # SPEKTROGRAM — z analysis_base (czyli Źródło progu)
            y_for_spec = analysis_base if analysis_base is not None else (y_filt if y_filt is not None else x_raw)
            f, t, Sdb = self._compute_spec_db(y_for_spec, fs)
            nyq = fs * 0.5
            if use_band_for_plot:
                lo = float(self.spin_lo.value()); hi = float(self.spin_hi.value())
                lo = max(0.0, min(lo, nyq * 0.999)); hi = max(0.0, min(hi, nyq * 0.999))
                if hi <= lo:
                    fmax_plot = float(self.spin_fmax.value()); mask_f = f <= fmax_plot; y_lim = (0.0, min(fmax_plot, nyq))
                else:
                    mask_f = (f >= lo) & (f <= hi); y_lim = (lo, hi)
            else:
                fmax_plot = float(self.spin_fmax.value()); mask_f = f <= fmax_plot; y_lim = (0.0, min(fmax_plot, nyq))

            f_plot = f[mask_f]; Sdb_plot = Sdb[mask_f, :]; t_plot = t + start
            if f_plot.size == 0 or t_plot.size == 0:
                ax_spc.text(0.5, 0.5, "Za mało danych do spektrogramu", ha="center", va="center", transform=ax_spc.transAxes, fontsize=10)
            else:
                if auto_db and Sdb_plot.size > 0:
                    lo_p = float(np.percentile(Sdb_plot, 5.0)); hi_p = float(np.percentile(Sdb_plot, 95.0)); vmin_use, vmax_use = lo_p, hi_p
                else:
                    vmin_use, vmax_use = vmin, vmax;
                    if vmax_use <= vmin_use: vmax_use = vmin_use + 40.0
                pc = ax_spc.pcolormesh(t_plot, f_plot, Sdb_plot, shading="auto", vmin=vmin_use, vmax=vmax_use)
                if spec_mappable is None: spec_mappable = pc

            # PASMA ZDARZEŃ (transparent) — zanim narysujemy linie
            if thr_on and events and self.check_fill_events.isChecked():
                for ev in events:
                    t0_abs = start + ev["start_s"]; t1_abs = start + ev["end_s"]
                    ax_sig.axvspan(t0_abs, t1_abs, color=fill_color, alpha=fill_alpha, zorder=1)
                    ax_spc.axvspan(t0_abs, t1_abs, color=fill_color, alpha=fill_alpha, zorder=1)

            # rysuj sygnały
            line_raw = None; line_filt = None
            if plot_raw:
                t_raw, x_dec = decimate_for_plot(x_raw_norm, fs, start, max_pts);
                line_raw_list = ax_sig.plot(t_raw, x_dec, linewidth=0.6, alpha=0.9, label="surowy", zorder=2.0)
                line_raw = line_raw_list[0] if line_raw_list else None
            if plot_filt:
                y_plot = self._cache_pairs_data[slot.index]["y_plot"]
                t_f, y_dec = decimate_for_plot(y_plot, fs, start, max_pts);
                line_filt_list = ax_sig.plot(t_f, y_dec, linewidth=0.8, alpha=0.95, label="po filtrze", zorder=2.1)
                line_filt = line_filt_list[0] if line_filt_list else None

            # LINIE PROGÓW — horyzontalne
            if thr_on:
                thr_color = "tab:red"
                if thr_mode == "abs":
                    ax_sig.axhline(+thr, linestyle="--", linewidth=1.0, alpha=0.9, color=thr_color, zorder=2.9, label="próg ±T")
                    ax_sig.axhline(-thr, linestyle="--", linewidth=1.0, alpha=0.9, color=thr_color, zorder=2.9, label="_nolegend_")
                else:
                    ax_sig.axhline(thr, linestyle="--", linewidth=1.0, alpha=0.9, color=thr_color, zorder=2.9, label="próg T")

            # vlines zdarzeń + tor f(t) na spektrogramie
            if thr_on and events:
                self._freq_tracks_by_slot[slot.index] = []
                for e_idx, ev in enumerate(events):
                    t0_abs = start + ev["start_s"]; t1_abs = start + ev["end_s"]
                    ax_sig.axvline(t0_abs, linewidth=1.2, alpha=0.85, zorder=2.95)
                    ax_sig.axvline(t1_abs, linewidth=1.0, alpha=0.6, linestyle=":", zorder=2.95)
                    ax_spc.axvline(t0_abs, linewidth=1.2, alpha=0.85); ax_spc.axvline(t1_abs, linewidth=1.0, alpha=0.6, linestyle=":")

                    if do_freq and (analysis_base is not None):
                        seg = analysis_base[ev["start_idx"]:ev["end_idx"]+1]
                        fs_loc = fs
                        step_samples = max(1, int(round((freq_step_ms/1000.0) * fs_loc)))
                        if self.combo_freq_method.currentData() == "hilbert":
                            fi = self._inst_freq_hilbert(seg, fs_loc, freq_smooth_ms)
                            idx = np.arange(0, fi.size, step_samples, dtype=int)
                            fi = fi[idx]; tt_rel = idx.astype(np.float64) / float(fs_loc)
                        else:
                            tt_rel, fi = self._freq_track_stft_local(seg, fs_loc, freq_smooth_ms)
                            if tt_rel.size >= 2:
                                dt = float(freq_step_ms/1000.0)
                                t_new = np.arange(tt_rel[0], tt_rel[-1]+1e-9, dt, dtype=np.float64)
                                fi = np.interp(t_new, tt_rel.astype(np.float64), fi.astype(np.float64))
                                tt_rel = t_new.astype(np.float64)
                        if fi.size > 0:
                            nyq2 = fs_loc * 0.5
                            if self.check_freq_clip_band.isChecked() and self.check_filter.isChecked():
                                lo = max(0.0, min(float(self.spin_lo.value()), nyq2)); hi = max(0.0, min(float(self.spin_hi.value()), nyq2))
                                fi = np.clip(fi, lo, hi)
                            maskv = np.isfinite(fi)
                            tt_abs = t0_abs + tt_rel
                            ax_spc.plot(tt_abs[maskv], fi[maskv], linewidth=1.2, alpha=0.95, zorder=3.0)
                            self._freq_tracks_by_slot[slot.index].append({"event_idx": e_idx, "t_abs": tt_abs[maskv].astype(np.float64), "t_rel": tt_rel[maskv].astype(np.float64), "f": fi[maskv].astype(np.float64)})

            # osie/legendy
            ax_sig.set_xlim(start, start + win); ax_sig.grid(True, alpha=0.2); ax_sig.set_ylabel("Amp. (norm.)" if do_norm else "Amp.")
            if (plot_raw or plot_filt or thr_on): ax_sig.legend(loc="upper right", fontsize=8)
            ax_spc.set_xlim(t_plot[0] if t_plot.size else start, t_plot[-1] if t_plot.size else (start + win))
            ax_spc.set_ylim(*y_lim); ax_spc.set_xlabel("Czas [s]"); ax_spc.set_ylabel("Częst. [Hz]"); ax_spc.grid(True, alpha=0.2)

            if show_band and self.check_filter.isChecked():
                lo = float(self.spin_lo.value()); hi = float(self.spin_hi.value())
                lo = max(0.0, min(lo, nyq * 0.999)); hi = max(0.0, min(hi, nyq * 0.999))
                ax_spc.axhline(lo, linestyle="--", linewidth=0.8, alpha=0.7); ax_spc.axhline(hi, linestyle="--", linewidth=0.8, alpha=0.7)
                if hi > lo: ax_spc.axhspan(lo, hi, alpha=0.08)

            # mapuj oś sygnału na slot + uchwyty linii do zoom‑redecymacji
            self._axis_slot_map[ax_sig] = slot.index
            self._line_handles[slot.index] = {"raw": line_raw, "filt": line_filt, "ax": ax_sig}
            ax_sig.callbacks.connect('xlim_changed', self._on_xlim_changed_signal)

        if spec_mappable is not None:
            self._create_cbar(spec_mappable)

        self.canvas_pairs.draw_idle()

    # ----- dynamiczna re‑decymacja przy zoomie -----
    def _on_xlim_changed_signal(self, ax):
        slot_idx = self._axis_slot_map.get(ax, None)
        if slot_idx is None: return
        h = self._line_handles.get(slot_idx, None)
        d = self._cache_pairs_data.get(slot_idx, None)
        if h is None or d is None: return
        fs = d["fs"]; start = d["start"]; max_pts = int(self.spin_wave_maxpts.value())
        x0, x1 = ax.get_xlim()
        if d["plot_raw"] and h["raw"] is not None:
            t, y = decimate_visible(d["x_raw_norm"], fs, start, x0, x1, max_pts)
            h["raw"].set_data(t, y)
        if d["plot_filt"] and h["filt"] is not None:
            t, y = decimate_visible(d["y_plot"], fs, start, x0, x1, max_pts)
            h["filt"].set_data(t, y)
        self.canvas_pairs.draw_idle()

    # ----- rysowanie Tab 2 -----
    def redraw_freq(self):
        loaded = self._loaded_slots(); n_axes = max(1, len(loaded))
        if self.canvas_freq.n_axes != n_axes or self.canvas_freq.row_height_px != self.spin_row_height.value():
            self.canvas_freq.rebuild_axes(n_axes, self.spin_row_height.value())

        axes = self.canvas_freq.axes
        for ax in axes: ax.clear()

        if len(loaded) == 0:
            ax = axes[0]; ax.text(0.5, 0.5, "Brak plików — wczytaj w panelu", ha="center", va="center", transform=ax.transAxes); self.canvas_freq.draw_idle(); return

        mode_time = self.combo_freq_time.currentData()
        for i, slot in enumerate(loaded):
            ax = axes[i]
            ax.set_title(f"{slot.display_name} — tory f zdarzeń", fontsize=9, loc='left')
            tracks = self._freq_tracks_by_slot.get(slot.index, [])
            if not tracks:
                ax.text(0.5, 0.5, "Brak zdarzeń w bieżącym oknie lub brak torów", ha="center", va="center", transform=ax.transAxes)
            else:
                for tr in tracks:
                    t = tr["t_abs"] if mode_time == "abs" else tr["t_rel"]
                    f = tr["f"]
                    ax.plot(t, f, linewidth=1.2, alpha=0.95)
            ax.grid(True, alpha=0.2); ax.set_ylabel("f [Hz]")

        if mode_time == "abs":
            tmins = []; tmaxs = []
            for tracks in self._freq_tracks_by_slot.values():
                for tr in tracks:
                    if tr["t_abs"].size > 0:
                        tmins.append(tr["t_abs"][0]); tmaxs.append(tr["t_abs"][-1])
            if tmins and tmaxs:
                t0 = float(min(tmins)); t1 = float(max(tmaxs));
                for ax in axes: ax.set_xlim(t0, t1)
            for ax in axes: ax.set_xlabel("Czas [s] (absolutny)")
        else:
            for ax in axes: ax.set_xlabel("Czas od startu zdarzenia [s]")

        if self.check_filter.isChecked() and self.check_freq_clip_band.isChecked():
            lo = float(self.spin_lo.value()); hi = float(self.spin_hi.value())
            for ax in axes: ax.set_ylim(lo, hi)

        self.canvas_freq.draw_idle()

    # ----- helpers -----
    def _loaded_slots(self) -> list: return [s for s in self.slots if s.path is not None]

    def redraw_all(self):
        self.redraw_pairs()
        self.redraw_freq()

    def _on_row_height_changed(self, val: int):
        self.canvas_pairs.rebuild_pairs(max(1, len(self._loaded_slots())), val)
        self.canvas_freq.rebuild_axes(max(1, len(self._loaded_slots())), val)
        self.redraw_all()

    def _on_window_changed(self):
        ident, param = self.combo_window.currentData()
        self.spin_beta.setEnabled(ident == "kaiser"); self.spin_alpha.setEnabled(ident == "tukey")
        self.redraw_all()

    def _on_norm_only_filt_toggled(self, checked: bool):
        self.combo_thr_source.setCurrentIndex(0); self.combo_thr_source.setEnabled(not checked); self.redraw_all()

    # ----- eksport CSV -----
    def export_freq_tracks_csv(self):
        any_tracks = sum(len(v) for v in self._freq_tracks_by_slot.values())
        if any_tracks == 0:
            QtWidgets.QMessageBox.information(self, "Brak danych", "Brak torów częstotliwości do eksportu w bieżącym oknie."); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Eksport torów częstotliwości do CSV", str(Path.home() / "tory_czestotliwosci.csv"), "CSV (*.csv)")
        if not path: return
        import csv
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["slot", "plik", "event_idx", "t_abs_s", "t_rel_s", "f_hz"])
                for slot in self._loaded_slots():
                    tracks = self._freq_tracks_by_slot.get(slot.index, [])
                    for tr in tracks:
                        event_idx = tr["event_idx"]
                        for t_abs, t_rel, f in zip(tr["t_abs"], tr["t_rel"], tr["f"]):
                            writer.writerow([slot.index+1, slot.display_name, event_idx, float(t_abs), float(t_rel), float(f)])
            QtWidgets.QMessageBox.information(self, "Zapisano", f"Zapisano tory częstotliwości ({any_tracks} zdarzeń) do:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Błąd zapisu", f"Nie udało się zapisać pliku:\n{e}")

    def save_figure_pairs(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Zapisz rysunek (Tab 1)", str(Path.home() / "sygnaly_spektrogramy.png"), "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path: return
        try: self.canvas_pairs.figure.savefig(path, dpi=150)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Błąd zapisu", f"Nie udało się zapisać pliku:\n{path}\n\n{e}")

    def save_figure_freq(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Zapisz rysunek (Tab 2)", str(Path.home() / "tory_czestotliwosci.png"), "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path: return
        try: self.canvas_freq.figure.savefig(path, dpi=150)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Błąd zapisu", f"Nie udało się zapisać pliku:\n{path}\n\n{e}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None: return super().closeEvent(event)


def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv); win = MainWindow(); win.show(); sys.exit(app.exec_())


if __name__ == "__main__":
    main()

