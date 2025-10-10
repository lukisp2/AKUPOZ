#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Burst Analyzer GUI (fs=192 kHz preferowane)

Zmiany:
- Naprawa dekodera bitów „tylko > threshold”:
  • Wygładzanie sygnału detekcji (MA) przed porównaniem z progiem
  • Dylatacja maski aktywności + margines czasowy na obrzeżach
  • Wyrównanie bitów wewnątrz każdej aktywnej „run” przez skanowanie offsetu
- Ograniczenie rysowania do kilku procent próbek (global, okno, spektrogram)
  • Parametr: „Rysuj % próbek” w sekcji „Wydajność / filtr globalny”
  • Wykresy 1,4,5 podsamplowane wspólnym indeksem; spektrogram – redukcja kolumn

- Spektrogram (nperseg=1024) na dole – tylko pasmo filtru
- Poprawione rysowanie >threshold (maskowanie NaN – brak „zalewania”)
- Dekoder 2‑FSK: 49.6 kHz -> 0, 50.6 kHz -> 1
  * WYŁĄCZNIE z maski progu (det_sig > threshold) – bez fallbacku „siatki symboli”.
"""

import os, json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy.io import wavfile
from scipy.signal import (
    butter, sosfiltfilt, filtfilt, hilbert, find_peaks, get_window, spectrogram
)
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

APP_TITLE = "Burst Analyzer GUI"

# -------- Tooltip --------
class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwin = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        if self.tipwin or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 2
        self.tipwin = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         relief=tk.SOLID, borderwidth=1,
                         font=("TkDefaultFont", 9), padx=6, pady=3)
        label.pack(ipadx=1)

    def hide(self, event=None):
        tw = self.tipwin
        self.tipwin = None
        if tw:
            tw.destroy()

# -------- Scrollable Frame --------
class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.vscroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vscroll.set)
        self.vscroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner_config)
        self.canvas.bind("<Configure>", self._on_canvas_config)
        self.inner.bind("<Enter>", self._bind_mousewheel)
        self.inner.bind("<Leave>", self._unbind_mousewheel)
        self.canvas.bind_all("<Prior>", lambda e: self.canvas.yview_scroll(-1, "page"))
        self.canvas.bind_all("<Next>", lambda e: self.canvas.yview_scroll(1, "page"))

    def _on_inner_config(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_config(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event):
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            self.canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
            self.canvas.yview_scroll(1, "units")

    def _bind_mousewheel(self, event=None):
        try:
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
            self.canvas.bind_all("<Button-4>", self._on_mousewheel)
            self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        except Exception:
            pass

    def _unbind_mousewheel(self, event=None):
        try:
            self.canvas.unbind_all("<MouseWheel>")
        except Exception:
            pass
        try:
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
        except Exception:
            pass

# -------- DSP helpers --------
def read_wav(path):
    sr, x = wavfile.read(path)
    if x.dtype.kind in ("i", "u"):
        max_val = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / max_val
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return sr, x

def write_wav(path, fs, x):
    y = np.clip(x, -1.0, 1.0)
    wavfile.write(path, fs, (y * 32767).astype(np.int16))

def butter_bandpass_sos(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    if not (0 < lowcut < highcut < nyq):
        raise ValueError(f"Niepoprawne pasmo: low={lowcut}, high={highcut}, fs={fs}, Nyquist={nyq}")
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def apply_bandpass(x, fs, lowcut, highcut, order=4):
    xin = np.asarray(x, dtype=np.float32, order="C")
    try:
        sos = butter_bandpass_sos(lowcut, highcut, fs, order)
        y = sosfiltfilt(sos, xin).astype(np.float32, copy=False)
        return y
    except Exception:
        # fallback: filtfilt z parametrem fs (SciPy 1.4+)
        b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
        y = filtfilt(b, a, xin).astype(np.float32, copy=False)
        return y

def envelope_hilbert(x):
    analytic = hilbert(x)
    env = np.abs(analytic)
    return env

def envelope_quick_abs_ma(x, fs, win_ms=0.5):
    xa = np.abs(np.asarray(x, dtype=np.float32))
    w = max(1, int(round((win_ms/1000.0)*fs)))
    if len(xa) <= w:
        return np.full_like(xa, xa.mean(), dtype=np.float32)
    c = np.cumsum(xa, dtype=np.float64)
    valid = (c[w:] - c[:-w]) / float(w)
    left = np.full(w//2, valid[0], dtype=np.float64)
    right = np.full(len(xa) - len(valid) - len(left), valid[-1], dtype=np.float64)
    env = np.concatenate([left, valid, right]).astype(np.float32, copy=False)
    return env

# ---- Dominant frequency estimators ----
def _qifft_parabolic_log(mag, k):
    m1 = np.log(mag[k-1] + 1e-20)
    m0 = np.log(mag[k]   + 1e-20)
    m2 = np.log(mag[k+1] + 1e-20)
    denom = (m1 - 2*m0 + m2)
    if abs(denom) < 1e-20:
        return 0.0
    return 0.5 * (m1 - m2) / denom

def _qifft_parabolic_lin(mag, k):
    a = mag[k-1]; b = mag[k]; c = mag[k+1]
    denom = (a - 2*b + c)
    if abs(denom) < 1e-20:
        return 0.0
    return 0.5 * (a - c) / denom

def _centroid_pm1bin(freqs, mag, k):
    idx = np.array([k-1, k, k+1], dtype=int)
    idx = idx[(idx >= 0) & (idx < len(mag))]
    w = mag[idx]
    if np.sum(w) <= 0:
        return freqs[k]
    return np.sum(freqs[idx] * w) / np.sum(w)

def dominant_freq(segment, fs, method="Parabolic (log-mag)", zp_mult=8):
    seg = segment - np.mean(segment)
    N = len(seg)
    if N < 8:
        return np.nan
    win = get_window("hann", N, fftbins=True)
    nfft_base = int(2**np.ceil(np.log2(N)))
    nfft = max(1, int(zp_mult)) * nfft_base
    spec = np.fft.rfft(seg * win, n=nfft)
    mag = np.abs(spec)
    if len(mag) <= 2:
        return np.nan
    mag[0] = 0.0
    k = int(np.argmax(mag))
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
    if k == 0 or k == len(mag)-1:
        return float(freqs[k])
    if method == "Parabolic (lin-mag)":
        delta = _qifft_parabolic_lin(mag, k)
        k_refined = k + float(delta)
        return float(k_refined * fs / nfft)
    elif method == "Centroid (±1 bin)":
        return float(_centroid_pm1bin(freqs, mag, k))
    else:
        delta = _qifft_parabolic_log(mag, k)
        k_refined = k + float(delta)
        return float(k_refined * fs / nfft)

def next_pow2(n):
    return 1 if n <= 1 else 2**int(np.ceil(np.log2(n)))

# --- FAST band power (dla krótkich okien bitów) ---
def bandpower_fft_fast(seg, fs, f_center, bw_hz, zp_mult=2):
    """
    Moc w paśmie [f_center ± bw_hz/2] dla okna 'seg' przy fs.
    Używa FFT (nfft = zp_mult * next_pow2(N)) -> szybkie i stabilne.
    """
    seg = np.asarray(seg, dtype=np.float32)
    N = len(seg)
    if N < 8:
        return 0.0
    nfft = int(zp_mult) * next_pow2(N)
    win = get_window("hann", N, fftbins=True)
    spec = np.fft.rfft((seg - np.mean(seg)) * win, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
    mag2 = np.abs(spec)**2
    half = bw_hz / 2.0
    mask = (freqs >= (f_center - half)) & (freqs <= (f_center + half))
    if not np.any(mask):
        k = int(round(f_center * nfft / fs))
        k = max(1, min(k, len(freqs)-1))
        return float(mag2[k])
    return float(np.sum(mag2[mask]))

# --- Plot thinning helpers ---
def indices_for_pct(n, pct, max_points=200_000):
    """Zwróć równomiernie rozłożone indeksy do n próbek wg odsetka pct (%)."""
    if n <= 2:
        return np.arange(n, dtype=int)
    pct = max(0.001, min(float(pct), 100.0)) / 100.0
    k = min(max_points, max(2, int(np.ceil(n * pct))))
    return np.linspace(0, n - 1, num=k, dtype=int)

def thin_xy_by_pct(t, y, pct, max_points=200_000):
    n = len(y)
    idx = indices_for_pct(n, pct, max_points=max_points)
    return t[idx], y[idx]

# -------- Main App --------
class BurstAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        try:
            ttk.Style().theme_use("clam")
        except Exception:
            pass
        self.title(APP_TITLE)
        self.geometry("1400x1200")
        self.minsize(1200, 950)

        # Status
        self.status_var = tk.StringVar(value="Gotowy")
        self.status = ttk.Label(self, textvariable=self.status_var, anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # Data
        self.fs = None
        self.x = None
        self.xf = None
        self.env_full = None
        self.filter_state = "unknown"
        self.filter_msg = ""

        # Analysis (window)
        self.s0 = 0; self.s1 = 0
        self.det_sig_win = None
        self.env_win = None
        self.thr_win = None
        self.peaks_win_abs = np.array([], dtype=int)
        self.peak_times = np.array([])
        self.ibi = np.array([]); self.rep_freqs = np.array([])
        self.burst_bounds = []
        self.intraburst_freqs = np.array([])
        self.burst_durations_ms = np.array([])
        self.summary = {}
        self.per_burst_df = pd.DataFrame()

        # Bits (window)
        self.bits_df = pd.DataFrame()
        self.bits_segments = []  # list of (L_abs, R_abs)
        self.bits_string = ""

        # Windowed view
        self.window_len_var = tk.DoubleVar(value=6.0)
        self.window_start_var = tk.DoubleVar(value=0.0)
        self.show_env_in_window_var = tk.BooleanVar(value=True)

        # Threshold
        self.thr_domain_var = tk.StringVar(value="envelope")
        self.thr_mode_var = tk.StringVar(value="relative")
        self.thr_factor_var = tk.DoubleVar(value=5.0)
        self.thr_abs_var = tk.DoubleVar(value=0.1)
        self.thr_show_pm_var = tk.BooleanVar(value=True)

        # f0 estimator
        self.fft_method_var = tk.StringVar(value="Parabolic (log-mag)")
        self.fft_zp_var = tk.IntVar(value=8)

        # Global filtering
        self.global_filter_mode_var = tk.StringVar(value="Auto")
        self.global_filter_auto_max_sec_var = tk.DoubleVar(value=120.0)

        # Plot decimation (% próbek do rysowania)
        self.plot_pct_var = tk.DoubleVar(value=3.0)  # domyślnie 3%

        # Auto analyze toggle
        self.auto_analyze_var = tk.BooleanVar(value=False)

        # Nyquist label
        self.nyquist_var = tk.StringVar(value="Nyquist: - (fs oczekiwane: 192000 Hz)")

        # Default BP
        self.bp_low_var = tk.DoubleVar(value=49000.0)
        self.bp_high_var = tk.DoubleVar(value=52000.0)
        self.bp_order_var = tk.IntVar(value=4)
        self.min_sep_ms_var = tk.DoubleVar(value=10.0)
        self.min_width_ms_var = tk.DoubleVar(value=2.0)
        self.burst_pad_ms_var = tk.DoubleVar(value=3.0)

        # Bit decode params
        self.bit_len_ms_var = tk.DoubleVar(value=6.0)
        self.bit_gap_ms_var = tk.DoubleVar(value=1.0)
        self.bit_f0_low_var = tk.DoubleVar(value=49600.0)   # -> bit 0
        self.bit_f0_high_var = tk.DoubleVar(value=50600.0)  # -> bit 1
        self.bit_bw_hz_var = tk.DoubleVar(value=200.0)      # całe BW (±BW/2)

        # UI
        self.scroller = ScrollableFrame(self)
        self.scroller.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._build_ui(self.scroller.inner)

        # Bindings
        self.bind_all("<space>", lambda e: self.analyze_window())
        self.bind_all("<Left>",  lambda e: self.nudge_window(-0.5))
        self.bind_all("<Right>", lambda e: self.nudge_window(0.5))
        self.bind_all("c", lambda e: self.center_on_selected_burst())
        self.bind_all("a", lambda e: self.suggest_threshold())
        self.bind_all("f", lambda e: self.find_active_window())
        self.bind_all("s", lambda e: self.save_pngs())
        self.bind_all("e", lambda e: self.save_csv())
        self.bind_all("w", lambda e: self.export_window_wav())

    def _build_ui(self, parent):
        # Top controls
        top = ttk.Frame(parent)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)
        top.grid_columnconfigure(1, weight=1)

        self.wav_path_var = tk.StringVar()
        ttk.Label(top, text="Plik WAV:").grid(row=0, column=0, sticky="w")
        self.wav_entry = ttk.Entry(top, textvariable=self.wav_path_var)
        self.wav_entry.grid(row=0, column=1, sticky="we", padx=6)
        btn_browse = ttk.Button(top, text="Wybierz...", command=self.choose_file)
        btn_browse.grid(row=0, column=2, padx=4)
        Tooltip(btn_browse, "Wybierz plik WAV (preferowane fs=192 kHz)")

        # Parametry
        param_frame = ttk.LabelFrame(top, text="Parametry analizy")
        param_frame.grid(row=1, column=0, columnspan=3, sticky="we", pady=6, padx=2)
        for col in [1,3,5]: param_frame.grid_columnconfigure(col, minsize=80)

        row = 0
        ttk.Label(param_frame, text="BP low [Hz]").grid(row=row, column=0, padx=4, pady=2, sticky="w")
        e1 = ttk.Entry(param_frame, textvariable=self.bp_low_var, width=10); e1.grid(row=row, column=1, padx=4)
        ttk.Label(param_frame, text="BP high [Hz]").grid(row=row, column=2, padx=4, pady=2, sticky="w")
        e2 = ttk.Entry(param_frame, textvariable=self.bp_high_var, width=10); e2.grid(row=row, column=3, padx=4)
        ttk.Label(param_frame, text="Order").grid(row=row, column=4, padx=4, pady=2, sticky="w")
        e3 = ttk.Entry(param_frame, textvariable=self.bp_order_var, width=6); e3.grid(row=row, column=5, padx=4)
        row += 1
        ttk.Label(param_frame, text="Min odstęp [ms]").grid(row=row, column=0, padx=4, sticky="w")
        e4 = ttk.Entry(param_frame, textvariable=self.min_sep_ms_var, width=10); e4.grid(row=row, column=1, padx=4)
        ttk.Label(param_frame, text="Min szer. [ms]").grid(row=row, column=2, padx=4, sticky="w")
        e5 = ttk.Entry(param_frame, textvariable=self.min_width_ms_var, width=10); e5.grid(row=row, column=3, padx=4)
        ttk.Label(param_frame, text="Pad FFT [ms]").grid(row=row, column=4, padx=4, sticky="w")
        e6 = ttk.Entry(param_frame, textvariable=self.burst_pad_ms_var, width=10); e6.grid(row=row, column=5, padx=4)

        self.nyquist_var = tk.StringVar(value="Nyquist: - (fs oczekiwane: 192000 Hz)")
        ttk.Label(param_frame, textvariable=self.nyquist_var).grid(row=row+1, column=0, columnspan=6, sticky="w", padx=4, pady=(3,0))

        # Próg i detekcja
        thr_frame = ttk.LabelFrame(top, text="Próg i detekcja")
        thr_frame.grid(row=2, column=0, columnspan=3, sticky="we", pady=6, padx=2)
        for col in range(6): thr_frame.grid_columnconfigure(col, minsize=80)

        ttk.Label(thr_frame, text="Domena:").grid(row=0, column=0, padx=4, sticky="w")
        rb_env = ttk.Radiobutton(thr_frame, text="Obwiednia", value="envelope",
                        variable=self.thr_domain_var, command=self._on_thr_ui_change)
        rb_env.grid(row=0, column=1, padx=4, sticky="w")
        rb_abs = ttk.Radiobutton(thr_frame, text="|Sygnał po filtracji|", value="abs_filtered",
                        variable=self.thr_domain_var, command=self._on_thr_ui_change)
        rb_abs.grid(row=0, column=2, padx=4, sticky="w")

        ttk.Label(thr_frame, text="Tryb:").grid(row=1, column=0, padx=4, sticky="w")
        rb_rel = ttk.Radiobutton(thr_frame, text="Relatywny (× mediana)", value="relative",
                        variable=self.thr_mode_var, command=self._on_thr_ui_change)
        rb_rel.grid(row=1, column=1, padx=4, sticky="w")
        rb_abs2 = ttk.Radiobutton(thr_frame, text="Absolutny", value="absolute",
                        variable=self.thr_mode_var, command=self._on_thr_ui_change)
        rb_abs2.grid(row=1, column=2, padx=4, sticky="w")

        ttk.Label(thr_frame, text="Krotność (rel.)").grid(row=2, column=0, padx=4, sticky="w")
        self.thr_factor_entry = ttk.Entry(thr_frame, textvariable=self.thr_factor_var, width=10)
        self.thr_factor_entry.grid(row=2, column=1, padx=4, sticky="w")
        ttk.Label(thr_frame, text="Wartość (abs.)").grid(row=2, column=2, padx=4, sticky="w")
        self.thr_abs_entry = ttk.Entry(thr_frame, textvariable=self.thr_abs_var, width=10)
        self.thr_abs_entry.grid(row=2, column=3, padx=4, sticky="w")
        self.thr_pm_chk = ttk.Checkbutton(thr_frame, text="Rysuj ±thr dla |xf|", variable=self.thr_show_pm_var)
        self.thr_pm_chk.grid(row=2, column=4, padx=8, sticky="w")
        btn_suggest = ttk.Button(thr_frame, text="🔎 Proponuj próg", command=self.suggest_threshold)
        btn_suggest.grid(row=2, column=5, padx=8, sticky="w")

        # f0
        f0_frame = ttk.LabelFrame(top, text="Dominująca częstotliwość (f0)")
        f0_frame.grid(row=3, column=0, columnspan=3, sticky="we", pady=4, padx=2)
        ttk.Label(f0_frame, text="Estimator:").grid(row=0, column=0, padx=4, sticky="w")
        self.f0_method_cb = ttk.Combobox(f0_frame, state="readonly",
                                         values=["Parabolic (log-mag)", "Parabolic (lin-mag)", "Centroid (±1 bin)"],
                                         textvariable=self.fft_method_var, width=22)
        self.f0_method_cb.grid(row=0, column=1, padx=4, sticky="w")
        ttk.Label(f0_frame, text="Zero-pad ×").grid(row=0, column=2, padx=8, sticky="w")
        self.f0_zp_cb = ttk.Combobox(f0_frame, state="readonly",
                                     values=[1,2,4,8,16], width=6, textvariable=self.fft_zp_var)
        self.f0_zp_cb.grid(row=0, column=3, padx=4, sticky="w")
        self.f0_method_cb.bind("<<ComboboxSelected>>", lambda e: self.maybe_auto_analyze())
        self.f0_zp_cb.bind("<<ComboboxSelected>>", lambda e: self.maybe_auto_analyze())

        # Wydajność / filtr globalny
        perf_frame = ttk.LabelFrame(top, text="Wydajność / filtr globalny")
        perf_frame.grid(row=4, column=0, columnspan=3, sticky="we", pady=4, padx=2)
        ttk.Label(perf_frame, text="Filtr globalny:").grid(row=0, column=0, padx=4, sticky="w")
        self.global_filter_cb = ttk.Combobox(perf_frame, state="readonly",
                     values=["Auto", "Tylko ramka", "Cały sygnał"],
                     textvariable=self.global_filter_mode_var, width=14)
        self.global_filter_cb.grid(row=0, column=1, padx=4, sticky="w")
        ttk.Label(perf_frame, text="Limit (Auto) [s]").grid(row=0, column=2, padx=8, sticky="w")
        self.global_filter_limit_entry = ttk.Entry(perf_frame, textvariable=self.global_filter_auto_max_sec_var, width=8)
        self.global_filter_limit_entry.grid(row=0, column=3, padx=4, sticky="w")
        ttk.Label(perf_frame, text="Rysuj % próbek").grid(row=0, column=4, padx=8, sticky="w")
        self.plot_pct_entry = ttk.Entry(perf_frame, textvariable=self.plot_pct_var, width=6)
        self.plot_pct_entry.grid(row=0, column=5, padx=4, sticky="w")
        Tooltip(self.plot_pct_entry, "Ile % próbek rysować (np. 3 = ~3%). Dotyczy wykresów 1/4/5 i zagęszczenia kolumn spektrogramu.")

        # Auto-analiza + preset
        intel_frame = ttk.Frame(top)
        intel_frame.grid(row=5, column=0, columnspan=3, sticky="we", pady=2)
        self.auto_chk = ttk.Checkbutton(intel_frame, text="Auto-analiza przy zmianie parametrów",
                                        variable=self.auto_analyze_var, command=lambda: self._set_status("Gotowy"))
        self.auto_chk.pack(side=tk.LEFT, padx=4)
        btn_find = ttk.Button(intel_frame, text="🧭 Znajdź aktywne okno", command=self.find_active_window)
        btn_find.pack(side=tk.LEFT, padx=6)
        ttk.Button(intel_frame, text="💾 Zapisz preset", command=self.save_preset).pack(side=tk.RIGHT, padx=6)
        ttk.Button(intel_frame, text="📂 Wczytaj preset", command=self.load_preset).pack(side=tk.RIGHT, padx=6)

        # Dekoder bitów
        bits_frame = ttk.LabelFrame(top, text="Dekodowanie bitów (TYLKO ramka)")
        bits_frame.grid(row=6, column=0, columnspan=3, sticky="we", pady=4, padx=2)
        ttk.Label(bits_frame, text="Bit ON [ms]").grid(row=0, column=0, padx=4, sticky="w")
        ttk.Entry(bits_frame, textvariable=self.bit_len_ms_var, width=8).grid(row=0, column=1, padx=4, sticky="w")
        ttk.Label(bits_frame, text="Przerwa [ms]").grid(row=0, column=2, padx=4, sticky="w")
        ttk.Entry(bits_frame, textvariable=self.bit_gap_ms_var, width=8).grid(row=0, column=3, padx=4, sticky="w")
        ttk.Label(bits_frame, text="f0(0) [Hz]").grid(row=0, column=4, padx=4, sticky="w")
        ttk.Entry(bits_frame, textvariable=self.bit_f0_low_var, width=10).grid(row=0, column=5, padx=4, sticky="w")
        ttk.Label(bits_frame, text="f0(1) [Hz]").grid(row=0, column=6, padx=4, sticky="w")
        ttk.Entry(bits_frame, textvariable=self.bit_f0_high_var, width=10).grid(row=0, column=7, padx=4, sticky="w")
        ttk.Label(bits_frame, text="BW [Hz]").grid(row=0, column=8, padx=4, sticky="w")
        ttk.Entry(bits_frame, textvariable=self.bit_bw_hz_var, width=8).grid(row=0, column=9, padx=4, sticky="w")
        ttk.Button(bits_frame, text="▶ Dekoduj bity w RAMCE", command=self.decode_bits_only).grid(row=0, column=10, padx=10)

        # Akcje
        btn_frame = ttk.Frame(top)
        btn_frame.grid(row=7, column=0, columnspan=3, sticky="we", pady=6)
        ttk.Button(btn_frame, text="▶ Analizuj RAMKĘ", command=self.analyze_window).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="📊 Zapisz wyniki CSV", command=self.save_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="🖼 Zapisz wykresy PNG", command=self.save_pngs).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="🔉 Eksportuj OKNO do WAV", command=self.export_window_wav).pack(side=tk.LEFT, padx=4)

        # Middle: Stats + Lists
        mid = ttk.Frame(parent)
        mid.pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)

        stats = ttk.LabelFrame(mid, text="Podsumowanie (TYLKO ramka)")
        stats.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self.stats_vars = {
            "fs": tk.StringVar(value="-"),
            "win_start": tk.StringVar(value="-"),
            "win_len": tk.StringVar(value="-"),
            "num_bursts": tk.StringVar(value="-"),
            "mean_ibi_ms": tk.StringVar(value="-"),
            "median_ibi_ms": tk.StringVar(value="-"),
            "mean_rep_hz": tk.StringVar(value="-"),
            "median_rep_hz": tk.StringVar(value="-"),
            "median_intra_hz": tk.StringVar(value="-"),
            "thr_win": tk.StringVar(value="-"),
            "num_bits": tk.StringVar(value="-"),
            "bits_str": tk.StringVar(value="-"),
        }

        def add_stat(row, label, var):
            ttk.Label(stats, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=1)
            ttk.Label(stats, textvariable=var).grid(row=row, column=1, sticky="w", padx=4, pady=1)

        add_stat(0, "fs [Hz]:", self.stats_vars["fs"])
        add_stat(1, "Start ramki [s]:", self.stats_vars["win_start"])
        add_stat(2, "Dł. ramki [s]:", self.stats_vars["win_len"])
        add_stat(3, "Liczba burstów:", self.stats_vars["num_bursts"])
        add_stat(4, "Średni IBI [ms]:", self.stats_vars["mean_ibi_ms"])
        add_stat(5, "Mediana IBI [ms]:", self.stats_vars["median_ibi_ms"])
        add_stat(6, "Średnia rep. [Hz]:", self.stats_vars["mean_rep_hz"])
        add_stat(7, "Mediana rep. [Hz]:", self.stats_vars["median_rep_hz"])
        add_stat(8, "Mediana f0 (intra) [Hz]:", self.stats_vars["median_intra_hz"])
        add_stat(9, "Threshold:", self.stats_vars["thr_win"])
        add_stat(10, "Liczba bitów:", self.stats_vars["num_bits"])
        add_stat(11, "Ciąg bitów:", self.stats_vars["bits_str"])

        # List of bursts
        burst_list_frame = ttk.LabelFrame(mid, text="Bursty (TYLKO ramka)")
        burst_list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        columns = ("idx", "time_s", "dur_ms", "f0_hz")
        self.tree = ttk.Treeview(burst_list_frame, columns=columns, show="headings", height=10)
        self.tree.heading("idx", text="Idx")
        self.tree.heading("time_s", text="Czas [s] (abs)")
        self.tree.heading("dur_ms", text="Czas trwania [ms]")
        self.tree.heading("f0_hz", text="f0 [Hz]")
        self.tree.column("idx", width=60, anchor="center")
        self.tree.column("time_s", width=140, anchor="center")
        self.tree.column("dur_ms", width=140, anchor="center")
        self.tree.column("f0_hz", width=120, anchor="center")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_burst)
        self.tree.bind("<Double-1>", lambda e: self.center_on_selected_burst())
        vsb = ttk.Scrollbar(burst_list_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill="y")
        self.tree.configure(yscrollcommand=vsb.set)

        # List of bits
        bits_list_frame = ttk.LabelFrame(mid, text="Bity (TYLKO ramka)")
        bits_list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        bits_cols = ("bidx", "t_s_abs", "bit", "f0_est", "P0", "P1", "conf")
        self.bits_tree = ttk.Treeview(bits_list_frame, columns=bits_cols, show="headings", height=10)
        for c, w in zip(bits_cols, [60, 120, 60, 100, 100, 100, 80]):
            self.bits_tree.heading(c, text=c)
            self.bits_tree.column(c, width=w, anchor="center")
        self.bits_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb2 = ttk.Scrollbar(bits_list_frame, orient="vertical", command=self.bits_tree.yview)
        vsb2.pack(side=tk.RIGHT, fill="y")
        self.bits_tree.configure(yscrollcommand=vsb2.set)

        # Window controls
        window_ctrl = ttk.LabelFrame(parent, text="Podgląd okna (surowy i filtrowany)")
        window_ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Label(window_ctrl, text="Długość okna [s]").grid(row=0, column=0, padx=4, pady=2, sticky="w")
        self.window_len_entry = ttk.Entry(window_ctrl, textvariable=self.window_len_var, width=8)
        self.window_len_entry.grid(row=0, column=1, padx=4)
        ttk.Label(window_ctrl, text="Start okna [s]").grid(row=0, column=2, padx=4, pady=2, sticky="w")
        self.window_start_entry = ttk.Entry(window_ctrl, textvariable=self.window_start_var, width=10)
        self.window_start_entry.grid(row=0, column=3, padx=4)

        btn_apply = ttk.Button(window_ctrl, text="Zastosuj okno", command=self.apply_window_params)
        btn_apply.grid(row=0, column=4, padx=6)
        ttk.Button(window_ctrl, text="⟸ w lewo", command=lambda: self.nudge_window(-0.5)).grid(row=0, column=5, padx=2)
        ttk.Button(window_ctrl, text="w prawo ⟹", command=lambda: self.nudge_window(0.5)).grid(row=0, column=6, padx=2)
        ttk.Button(window_ctrl, text="Centruj na burscie", command=self.center_on_selected_burst).grid(row=0, column=7, padx=6)

        ttk.Checkbutton(window_ctrl, text="Pokaż sygnał detekcji (env/|xf|)",
                        variable=self.show_env_in_window_var, command=self.update_window_plots).grid(row=0, column=8, padx=8)

        self.window_scale = tk.Scale(window_ctrl, from_=0.0, to=0.0, resolution=0.001,
                                     orient=tk.HORIZONTAL, length=700,
                                     command=self.on_window_scale_change)
        self.window_scale.grid(row=1, column=0, columnspan=9, sticky="we", padx=4, pady=4)

        # Plots area
        plot_area = ttk.Frame(parent)
        plot_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Figure 1: global
        self.fig1 = Figure(figsize=(5, 3), dpi=100, constrained_layout=True)
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=plot_area)
        self.canvas1.draw_idle()
        self.canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar1 = NavigationToolbar2Tk(self.canvas1, plot_area, pack_toolbar=False)
        self.toolbar1.update(); self.toolbar1.pack(side=tk.TOP, fill=tk.X)

        # Figure 2: IBI histogram
        self.fig2 = Figure(figsize=(5, 3), dpi=100, constrained_layout=True)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=plot_area)
        self.canvas2.draw_idle()
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar2 = NavigationToolbar2Tk(self.canvas2, plot_area, pack_toolbar=False)
        self.toolbar2.update(); self.toolbar2.pack(side=tk.TOP, fill=tk.X)

        # Figure 3: FFT burst
        self.fig3 = Figure(figsize=(5, 3), dpi=100, constrained_layout=True)
        self.ax3 = self.fig3.add_subplot(111)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=plot_area)
        self.canvas3.draw_idle()
        self.canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar3 = NavigationToolbar2Tk(self.canvas3, plot_area, pack_toolbar=False)
        self.toolbar3.update(); self.toolbar3.pack(side=tk.TOP, fill=tk.X)

        # Figure 4: raw window
        self.fig4 = Figure(figsize=(5, 2.8), dpi=100, constrained_layout=True)
        self.ax4 = self.fig4.add_subplot(111)
        self.canvas4 = FigureCanvasTkAgg(self.fig4, master=plot_area)
        self.canvas4.draw_idle()
        self.canvas4.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar4 = NavigationToolbar2Tk(self.canvas4, plot_area, pack_toolbar=False)
        self.toolbar4.update(); self.toolbar4.pack(side=tk.TOP, fill=tk.X)

        # Figure 5: filtered window + detection
        self.fig5 = Figure(figsize=(5, 2.8), dpi=100, constrained_layout=True)
        self.ax5 = self.fig5.add_subplot(111)
        self.canvas5 = FigureCanvasTkAgg(self.fig5, master=plot_area)
        self.canvas5.draw_idle()
        self.canvas5.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar5 = NavigationToolbar2Tk(self.canvas5, plot_area, pack_toolbar=False)
        self.toolbar5.update(); self.toolbar5.pack(side=tk.TOP, fill=tk.X)

        # Figure 6: spectrogram
        self.fig6 = Figure(figsize=(5, 3.2), dpi=100, constrained_layout=True)
        self.ax6 = self.fig6.add_subplot(111)
        self.canvas6 = FigureCanvasTkAgg(self.fig6, master=plot_area)
        self.canvas6.draw_idle()
        self.canvas6.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar6 = NavigationToolbar2Tk(self.canvas6, plot_area, pack_toolbar=False)
        self.toolbar6.update(); self.toolbar6.pack(side=tk.TOP, fill=tk.X)

        # Init titles
        self.ax1.set_title("PODGLĄD GLOBALNY — bandpass/raw + envelope (z ramką)")
        self.ax1.set_xlabel("Czas [s]"); self.ax1.set_ylabel("Amplituda / Obwiednia")
        self.ax2.set_title("Histogram IBI (TYLKO ramka)"); self.ax2.set_xlabel("IBI [ms]"); self.ax2.set_ylabel("Liczność")
        self.ax3.set_title("FFT wybranego bursta (TYLKO ramka)"); self.ax3.set_xlabel("Częstotliwość [Hz]"); self.ax3.set_ylabel("Magnituda")
        self.ax4.set_title("Surowy sygnał — widok okna"); self.ax4.set_xlabel("Czas [s]"); self.ax4.set_ylabel("Amplituda")
        self.ax5.set_title("Sygnał po filtracji — widok okna (detekcja, próg, piki, bity)")
        self.ax5.set_xlabel("Czas [s]"); self.ax5.set_ylabel("Amplituda / Obwiednia")
        self.ax6.set_title("SPEKTROGRAM (ramka, przefiltrowany, tylko pasmo BP)")
        self.ax6.set_xlabel("Czas [s]"); self.ax6.set_ylabel("Częstotliwość [Hz]")

    # ---------- Helpers ----------
    def _set_status(self, text):
        self.status_var.set(text)
        self.status.update_idletasks()

    def _prepare_global_signals(self, x, fs):
        mode = self.global_filter_mode_var.get()
        try:
            low = float(self.bp_low_var.get()); high = float(self.bp_high_var.get()); order = int(self.bp_order_var.get())
        except Exception as e:
            return None, envelope_quick_abs_ma(x, fs, 0.5), 'error', f'Błędne parametry BP: {e}'
        nyq = 0.5 * fs
        if not (0 < low < high < nyq):
            env = envelope_quick_abs_ma(x, fs, 0.5)
            return None, env, 'invalid', f'Pasmo {low:.1f}–{high:.1f} Hz poza Nyquistem ({nyq:.1f} Hz).'
        N = len(x); T = N / float(fs)
        if mode == 'Tylko ramka':
            env = envelope_quick_abs_ma(x, fs, 0.5)
            return None, env, 'window-only', 'Tryb: Tylko ramka (global bez filtracji)'
        if mode == 'Auto':
            limit = float(self.global_filter_auto_max_sec_var.get())
            if T > limit:
                env = envelope_quick_abs_ma(x, fs, 0.5)
                return None, env, 'window-only', f'Auto: długość {T:.1f}s > {limit:.1f}s (global bez filtracji)'
        try:
            xf = apply_bandpass(x, fs, low, high, order)
        except Exception as e:
            env = envelope_quick_abs_ma(x, fs, 0.5)
            return None, env, 'error', f'Problem filtru globalnego: {e}'
        try:
            env = envelope_quick_abs_ma(xf, fs, 0.5)
        except Exception:
            env = envelope_quick_abs_ma(x, fs, 0.5)
        return xf, env, 'ok', ''

    # ---------- File/load & window ----------
    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Wybierz plik WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if path:
            self.wav_path_var.set(path)
            self.prepare_global_view()

    def prepare_global_view(self):
        path = self.wav_path_var.get().strip()
        if not path or not os.path.exists(path):
            return
        try:
            fs, x = read_wav(path)
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Błąd odczytu WAV: {e}")
            return
        self.fs, self.x = fs, x
        self.nyquist_var.set(f"Nyquist: {0.5*fs:.1f} Hz  (fs={fs} Hz; oczekiwane 192000 Hz)")
        if fs != 192000:
            self._set_status(f"UWAGA: fs={fs} Hz (różne od 192000 Hz) — kontynuuję.")

        self.xf, self.env_full, self.filter_state, self.filter_msg = self._prepare_global_signals(x, fs)
        if self.filter_state != "ok":
            self._set_status(f"Filtr globalny: {self.filter_state} — {self.filter_msg}")
        else:
            self._set_status("Załadowano i przygotowano przegląd globalny.")

        self.configure_window_scale()
        self.update_global_plot()
        self.update_window_plots()
        try:
            self.analyze_window()
        except Exception:
            pass

    def on_window_scale_change(self, val):
        try: v = float(val)
        except Exception: return
        self.window_start_var.set(v)
        self.update_window_plots()
        self.maybe_auto_analyze()

    def apply_window_params(self):
        if self.fs is None or self.x is None:
            return
        try:
            win = float(self.window_len_var.get()); start = float(self.window_start_var.get())
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Błędne parametry okna: {e}")
            return
        win = max(0.001, win)
        T = len(self.x) / self.fs
        max_start = max(0.0, T - win)
        start = min(max(0.0, start), max_start)
        self.window_len_var.set(win); self.window_start_var.set(start)
        self.configure_window_scale()
        self.update_window_plots()
        self.maybe_auto_analyze()

    def nudge_window(self, fraction):
        if self.fs is None or self.x is None: return
        win = float(self.window_len_var.get())
        delta = win * float(fraction)
        start = float(self.window_start_var.get()) + delta
        T = len(self.x) / self.fs
        max_start = max(0.0, T - win)
        start = min(max(0.0, start), max_start)
        self.window_start_var.set(start)
        self.window_scale.set(start)
        self.update_window_plots()
        self.maybe_auto_analyze()

    def center_on_selected_burst(self):
        if self.fs is None or self.x is None or not len(self.peak_times): return
        sel = self.tree.selection()
        if not sel: idx = 0
        else:
            vals = self.tree.item(sel[0], "values"); idx = int(vals[0]) if vals else 0
        t0 = float(self.peak_times[idx])
        win = float(self.window_len_var.get())
        start = t0 - 0.5 * win
        T = len(self.x) / self.fs
        max_start = max(0.0, T - win)
        start = min(max(0.0, start), max_start)
        self.window_start_var.set(start); self.window_scale.set(start)
        self.update_window_plots()
        self.maybe_auto_analyze()

    def configure_window_scale(self):
        if self.fs is None or self.x is None: return
        T = len(self.x) / self.fs
        win = max(0.001, float(self.window_len_var.get()))
        max_start = max(0.0, T - win)
        self.window_scale.configure(from_=0.0, to=max_start, resolution=max(0.001, win/1000.0))
        start = float(self.window_start_var.get())
        start = min(max(0.0, start), max_start)
        self.window_start_var.set(start); self.window_scale.set(start)

    # ---------- Threshold UI ----------
    def _on_thr_ui_change(self):
        mode = self.thr_mode_var.get()
        if mode == "relative":
            self.thr_factor_entry.configure(state="normal")
            self.thr_abs_entry.configure(state="disabled")
        else:
            self.thr_factor_entry.configure(state="disabled")
            self.thr_abs_entry.configure(state="normal")
        self.update_window_plots()
        self.maybe_auto_analyze()

    # ---------- Intelligent helpers ----------
    def suggest_threshold(self):
        if self.fs is None or self.x is None:
            messagebox.showwarning(APP_TITLE, "Wczytaj plik i ustaw okno.")
            return
        fs = self.fs
        win = max(0.001, float(self.window_len_var.get()))
        start = float(self.window_start_var.get())
        s0 = int(max(0, np.floor(start * fs)))
        s1 = int(min(len(self.x), np.ceil((start + win) * fs)))
        if s1 <= s0: s1 = min(len(self.x), s0 + 1)
        low = float(self.bp_low_var.get()); high = float(self.bp_high_var.get()); order = int(self.bp_order_var.get())
        nyq = 0.5 * fs
        try:
            if 0 < low < high < nyq:
                xf_win = apply_bandpass(self.x[s0:s1], fs, low, high, order)
            else:
                xf_win = self.x[s0:s1]
        except Exception:
            xf_win = self.x[s0:s1]
        env_win = envelope_hilbert(xf_win)
        absf_win = np.abs(xf_win)
        sig = env_win if self.thr_domain_var.get() == "envelope" else absf_win
        med = float(np.median(sig)); mad = float(np.median(np.abs(sig - med)) + 1e-12)
        p98 = float(np.percentile(sig, 98))
        thr_abs_suggest = max(med + 6.0 * mad, p98)
        thr_rel_suggest = (thr_abs_suggest / med) if med > 1e-9 else 5.0
        if self.thr_mode_var.get() == "relative":
            self.thr_factor_var.set(float(thr_rel_suggest))
            self._set_status(f"Proponowana krotność ≈ {thr_rel_suggest:.3f}")
        else:
            self.thr_abs_var.set(float(thr_abs_suggest))
            self._set_status(f"Proponowany próg abs ≈ {thr_abs_suggest:.3f}")
        self.update_window_plots()
        self.maybe_auto_analyze()

    def find_active_window(self):
        if self.fs is None or self.x is None or self.env_full is None:
            messagebox.showwarning(APP_TITLE, "Najpierw wczytaj plik (podgląd globalny).")
            return
        fs = self.fs
        win = max(0.1, float(self.window_len_var.get()))
        N = len(self.env_full)
        W = max(1, int(win * fs))
        env = self.env_full
        c = np.cumsum(np.insert(env, 0, 0.0))
        best_start = 0; best_sum = -1.0
        step = max(1, W // 20)
        for s in range(0, N - W + 1, step):
            val = c[s+W] - c[s]
            if val > best_sum:
                best_sum = val; best_start = s
        start_sec = best_start / fs
        self.window_start_var.set(float(start_sec)); self.window_scale.set(float(start_sec))
        self.update_window_plots()
        self.maybe_auto_analyze()
        self._set_status("Ustawiono okno o maksymalnej energii obwiedni.")

    def maybe_auto_analyze(self):
        if not hasattr(self, "_auto_job"):
            self._auto_job = None
        if not getattr(self, "auto_analyze_var", tk.BooleanVar(value=False)).get():
            return
        if self._auto_job is not None:
            self.after_cancel(self._auto_job)
        self._auto_job = self.after(250, self.analyze_window)

    # ---------- Bit helpers ----------
    @staticmethod
    def _find_runs_bool(a_bool):
        """Zwraca listę (start, end) przedziałów True (end exclusive)."""
        a = np.asarray(a_bool, dtype=np.uint8)
        if a.size == 0:
            return []
        da = np.diff(np.r_[0, a, 0])
        starts = np.where(da == 1)[0]
        ends   = np.where(da == -1)[0]
        return list(zip(starts, ends))

    @staticmethod
    def _smooth_ma(x, win_samp):
        if win_samp <= 1:
            return x.astype(np.float32, copy=False)
        kernel = np.ones(int(win_samp), dtype=np.float32) / float(win_samp)
        return np.convolve(x.astype(np.float32), kernel, mode="same")

    def _decode_bits_from_mask(self, xf_win, det_sig, fs, s0_abs):
        """
        Dekoder z maski progu: tnie tylko tam, gdzie (det_sig_smooth > threshold).
        Wewnętrznie:
          • wygładzenie det_sig oknem ~0.3*bit_len (min 0.2 ms, max 2 ms),
          • dylatacja maski o ~0.5 ms,
          • margines ±0.2*bit_len na krańcach run,
          • skanowanie offsetu (0..slot) w kroku ~0.5 ms w obrębie każdej run.
        """
        bit_on = float(self.bit_len_ms_var.get()) / 1000.0
        bit_gap = float(self.bit_gap_ms_var.get()) / 1000.0
        f0_0 = float(self.bit_f0_low_var.get())
        f0_1 = float(self.bit_f0_high_var.get())
        bw = float(self.bit_bw_hz_var.get())
        thr = float(self.thr_win if self.thr_win is not None else 0.0)

        on_samp = int(round(bit_on * fs))
        gap_samp = int(round(bit_gap * fs))
        slot_samp = on_samp + gap_samp
        if on_samp < 2 or slot_samp < on_samp + 1:
            return pd.DataFrame(columns=["bidx","t_s_abs","bit","f0_est","P0","P1","conf"]), [], ""

        # 1) Wygładzenie i maska > threshold
        smooth_ms = float(np.clip(0.3 * self.bit_len_ms_var.get(), 0.2, 2.0))
        w_smooth = max(1, int(round((smooth_ms/1000.0)*fs)))
        det_smooth = self._smooth_ma(det_sig, w_smooth)
        active = det_smooth > thr

        # 2) Dylatacja maski o ~0.5 ms, aby zalać krótkie dołki poniżej progu
        dilate_ms = 0.5
        k = max(1, int(round((dilate_ms/1000.0)*fs)))
        if k > 1:
            active = (np.convolve(active.astype(np.uint8), np.ones(k, dtype=np.uint8), mode="same") > 0)

        runs = self._find_runs_bool(active)
        bits = []
        segs = []
        bcount = 0

        # 3) Margines na krańcach run
        margin = max(0, int(round(0.2 * on_samp)))

        # 4) Dla każdej runy – znajdź najlepszy offset bitowy i dekoduj
        step_offset = max(1, int(round(0.0005 * fs)))  # ~0.5 ms

        for L, R in runs:
            L2 = max(0, L - margin)
            R2 = min(len(xf_win), R + margin)
            run_len = R2 - L2
            if run_len < on_samp:
                continue

            best_score = -np.inf
            best_off = 0

            # Skanuj offset, oceniaj średnią pewność
            for off in range(0, slot_samp, step_offset):
                pos = L2 + off
                if pos + on_samp > R2:
                    break
                confs = []
                cnt = 0
                p = pos
                while p + on_samp <= R2:
                    seg = xf_win[p:p+on_samp]
                    P0 = bandpower_fft_fast(seg, fs, f0_0, bw, zp_mult=2)
                    P1 = bandpower_fft_fast(seg, fs, f0_1, bw, zp_mult=2)
                    conf = (abs(P1 - P0)) / (P0 + P1 + 1e-12)
                    confs.append(conf)
                    cnt += 1
                    p += slot_samp
                if cnt == 0:
                    continue
                score = float(np.mean(confs))
                if score > best_score:
                    best_score = score
                    best_off = off

            # Dekoduj z najlepszym offsetem (wciąż w granicach runy)
            pos = L2 + best_off
            while pos + on_samp <= R2:
                seg = xf_win[pos:pos+on_samp]
                P0 = bandpower_fft_fast(seg, fs, f0_0, bw, zp_mult=2)
                P1 = bandpower_fft_fast(seg, fs, f0_1, bw, zp_mult=2)
                f0_est = dominant_freq(seg, fs, method=self.fft_method_var.get(), zp_mult=int(self.fft_zp_var.get()))
                bit = 0 if P0 >= P1 else 1
                conf = (max(P0, P1) - min(P0, P1)) / (P0 + P1 + 1e-12)
                t_abs = (s0_abs + pos) / fs
                bits.append(dict(bidx=bcount, t_s_abs=float(t_abs), bit=int(bit),
                                 f0_est=float(f0_est), P0=float(P0), P1=float(P1), conf=float(conf)))
                segs.append((s0_abs + pos, s0_abs + pos + on_samp))
                bcount += 1
                pos += slot_samp

        bits_df = pd.DataFrame(bits) if bits else pd.DataFrame(columns=["bidx","t_s_abs","bit","f0_est","P0","P1","conf"])
        bits_str = "".join(str(int(b)) for b in bits_df["bit"].tolist()) if len(bits_df) else ""
        return bits_df, segs, bits_str

    def decode_bits_only(self):
        """Ręczne wywołanie dekodera dla aktualnej ramki — WYŁĄCZNIE > threshold."""
        if self.fs is None or self.x is None or self.det_sig_win is None:
            messagebox.showwarning(APP_TITLE, "Najpierw wczytaj plik i uruchom analizę ramki.")
            return
        fs = self.fs
        s0 = self.s0; s1 = self.s1
        low = float(self.bp_low_var.get()); high = float(self.bp_high_var.get()); order = int(self.bp_order_var.get())
        nyq = 0.5 * fs
        if self.xf is not None and len(self.xf) == len(self.x):
            xf_win = self.xf[s0:s1]
        else:
            try:
                if 0 < low < high < nyq:
                    xf_win = apply_bandpass(self.x[s0:s1], fs, low, high, order)
                else:
                    xf_win = self.x[s0:s1]
            except Exception:
                xf_win = self.x[s0:s1]

        bits_df, segs, bits_str = self._decode_bits_from_mask(xf_win, self.det_sig_win, fs, s0)
        self.bits_df = bits_df
        self.bits_segments = segs
        self.bits_string = bits_str
        self.update_bits_table_and_stats()
        self.update_window_plots()
        self._set_status(f"Dekodowanie bitów (tylko > threshold). Znaleziono {len(bits_df)}.")

    # ---------- Analysis (WINDOW ONLY) ----------
    def analyze_window(self):
        path = self.wav_path_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror(APP_TITLE, "Wskaż istniejący plik WAV.")
            return
        self._set_status("Analiza ramki...")
        try:
            fs, x = read_wav(path)
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Błąd odczytu WAV: {e}")
            self._set_status("Błąd odczytu WAV"); return
        self.fs, self.x = fs, x
        self.nyquist_var.set(f"Nyquist: {0.5*fs:.1f} Hz  (fs={fs} Hz; oczekiwane 192000 Hz)")
        if fs != 192000:
            self._set_status(f"UWAGA: fs={fs} Hz (różne od 192000 Hz) — kontynuuję.")

        # Global signals
        self.xf, self.env_full, self.filter_state, self.filter_msg = self._prepare_global_signals(x, fs)
        if self.filter_state != "ok":
            self._set_status(f"Filtr globalny: {self.filter_state} — {self.filter_msg}")

        # Window indices
        win = max(0.001, float(self.window_len_var.get())); start = float(self.window_start_var.get())
        s0 = int(max(0, np.floor(start * fs))); s1 = int(min(len(x), np.ceil((start + win) * fs)))
        if s1 <= s0: s1 = min(len(x), s0 + 1)
        self.s0, self.s1 = s0, s1

        # Window slice — prefer global filtered, else lokalnie
        low = float(self.bp_low_var.get()); high = float(self.bp_high_var.get()); order = int(self.bp_order_var.get())
        nyq = 0.5 * fs
        if self.xf is not None and len(self.xf) == len(self.x):
            xf_win = self.xf[s0:s1]; slice_filtered = True
        else:
            try:
                if 0 < low < high < nyq:
                    xf_win = apply_bandpass(self.x[s0:s1], fs, low, high, order)
                    slice_filtered = True
                else:
                    xf_win = self.x[s0:s1]; slice_filtered = False
            except Exception:
                xf_win = self.x[s0:s1]; slice_filtered = False

        env_win = envelope_hilbert(xf_win)
        absf_win = np.abs(xf_win)

        # Detection signal & threshold
        domain = self.thr_domain_var.get()
        det_sig = env_win if domain == "envelope" else absf_win
        det_label = "envelope" if domain == "envelope" else ("|filtered|" if slice_filtered else "|raw|")
        mode = self.thr_mode_var.get()
        thr_factor = float(self.thr_factor_var.get()); thr_abs = float(self.thr_abs_var.get())
        thr_win = float(np.median(det_sig) * thr_factor) if mode == "relative" else float(thr_abs)

        min_sep = float(self.min_sep_ms_var.get()); min_width = float(self.min_width_ms_var.get()); pad_ms = float(self.burst_pad_ms_var.get())
        min_distance = int((min_sep / 1000.0) * fs)
        min_width_samp = int((min_width / 1000.0) * fs)
        if min_distance < 1: min_distance = 1
        if min_width_samp < 1: min_width_samp = 1
        peaks_loc, _ = find_peaks(det_sig, height=thr_win, distance=min_distance, width=min_width_samp)
        peaks_abs = s0 + peaks_loc

        # Intervals & repetition frequency
        peak_times = peaks_abs / fs
        ibi = np.diff(peak_times) if len(peak_times) > 1 else np.array([])
        rep_freqs = 1.0 / ibi if len(ibi) else np.array([])

        # Segment & intra-burst f0
        method = self.fft_method_var.get(); zp_mult = int(self.fft_zp_var.get())
        burst_bounds = []
        intraburst_freqs = []; burst_durations_ms = []
        for p_loc in peaks_loc:
            local_thr = 0.5 * det_sig[int(p_loc)]
            L = int(p_loc);  R = int(p_loc)
            while L > 0 and det_sig[L] > local_thr: L -= 1
            Nw = len(det_sig)
            while R < Nw - 1 and det_sig[R] > local_thr: R += 1
            pad = int((pad_ms / 1000.0) * fs)
            L = max(0, L - pad); R = min(Nw, R + pad)
            L_abs = s0 + L; R_abs = s0 + R
            burst_bounds.append((L_abs, R_abs))
            dur_ms = (R_abs - L_abs) * 1000.0 / fs
            burst_durations_ms.append(dur_ms)
            seg = xf_win[L:R]
            f0 = dominant_freq(seg, fs, method=method, zp_mult=zp_mult)
            intraburst_freqs.append(f0)
        intraburst_freqs = np.array(intraburst_freqs)
        burst_durations_ms = np.array(burst_durations_ms)

        # Bit decoding on this window — WYŁĄCZNIE maska progu
        self.thr_win = thr_win  # ustaw zanim użyjemy _decode_bits_from_mask
        bits_df, segs, bits_str = self._decode_bits_from_mask(xf_win, det_sig, fs, s0)
        bits_mode = "threshold_only"

        # Summaries
        summary = {
            "fs_Hz": fs,
            "window_start_s": float(start),
            "window_len_s": float(win),
            "num_bursts": int(len(peaks_abs)),
            "mean_inter_burst_interval_ms": float(np.mean(ibi)*1000.0) if len(ibi) else np.nan,
            "median_inter_burst_interval_ms": float(np.median(ibi)*1000.0) if len(ibi) else np.nan,
            "mean_repetition_frequency_Hz": float(np.mean(rep_freqs)) if len(rep_freqs) else np.nan,
            "median_repetition_frequency_Hz": float(np.median(rep_freqs)) if len(rep_freqs) else np.nan,
            "median_intraburst_frequency_Hz": float(np.nanmedian(intraburst_freqs)) if len(intraburst_freqs) else np.nan,
            "median_burst_duration_ms": float(np.nanmedian(burst_durations_ms)) if len(burst_durations_ms) else np.nan,
            "threshold_used": float(thr_win),
            "threshold_mode": mode,
            "threshold_domain": det_label,
            "filter_state": self.filter_state,
            "num_bits": int(len(bits_df)),
            "bits_str": bits_str,
            "bits_mode": bits_mode
        }

        per_burst = pd.DataFrame({
            "burst_index": np.arange(len(peaks_abs)),
            "peak_time_s_abs": peak_times.astype(float),
            "burst_duration_ms": burst_durations_ms.astype(float),
            "intraburst_f0_Hz": intraburst_freqs.astype(float)
        })

        # Store
        self.env_win = env_win
        self.det_sig_win = det_sig
        self.thr_win = thr_win
        self.peaks_win_abs = peaks_abs
        self.peak_times = peak_times
        self.ibi = ibi
        self.rep_freqs = rep_freqs
        self.burst_bounds = burst_bounds
        self.intraburst_freqs = intraburst_freqs
        self.burst_durations_ms = burst_durations_ms
        self.summary = summary
        self.per_burst_df = per_burst
        self.bits_df = bits_df
        self.bits_segments = segs
        self.bits_string = bits_str

        # Update GUI
        self.update_stats()
        self.populate_bursts()
        self.update_bits_table_and_stats()
        self.update_global_plot()
        self.update_window_plots()
        if len(per_burst):
            self.select_tree_index(0)
        self._set_status(f"Dekodowanie bitów: {bits_mode}, liczba bitów = {len(bits_df)}.")

    # ---------- UI updates ----------
    def update_stats(self):
        s = self.summary or {}
        self.stats_vars["fs"].set(self._fmt_num(s.get('fs_Hz')))
        self.stats_vars["win_start"].set(self._fmt_num(s.get('window_start_s')))
        self.stats_vars["win_len"].set(self._fmt_num(s.get('window_len_s')))
        self.stats_vars["num_bursts"].set(f"{s.get('num_bursts', '-')}")
        self.stats_vars["mean_ibi_ms"].set(self._fmt_num(s.get('mean_inter_burst_interval_ms')))
        self.stats_vars["median_ibi_ms"].set(self._fmt_num(s.get('median_inter_burst_interval_ms')))
        self.stats_vars["mean_rep_hz"].set(self._fmt_num(s.get('mean_repetition_frequency_Hz')))
        self.stats_vars["median_rep_hz"].set(self._fmt_num(s.get('median_repetition_frequency_Hz')))
        self.stats_vars["median_intra_hz"].set(self._fmt_num(s.get('median_intraburst_frequency_Hz')))
        thr_show = f"{self._fmt_num(s.get('threshold_used'))} ({s.get('threshold_mode','-')}, {s.get('threshold_domain','-')})"
        if "median_burst_duration_ms" in s and s["median_burst_duration_ms"] == s["median_burst_duration_ms"]:
            thr_show += f" | med. czas bursta: {s['median_burst_duration_ms']:.3f} ms"
        self.stats_vars["thr_win"].set(thr_show)
        self.stats_vars["num_bits"].set(f"{s.get('num_bits','-')}")
        self.stats_vars["bits_str"].set(s.get('bits_str','-') if s.get('bits_str','') else "-")

    def populate_bursts(self):
        for it in self.tree.get_children():
            self.tree.delete(it)
        for i in range(len(self.peaks_win_abs)):
            t = float(self.peak_times[i])
            dur = float(self.burst_durations_ms[i]) if i < len(self.burst_durations_ms) else np.nan
            f0 = float(self.intraburst_freqs[i]) if i < len(self.intraburst_freqs) else np.nan
            self.tree.insert("", "end", values=(i, f"{t:.6f}", self._fmt_num(dur), self._fmt_num(f0)))

        self.ax2.cla()
        if len(self.ibi):
            self.ax2.hist(self.ibi * 1000.0, bins=20)
        self.ax2.set_title("Histogram IBI (TYLKO ramka)")
        self.ax2.set_xlabel("IBI [ms]")
        self.ax2.set_ylabel("Liczność")
        self.canvas2.draw_idle()

    def update_bits_table_and_stats(self):
        for it in self.bits_tree.get_children():
            self.bits_tree.delete(it)
        if self.bits_df is None or self.bits_df.empty:
            return
        for _, row in self.bits_df.iterrows():
            vals = (int(row["bidx"]), f"{row['t_s_abs']:.6f}", int(row["bit"]),
                    self._fmt_num(row["f0_est"]), self._fmt_num(row["P0"]),
                    self._fmt_num(row["P1"]), self._fmt_num(row["conf"]))
            self.bits_tree.insert("", "end", values=vals)

    def update_global_plot(self):
        self.ax1.cla()
        self.ax1.set_title("PODGLĄD GLOBALNY — bandpass/raw + envelope (z ramką)")
        self.ax1.set_xlabel("Czas [s]"); self.ax1.set_ylabel("Amplituda / Obwiednia")
        if self.fs is None or self.x is None or self.env_full is None:
            self.canvas1.draw_idle(); return
        fs = self.fs
        pct = float(self.plot_pct_var.get())
        t = np.arange(len(self.x)) / fs
        idx = indices_for_pct(len(t), pct)

        if self.xf is not None and len(self.xf) == len(self.x):
            y = self.xf
            self.ax1.plot(t[idx], y[idx], label="bandpass")
        else:
            y = self.x
            self.ax1.plot(t[idx], y[idx], label="raw")

        e = self.env_full
        if len(e) == len(t):
            self.ax1.plot(t[idx], e[idx], label="envelope")
        else:
            # awaryjnie
            self.ax1.plot(t[idx], e[:len(idx)], label="envelope")

        win = max(0.001, float(self.window_len_var.get())); start = float(self.window_start_var.get())
        self.ax1.axvline(start, linestyle="--"); self.ax1.axvline(start + win, linestyle="--")
        self.ax1.legend()
        self.canvas1.draw_idle()

    def update_window_plots(self):
        self.ax4.cla(); self.ax5.cla(); self.ax6.cla()
        self.ax4.set_title("Surowy sygnał — widok okna"); self.ax4.set_xlabel("Czas [s]"); self.ax4.set_ylabel("Amplituda")
        self.ax5.set_title("Sygnał po filtracji — widok okna (detekcja, próg, piki, bity)")
        self.ax5.set_xlabel("Czas [s]"); self.ax5.set_ylabel("Amplituda / Obwiednia")
        self.ax6.set_title("SPEKTROGRAM (ramka, przefiltrowany, tylko pasmo BP)")
        self.ax6.set_xlabel("Czas [s]"); self.ax6.set_ylabel("Częstotliwość [Hz]")

        if self.fs is None or self.x is None:
            self.canvas4.draw_idle(); self.canvas5.draw_idle(); self.canvas6.draw_idle(); return

        fs = self.fs
        pct = float(self.plot_pct_var.get())
        win = max(0.001, float(self.window_len_var.get())); start = float(self.window_start_var.get())
        s0 = int(max(0, np.floor(start * fs))); s1 = int(min(len(self.x), np.ceil((start + win) * fs)))
        if s1 <= s0: s1 = min(len(self.x), s0 + 1)
        t = np.arange(s0, s1) / fs
        n = len(t)
        idx = indices_for_pct(n, pct)

        # raw
        self.ax4.plot(t[idx], self.x[s0:s1][idx])

        # filtered slice (prefer global)
        low = float(self.bp_low_var.get()); high = float(self.bp_high_var.get()); order = int(self.bp_order_var.get())
        nyq = 0.5 * fs
        if self.xf is not None and len(self.xf) == len(self.x):
            xf_slice = self.xf[s0:s1]; slice_filtered = True
        else:
            try:
                if 0 < low < high < nyq:
                    xf_slice = apply_bandpass(self.x[s0:s1], fs, low, high, order)
                    slice_filtered = True
                else:
                    xf_slice = self.x[s0:s1]; slice_filtered = False
            except Exception:
                xf_slice = self.x[s0:s1]; slice_filtered = False

        # wykres 5: sygnał przefiltrowany (cienki rysunek)
        self.ax5.plot(t[idx], xf_slice[idx], label=("bandpass" if slice_filtered else "raw (no filter)"))
        if slice_filtered:
            msg = f"Filtr OK: {low:.0f}–{high:.0f} Hz (Nyquist {nyq:.0f} Hz)"
        else:
            msg = f"FILTR WYŁĄCZONY: {self.filter_msg}" if getattr(self, "filter_msg", "") else "FILTR WYŁĄCZONY"
        self.ax5.text(0.01, 0.98, msg, transform=self.ax5.transAxes, va="top", ha="left")

        # detekcja + próg + piki (TYLKO jeśli analiza tej ramki)
        if self.det_sig_win is not None and self.s0 == s0 and self.s1 == s1:
            if self.show_env_in_window_var.get():
                self.ax5.plot(t[idx], self.det_sig_win[idx], label="detekcja (env/|xf|)")
            if self.thr_win is not None:
                y_high = self.det_sig_win.copy()
                y_high[self.det_sig_win <= self.thr_win] = np.nan
                self.ax5.plot(t[idx], y_high[idx], linewidth=2, label="> threshold")
                if self.thr_domain_var.get() == "abs_filtered" and self.thr_show_pm_var.get():
                    self.ax5.axhline(self.thr_win, linestyle="--", label="+threshold")
                    self.ax5.axhline(-self.thr_win, linestyle="--", label="-threshold")
                else:
                    self.ax5.axhline(self.thr_win, linestyle="--", label="threshold")
            if len(self.peaks_win_abs):
                mask = (self.peaks_win_abs >= s0) & (self.peaks_win_abs < s1)
                pk_times = self.peaks_win_abs[mask] / fs
                if self.det_sig_win is not None:
                    det_vals = self.det_sig_win[(self.peaks_win_abs[mask] - s0)]
                    self.ax5.plot(pk_times, det_vals, marker="o", linestyle="None", label="peaks")

        # Wykres 6: SPEKTROGRAM tylko w paśmie BP (redukcja kolumn wg pct)
        nperseg = 1024
        nperseg_eff = min(nperseg, len(xf_slice)) if len(xf_slice) else nperseg
        if nperseg_eff >= 8:
            noverlap = int(0.75 * nperseg_eff)
            if noverlap >= nperseg_eff:
                noverlap = max(0, nperseg_eff // 2)
            f_spec, t_spec, Sxx = spectrogram(
                xf_slice, fs=fs, nperseg=nperseg_eff, noverlap=noverlap, nfft=nperseg_eff,
                scaling='density', mode='psd', detrend=False
            )
            # Redukcja liczby kolumn wg pct (tylko rysowanie)
            if t_spec.size > 0:
                cols = max(8, int(np.ceil(t_spec.size * max(0.001, min(pct, 100.0))/100.0)))
                if cols < t_spec.size:
                    cidx = np.linspace(0, t_spec.size-1, num=cols, dtype=int)
                    t_spec = t_spec[cidx]
                    Sxx = Sxx[:, cidx]

            fmask = (f_spec >= low) & (f_spec <= high)
            if not np.any(fmask):
                fmask = slice(None, None, None)
            S = Sxx[fmask, :]
            eps = 1e-12
            S_dB = 10.0 * np.log10(S + eps)
            if t_spec.size:
                t0 = float(self.window_start_var.get())
                t_abs = t0 + t_spec
                extent = [t_abs[0], t_abs[-1], float(f_spec[fmask][0]), float(f_spec[fmask][-1])]
            else:
                extent = [float(self.window_start_var.get()), float(self.window_start_var.get()), float(low), float(high)]
            self.ax6.imshow(S_dB, origin='lower', aspect='auto', extent=extent)
            self.ax6.set_ylim(low, high)
            self.ax6.set_title(f"SPEKTROGRAM (nperseg=1024, pasmo {low:.0f}–{high:.0f} Hz)")
            self.ax6.set_xlabel("Czas [s]"); self.ax6.set_ylabel("Częstotliwość [Hz]")

            # linie referencyjne częstotliwości bitowych
            f0_0 = float(self.bit_f0_low_var.get())
            f0_1 = float(self.bit_f0_high_var.get())
            self.ax6.axhline(f0_0, linestyle="--", alpha=0.3)
            self.ax6.axhline(f0_1, linestyle="--", alpha=0.3)

        # Oznacz bity (jeśli są)
        if self.bits_segments and self.s0 == s0 and self.s1 == s1 and self.bits_df is not None:
            ylim5 = self.ax5.get_ylim()
            low = float(self.bp_low_var.get()); high = float(self.bp_high_var.get())
            for idx_b, (seg_pair) in enumerate(self.bits_segments):
                if idx_b >= len(self.bits_df):
                    break
                L_abs, R_abs = seg_pair
                row = self.bits_df.iloc[idx_b]
                tL = L_abs / fs; tR = R_abs / fs; tc = 0.5*(tL+tR); b = int(row["bit"])
                self.ax5.axvline(tL, linestyle=":", alpha=0.7)
                self.ax5.axvline(tR, linestyle=":", alpha=0.7)
                self.ax5.text(tc, ylim5[1]*0.9, str(b), ha="center", va="top")
                # na spektrogramie opcjonalnie linie pionowe:
                self.ax6.axvline(tL, linestyle=":", alpha=0.7)
                self.ax6.axvline(tR, linestyle=":", alpha=0.7)
                self.ax6.text(tc, low + 0.9*(high-low), str(b), ha="center", va="top")

        self.ax5.legend()
        self.canvas4.draw_idle(); self.canvas5.draw_idle(); self.canvas6.draw_idle()

    def on_select_burst(self, event):
        sel = self.tree.selection()
        if not sel: return
        item = sel[0]; vals = self.tree.item(item, "values")
        if not vals: return
        idx = int(vals[0])
        self.update_fft_plot(selected_index=idx)

    def select_tree_index(self, idx):
        children = self.tree.get_children()
        if not children: return
        idx = max(0, min(idx, len(children)-1))
        iid = children[idx]; self.tree.selection_set(iid); self.tree.see(iid)
        self.on_select_burst(None)

    def update_fft_plot(self, selected_index=None):
        self.ax3.cla()
        self.ax3.set_title("FFT wybranego bursta (TYLKO ramka)")
        self.ax3.set_xlabel("Częstotliwość [Hz]"); self.ax3.set_ylabel("Magnituda")
        if not self.burst_bounds:
            self.canvas3.draw_idle(); return

        if selected_index is None:
            sel = self.tree.selection()
            if sel:
                item = sel[0]; vals = self.tree.item(item, "values")
                if vals: selected_index = int(vals[0])
            else:
                selected_index = 0

        selected_index = max(0, min(selected_index, len(self.burst_bounds)-1))
        L_abs, R_abs = self.burst_bounds[selected_index]
        fs = self.fs
        low = float(self.bp_low_var.get()); high = float(self.bp_high_var.get()); order = int(self.bp_order_var.get())
        nyq = 0.5 * fs
        if self.xf is not None and len(self.xf) == len(self.x):
            seg = self.xf[L_abs:R_abs]
        else:
            try:
                if 0 < low < high < nyq:
                    seg = apply_bandpass(self.x[L_abs:R_abs], fs, low, high, order)
                else:
                    seg = self.x[L_abs:R_abs]
            except Exception:
                seg = self.x[L_abs:R_abs]

        f0 = dominant_freq(seg, fs, method=self.fft_method_var.get(), zp_mult=int(self.fft_zp_var.get()))
        N = len(seg)
        if N > 16:
            win = get_window("hann", N, fftbins=True)
            nfft = int(2**np.ceil(np.log2(N))) * int(max(1, int(self.fft_zp_var.get())))
            spec = np.abs(np.fft.rfft(seg * win, n=nfft))
            freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
            if len(spec) > 1: spec[0] = 0.0
            self.ax3.plot(freqs, spec); self.ax3.set_xlim(0, fs/2.0)
        self.ax3.set_title(f"FFT bursta #{selected_index} (dominująca ≈ {self._fmt_num(f0)} Hz)")
        self.canvas3.draw_idle()

    # ---------- Presets ----------
    def _collect_preset(self):
        return dict(
            bp_low=float(self.bp_low_var.get()), bp_high=float(self.bp_high_var.get()), bp_order=int(self.bp_order_var.get()),
            min_sep_ms=float(self.min_sep_ms_var.get()), min_width_ms=float(self.min_width_ms_var.get()), pad_ms=float(self.burst_pad_ms_var.get()),
            thr_domain=self.thr_domain_var.get(), thr_mode=self.thr_mode_var.get(),
            thr_factor=float(self.thr_factor_var.get()), thr_abs=float(self.thr_abs_var.get()),
            win_len=float(self.window_len_var.get()), win_start=float(self.window_start_var.get()),
            fft_method=self.fft_method_var.get(), fft_zp=int(self.fft_zp_var.get()),
            global_filter_mode=self.global_filter_mode_var.get(), global_auto_limit=float(self.global_filter_auto_max_sec_var.get()),
            plot_pct=float(self.plot_pct_var.get()),
            bit_on_ms=float(self.bit_len_ms_var.get()), bit_gap_ms=float(self.bit_gap_ms_var.get()),
            bit_f0_0=float(self.bit_f0_low_var.get()), bit_f0_1=float(self.bit_f0_high_var.get()),
            bit_bw=float(self.bit_bw_hz_var.get())
        )

    def _apply_preset(self, p):
        try:
            self.bp_low_var.set(float(p["bp_low"])); self.bp_high_var.set(float(p["bp_high"])); self.bp_order_var.set(int(p["bp_order"]))
            self.min_sep_ms_var.set(float(p["min_sep_ms"])); self.min_width_ms_var.set(float(p["min_width_ms"])); self.burst_pad_ms_var.set(float(p["pad_ms"]))
            self.thr_domain_var.set(p["thr_domain"]); self.thr_mode_var.set(p["thr_mode"])
            self.thr_factor_var.set(float(p["thr_factor"])); self.thr_abs_var.set(float(p["thr_abs"]))
            self.window_len_var.set(float(p["win_len"])); self.window_start_var.set(float(p["win_start"]))
            if "fft_method" in p: self.fft_method_var.set(p["fft_method"])
            if "fft_zp" in p: self.fft_zp_var.set(int(p["fft_zp"]))
            if "global_filter_mode" in p: self.global_filter_mode_var.set(p["global_filter_mode"])
            if "global_auto_limit" in p: self.global_filter_auto_max_sec_var.set(float(p["global_auto_limit"]))
            if "plot_pct" in p: self.plot_pct_var.set(float(p["plot_pct"]))
            if "bit_on_ms" in p: self.bit_len_ms_var.set(float(p["bit_on_ms"]))
            if "bit_gap_ms" in p: self.bit_gap_ms_var.set(float(p["bit_gap_ms"]))
            if "bit_f0_0" in p: self.bit_f0_low_var.set(float(p["bit_f0_0"]))
            if "bit_f0_1" in p: self.bit_f0_high_var.set(float(p["bit_f0_1"]))
            if "bit_bw" in p: self.bit_bw_hz_var.set(float(p["bit_bw"]))
            self._on_thr_ui_change(); self.configure_window_scale(); self.update_window_plots()
            self.maybe_auto_analyze()
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Nieprawidłowy preset: {e}")

    def save_preset(self):
        p = self._collect_preset()
        path = filedialog.asksaveasfilename(title="Zapisz preset", defaultextension=".json",
                                            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
                                            initialfile="burst_preset.json")
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as f: json.dump(p, f, ensure_ascii=False, indent=2)
            self._set_status("Preset zapisany.")
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Błąd zapisu presetu: {e}")

    def load_preset(self):
        path = filedialog.askopenfilename(title="Wczytaj preset", filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f: p = json.load(f)
            self._apply_preset(p); self._set_status("Preset wczytany.")
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Błąd wczytania presetu: {e}")

    # ---------- Save / Export ----------
    def save_csv(self):
        if self.per_burst_df is None:
            messagebox.showwarning(APP_TITLE, "Brak wyników do zapisania. Najpierw uruchom analizę ramki.")
            return
        path = filedialog.asksaveasfilename(
            title="Zapisz CSV z wynikami (ramka)",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialfile="burst_results_window.csv"
        )
        if not path: return
        try:
            base, ext = os.path.splitext(path)
            per_burst_path = base + "_per_burst.csv"
            summary_path = base + "_summary.json"
            bits_path = base + "_bits.csv"
            self.per_burst_df.to_csv(per_burst_path, index=False)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(self.summary, f, ensure_ascii=False, indent=2)
            if self.bits_df is not None and not self.bits_df.empty:
                self.bits_df.to_csv(bits_path, index=False)
                saved_bits = f"\n{bits_path}"
            else:
                saved_bits = ""
            messagebox.showinfo(APP_TITLE, f"Zapisano:\n{per_burst_path}\n{summary_path}{saved_bits}")
            self._set_status("Wyniki zapisane.")
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Błąd zapisu CSV: {e}")

    def save_pngs(self):
        if self.x is None:
            messagebox.showwarning(APP_TITLE, "Brak wykresów do zapisania. Najpierw wczytaj i przeanalizuj ramkę.")
            return
        path = filedialog.asksaveasfilename(
            title="Wybierz bazową nazwę pliku PNG",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")],
            initialfile="burst_plots_window.png"
        )
        if not path: return
        base, ext = os.path.splitext(path)
        try:
            self.fig1.savefig(base + "_global_overview.png", dpi=150, bbox_inches="tight")
            self.fig2.savefig(base + "_ibi_hist_window.png", dpi=150, bbox_inches="tight")
            self.fig3.savefig(base + "_fft_window.png", dpi=150, bbox_inches="tight")
            self.fig4.savefig(base + "_raw_window.png", dpi=150, bbox_inches="tight")
            self.fig5.savefig(base + "_filtered_window.png", dpi=150, bbox_inches="tight")
            self.fig6.savefig(base + "_spectrogram_window.png", dpi=150, bbox_inches="tight")
            messagebox.showinfo(APP_TITLE, "Wykresy zapisane.")
            self._set_status("Wykresy zapisane.")
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Błąd zapisu PNG: {e}")

    def export_window_wav(self):
        if self.fs is None or self.x is None:
            messagebox.showwarning(APP_TITLE, "Brak danych. Wczytaj plik i ustaw okno.")
            return
        fs = self.fs; win = max(0.001, float(self.window_len_var.get())); start = float(self.window_start_var.get())
        s0 = int(max(0, np.floor(start * fs))); s1 = int(min(len(self.x), np.ceil((start + win) * fs)))
        if s1 <= s0: s1 = min(len(self.x), s0 + 1)
        path = filedialog.asksaveasfilename(
            title="Zapisz okno do WAV", defaultextension=".wav",
            filetypes=[("WAV", "*.wav"), ("All files", "*.*")],
            initialfile="burst_window.wav"
        )
        if not path: return
        try:
            write_wav(path, fs, self.x[s0:s1])
            messagebox.showinfo(APP_TITLE, "Okno zapisane do WAV.")
            self._set_status("Okno zapisane do WAV.")
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Błąd zapisu WAV: {e}")

    # ---------- Utils ----------
    @staticmethod
    def _fmt_num(x):
        try:
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                return "-"
            return f"{float(x):.3f}"
        except Exception:
            return "-"

# ---- main ----
def main():
    app = BurstAnalyzerApp()
    app.mainloop()

if __name__ == "__main__":
    main()
