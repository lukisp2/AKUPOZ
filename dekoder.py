#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSK 49.6/50.6 kHz (24 bity) + Detektor impulsów 1 s (szpilki)

Nowości:
- Stopa czasu z pliku (time-base):
  • WAV/BWF: pobieram OriginationDate + OriginationTime z chunka 'bext'
  • fallback: FS mtime pliku
  • GUI pokazuje źródło i ISO start
  • CSV: absolutne czasy zdarzeń (ISO) + źródło/ISO time-base na wiersz

- Impulsy:
  • cluster_count_1s liczone z tolerancją: [t, t + period_s*(1+tol)) (domyślnie 1.05 s)
  • period_ok nie jest zaniżane na końcach okna (look‑ahead i cross‑frame)
  • suwak „Refrakcja [ms]”
  • eksport szczegółów + agregatów

- FSK:
  • dekodowanie 24 bitów (6 ms + 1 ms)
  • przypisanie do slotów UN0..UN3, BER (z inwersją jeśli lepsze)
  • spektrogram (nperseg=1024) + naniesione bity
  • eksport okna i full‑scanu (10 s)
"""

from __future__ import annotations
import sys, csv, math, os, struct, time
from datetime import datetime, timedelta
import numpy as np
import soundfile as sf
from scipy import signal

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QDoubleSpinBox,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QComboBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# ==========================
# Wzorce UN0..UN3 (FSK)
# ==========================

def _bits(s: str) -> str:
    return s.replace(" ", "").strip()

EXPECTED_PATTERNS = {
    "UN0": _bits("0010 1010 1110 0011 1000 1100"),
    "UN1": _bits("0101 0101 1110 0011 1000 1100"),
    "UN2": _bits("1010 1010 1110 0011 1000 1100"),
    "UN3": _bits("1101 0101 1110 0011 1000 1100"),
}
SLOT_TO_LABEL = {1: "UN0", 2: "UN1", 3: "UN2", 4: "UN3"}

def invert_bits(bitstr: str) -> str:
    return "".join("1" if c == "0" else "0" for c in bitstr)

def compare_to_expected(decoded_bits: str, expected_bits: str):
    if len(decoded_bits) != len(expected_bits):
        return {'polarity': None, 'errors': None, 'ber': None, 'expected_used': None}
    err_norm = sum(1 for a, b in zip(decoded_bits, expected_bits) if a != b)
    inv = invert_bits(expected_bits)
    err_inv = sum(1 for a, b in zip(decoded_bits, inv) if a != b)
    if err_inv < err_norm:
        return {'polarity': 'inverted', 'errors': err_inv, 'ber': err_inv/len(expected_bits), 'expected_used': inv}
    else:
        return {'polarity': 'normal', 'errors': err_norm, 'ber': err_norm/len(expected_bits), 'expected_used': expected_bits}

# ==========================
# DSP narzędzia
# ==========================

def ensure_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1: return x.astype(np.float64, copy=False)
    return np.mean(x, axis=1).astype(np.float64, copy=False)

def bandpass_sos(low_hz: float, high_hz: float, fs: float, order: int = 6):
    nyq = 0.5*fs
    if high_hz >= nyq: raise ValueError(f"high_hz={high_hz} ≥ Nyquist ({nyq} Hz) dla fs={fs} Hz.")
    if low_hz <= 0: raise ValueError("low_hz musi być > 0.")
    wp = [low_hz/nyq, high_hz/nyq]
    return signal.butter(order, wp, btype='bandpass', output='sos')

def apply_bandpass(x: np.ndarray, fs: float, low_hz: float = 48_000.0, high_hz: float = 52_000.0) -> np.ndarray:
    sos = bandpass_sos(low_hz, high_hz, fs, order=6)
    return signal.sosfiltfilt(sos, x)

def analytic_envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(signal.hilbert(x))

def db_from_lin(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return 20.0*np.log10(np.maximum(x, floor))

def hann(n: int) -> np.ndarray:
    return signal.windows.hann(n, sym=False)

def tone_power_dot(x: np.ndarray, f: float, fs: float) -> float:
    n = x.size
    if n == 0: return 0.0
    t = np.arange(n, dtype=np.float64)/fs
    w = hann(n); xx = x*w
    cs = np.cos(2*np.pi*f*t); sn = np.sin(2*np.pi*f*t)
    c = float(np.dot(xx, cs)); s = float(np.dot(xx, sn))
    return c*c + s*s

def peak_near_freq(segment: np.ndarray, fs: float, f_target: float, search_hz: float = 300.0) -> float:
    n = segment.size
    nfft = 4096 if n < 2048 else 1 << (int(np.ceil(np.log2(n))) + 1)
    wlen = min(n, 8192)
    w = hann(wlen)
    xw = (segment[:wlen] * w) if n > wlen else (segment * hann(n))
    X = np.fft.rfft(xw, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
    band = (freqs >= (f_target - search_hz)) & (freqs <= (f_target + search_hz))
    if not np.any(band): return f_target
    idx = np.argmax(np.abs(X[band])**2)
    return float(freqs[band][idx])

# ==========================
# Dekoder FSK (24 bity)
# ==========================

def decode_fsk_segment(
    x: np.ndarray, fs: float,
    f0_nom: float = 49_600.0, f1_nom: float = 50_600.0,
    autocal: bool = True, bit_ms: float = 6.0, gap_ms: float = 1.0,
    n_bits: int = 24, search_step_ms: float = 0.25
):
    if autocal:
        f0 = peak_near_freq(x, fs, f0_nom, 300.0)
        f1 = peak_near_freq(x, fs, f1_nom, 300.0)
    else:
        f0, f1 = f0_nom, f1_nom

    bit_s = bit_ms/1000.0; gap_s = gap_ms/1000.0; sym_s = bit_s + gap_s
    n_needed = int(round(n_bits * sym_s * fs))
    if x.size < int(0.5*n_needed):
        return {'ok': False, 'reason': 'segment too short', 'f0': f0, 'f1': f1}

    best = None
    steps = max(1, int(round(sym_s / (search_step_ms/1000.0))))
    for i in range(steps):
        offset = (i/steps) * sym_s
        bits = []; margin_sum = 0.0; ok = True
        for k in range(n_bits):
            t0 = offset + k*sym_s; t1 = t0 + bit_s
            i0 = int(round(t0*fs)); i1 = int(round(t1*fs))
            if i1 > x.size or i0 < 0: ok = False; break
            chunk = x[i0:i1]
            p0 = tone_power_dot(chunk, f0, fs); p1 = tone_power_dot(chunk, f1, fs)
            bits.append('1' if p1 > p0 else '0'); margin_sum += abs(p1 - p0)
        if ok:
            cand = (margin_sum, offset, ''.join(bits))
            if best is None or cand[0] > best[0]: best = cand

    if best is None:
        return {'ok': False, 'reason': 'no alignment found', 'f0': f0, 'f1': f1}

    margin, best_offset, bits = best

    def bits_to_hex(bitstr: str, flip=False) -> str:
        if len(bitstr) % 8: return ""
        out = []
        for i in range(0, len(bitstr), 8):
            b = bitstr[i:i+8]; b = b[::-1] if flip else b
            out.append(f"{int(b,2):02X}")
        return ''.join(out)

    return {
        'ok': True, 'start_offset_ms': best_offset*1000.0,
        'f0': f0, 'f1': f1, 'bits': bits,
        'hex_msb': bits_to_hex(bits, False), 'hex_lsbflip': bits_to_hex(bits, True),
        'margin': margin
    }

# ==========================
# Tryb impulsów 1 s
# ==========================

def detect_spikes_onsets(env: np.ndarray, fs: float, thr: float, refractory_ms: float = 20.0):
    cross = np.flatnonzero((env[:-1] < thr) & (env[1:] >= thr)) + 1
    if cross.size == 0: return cross
    min_gap = int(round((refractory_ms/1000.0)*fs))
    keep = [int(cross[0])]; last = int(cross[0])
    for idx in cross[1:]:
        if idx - last >= min_gap:
            keep.append(int(idx)); last = int(idx)
    return np.array(keep, dtype=np.int64)

def spikes_metrics(onsets: np.ndarray, env: np.ndarray, fs: float,
                   peak_win_ms: float = 200.0, period_s: float = 1.0, period_tol: float = 0.05):
    """
    Lista dictów: onset_s (rel.), peak_env, prev_dt_s, next_dt_s,
    cluster_count_1s (z tolerancją), valid_spike, period_ok.
    """
    res = []; n = env.size; win = int(round((peak_win_ms/1000.0)*fs))
    cluster_horizon = period_s*(1.0+period_tol)  # np. 1.05 s
    on_times = onsets / fs
    for i, o in enumerate(onsets):
        i0, i1 = o, min(n, o+win)
        peak = float(np.max(env[i0:i1])) if i0 < i1 else float(env[i0])
        t = on_times[i]
        prev_dt = (on_times[i] - on_times[i-1]) if i>0 else np.nan
        next_dt = (on_times[i+1] - on_times[i]) if i < len(onsets)-1 else np.nan
        count_tol = int(np.sum((on_times >= t) & (on_times < t + cluster_horizon)))
        period_ok = (not np.isnan(next_dt)) and (abs(next_dt - period_s) <= period_s*period_tol)
        res.append({
            'onset_s': t, 'peak_env': peak,
            'prev_dt_s': prev_dt, 'next_dt_s': next_dt,
            'cluster_count_1s': count_tol,          # liczone z tolerancją
            'valid_spike': (count_tol == 1),        # brak drugiego przebicia w horyzoncie
            'period_ok': bool(period_ok)
        })
    return res

def patch_last_spike_crossframe(metrics: list[dict],
                                last_onset_abs: float,
                                next_onsets_abs: np.ndarray,
                                period_s: float, period_tol: float):
    """Dopina last-spike o next_dt/period_ok/cluster_count_1s z kolejnej ramki (horyzont z tolerancją)."""
    if not metrics: return metrics
    cluster_horizon = period_s*(1.0+period_tol)
    last = metrics[-1]
    in_window_next = int(np.sum((next_onsets_abs >= last_onset_abs) &
                                (next_onsets_abs <  last_onset_abs + cluster_horizon)))
    current = int(last.get('cluster_count_1s', 1))
    cluster_total = current + in_window_next
    mask_after = next_onsets_abs > last_onset_abs
    next_dt = float(np.min(next_onsets_abs[mask_after]) - last_onset_abs) if np.any(mask_after) else np.nan
    if not np.isnan(next_dt):
        last['next_dt_s'] = next_dt
        last['period_ok'] = (abs(next_dt - period_s) <= period_s*period_tol)
    last['cluster_count_1s'] = cluster_total
    last['valid_spike'] = (cluster_total == 1)
    return metrics

# ==========================
# Time-base (BWF / mtime)
# ==========================

def parse_wav_bext_timebase(filepath: str):
    """Zwraca dict z polami: 'dt' (datetime) oraz opcjonalnie 'time_reference_samples'. Albo None."""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(12)
            if len(header) < 12:
                return None
            # iteruj po chunkach
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                cid, csize = struct.unpack('<4sI', hdr)
                cid = cid.decode('ascii', errors='ignore')
                if cid == 'bext':
                    data = f.read(csize)
                    # wyrównanie do parzystości
                    if (csize % 2) == 1:
                        f.read(1)
                    # minimalna długość dla pól daty/czasu + timeref
                    if len(data) < 348:
                        return None
                    # pola bext
                    # 256 + 32 + 32 = 320
                    orig_date = data[320:330].decode('ascii', errors='ignore').strip('\x00').strip()
                    orig_time = data[330:338].decode('ascii', errors='ignore').strip('\x00').strip()
                    tr_low  = struct.unpack('<I', data[338:342])[0]
                    tr_high = struct.unpack('<I', data[342:346])[0]
                    time_ref = (tr_high << 32) | tr_low
                    # parsuj datetime
                    try:
                        dt = datetime.strptime(orig_date + ' ' + orig_time, '%Y-%m-%d %H:%M:%S')
                    except Exception:
                        return None
                    return {'dt': dt, 'time_reference_samples': int(time_ref)}
                else:
                    # pomiń chunk
                    f.seek(csize + (csize % 2), os.SEEK_CUR)
    except Exception:
        return None
    return None

def file_timebase(filepath: str):
    """
    Zwraca (epoch_start, source, iso_str). Preferuje WAV/BWF bext, w innym razie mtime.
    Uwaga: brak informacji o strefie → traktujemy czas jako lokalny systemowy.
    """
    base_epoch = None
    src = None
    # BWF
    bext = parse_wav_bext_timebase(filepath)
    if bext and isinstance(bext.get('dt'), datetime):
        # traktujemy OriginationDate+Time jako 'start' (time_reference bywa redundantny)
        # przelicz na epoch w czasie lokalnym
        try:
            base_epoch = time.mktime(bext['dt'].timetuple())
            src = 'BWF'
        except Exception:
            base_epoch = None
            src = None
    if base_epoch is None:
        # fallback: mtime pliku
        try:
            mtime = os.path.getmtime(filepath)
            base_epoch = float(mtime)
            src = 'mtime'
        except Exception:
            base_epoch = None
            src = 'unknown'
    iso = datetime.fromtimestamp(base_epoch).isoformat(timespec='milliseconds') if base_epoch is not None else ""
    return base_epoch, src, iso

# ==========================
# Matplotlib płótno
# ==========================

class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(constrained_layout=True)
        gs = self.fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1.2])
        super().__init__(self.fig)
        self.ax_spec = self.fig.add_subplot(gs[0, 0])
        self.ax_time = self.fig.add_subplot(gs[1, 0])

    def clear(self):
        self.ax_spec.clear(); self.ax_time.clear(); self.draw_idle()

# ==========================
# GUI
# ==========================

class DecoderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FSK 49.6/50.6 kHz & Detektor impulsów 1 s")
        self.resize(1360, 920)

        self.audio: np.ndarray | None = None
        self.fs: float | None = None
        self.filepath: str | None = None

        # Time-base
        self.tb_epoch: float | None = None
        self.tb_source: str | None = None
        self.tb_iso: str | None = None

        # Parametry pasma / okna
        self.low_hz = 48_000.0; self.high_hz = 52_000.0
        self.window_sec = 10.0; self.window_start_sec = 0.0

        # Tryby
        self.modes = ["Dekoder FSK", "Detektor impulsów 1 s"]
        self.mode = self.modes[0]

        # FSK
        self.bit_ms = 6.0; self.gap_ms = 1.0; self.n_bits = 24
        self.min_packet_ms = 200.0

        # Impulsy
        self.period_s = 1.0; self.period_tol = 0.05
        self.peak_win_ms = 200.0

        # UI
        root = QWidget(); self.setCentralWidget(root)
        main = QVBoxLayout(root)

        # Pasek narzędzi
        bar = QHBoxLayout()
        self.btn_open = QPushButton("Wczytaj plik…")
        self.combo_mode = QComboBox(); self.combo_mode.addItems(self.modes)
        self.btn_prev = QPushButton("⟵ Poprzednie okno")
        self.btn_next = QPushButton("Następne okno ⟶")
        self.btn_analyze = QPushButton("Analizuj okno")
        self.btn_analyze_full = QPushButton("Analizuj cały plik (10 s) → CSV")
        self.btn_savecsv = QPushButton("Zapisz wyniki CSV (okno)")
        bar.addWidget(self.btn_open)
        bar.addWidget(QLabel("Tryb:")); bar.addWidget(self.combo_mode)
        bar.addStretch(1); bar.addWidget(self.btn_prev); bar.addWidget(self.btn_next); bar.addStretch(1)
        bar.addWidget(self.btn_analyze); bar.addWidget(self.btn_analyze_full); bar.addWidget(self.btn_savecsv)
        main.addLayout(bar)

        # Sekcje ustawień
        cfg = QHBoxLayout()

        # Próg
        gb_thr = QGroupBox("Próg")
        lay_thr = QHBoxLayout(gb_thr)
        self.combo_thr_mode = QComboBox(); self.combo_thr_mode.addItems(["× mediana", "Stały", "Auto (percentyl)"])
        lay_thr.addWidget(QLabel("Tryb progu:")); lay_thr.addWidget(self.combo_thr_mode)
        self.spin_thr_factor = QDoubleSpinBox(); self.spin_thr_factor.setRange(0.1, 50.0); self.spin_thr_factor.setSingleStep(0.1); self.spin_thr_factor.setValue(2.0)
        self.spin_thr_abs = QDoubleSpinBox(); self.spin_thr_abs.setRange(0.0, 1e9); self.spin_thr_abs.setDecimals(6); self.spin_thr_abs.setSingleStep(0.0001); self.spin_thr_abs.setValue(0.01)
        self.spin_thr_pct = QDoubleSpinBox(); self.spin_thr_pct.setRange(80.0, 99.99); self.spin_thr_pct.setSingleStep(0.1); self.spin_thr_pct.setValue(99.5)
        lay_thr.addWidget(QLabel("× mediana:")); lay_thr.addWidget(self.spin_thr_factor)
        lay_thr.addWidget(QLabel("Stały:")); lay_thr.addWidget(self.spin_thr_abs)
        lay_thr.addWidget(QLabel("Percentyl:")); lay_thr.addWidget(self.spin_thr_pct)

        # Wykrywanie
        gb_det = QGroupBox("Wykrywanie (wspólne)")
        det = QHBoxLayout(gb_det)
        self.spin_overlap = QDoubleSpinBox(); self.spin_overlap.setRange(0.0, 20.0); self.spin_overlap.setSingleStep(0.5); self.spin_overlap.setValue(2.0)
        self.spin_merge = QDoubleSpinBox(); self.spin_merge.setRange(0.0, 10.0); self.spin_merge.setSingleStep(0.5); self.spin_merge.setValue(1.0)
        self.spin_refractory = QDoubleSpinBox(); self.spin_refractory.setRange(0.0, 500.0); self.spin_refractory.setSingleStep(1.0); self.spin_refractory.setValue(20.0)
        self.chk_autocal = QCheckBox("FSK: Autokalibracja tonów ±300 Hz"); self.chk_autocal.setChecked(True)
        det.addWidget(QLabel("Overlap [ms]:")); det.addWidget(self.spin_overlap)
        det.addWidget(QLabel("Scal przerwy ≤ [ms]:")); det.addWidget(self.spin_merge)
        det.addWidget(QLabel("Refrakcja [ms]:")); det.addWidget(self.spin_refractory)
        det.addWidget(self.chk_autocal)

        # Okno
        gb_win = QGroupBox("Okno")
        wlay = QHBoxLayout(gb_win)
        self.spin_start = QDoubleSpinBox(); self.spin_start.setRange(0.0, 1e9); self.spin_start.setDecimals(3); self.spin_start.setSingleStep(0.1); self.spin_start.setValue(0.0)
        self.spin_len = QDoubleSpinBox(); self.spin_len.setRange(1.0, 60.0); self.spin_len.setDecimals(1); self.spin_len.setSingleStep(1.0); self.spin_len.setValue(self.window_sec)
        wlay.addWidget(QLabel("Start [s]:")); wlay.addWidget(self.spin_start)
        wlay.addWidget(QLabel("Długość [s]:")); wlay.addWidget(self.spin_len)

        # Time-base
        gb_tb = QGroupBox("Czas bazowy (z pliku)")
        tlay = QHBoxLayout(gb_tb)
        self.lbl_tb_src = QLabel("Źródło: —")
        self.lbl_tb_iso = QLabel("Start: —")
        tlay.addWidget(self.lbl_tb_src)
        tlay.addWidget(self.lbl_tb_iso)

        cfg.addWidget(gb_thr, stretch=3)
        cfg.addWidget(gb_det, stretch=3)
        cfg.addWidget(gb_win, stretch=2)
        cfg.addWidget(gb_tb, stretch=3)
        main.addLayout(cfg)

        # Wykresy
        self.canvas = MplCanvas(); main.addWidget(self.canvas, stretch=3)

        # Tabela
        self.table = QTableWidget(0, 1)
        self.set_table_headers_for_mode(self.mode)
        main.addWidget(self.table, stretch=2)

        # Połączenia
        self.btn_open.clicked.connect(self.on_open)
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)
        self.btn_prev.clicked.connect(self.on_prev)
        self.btn_next.clicked.connect(self.on_next)
        self.btn_analyze.clicked.connect(self.on_analyze)
        self.btn_analyze_full.clicked.connect(self.on_analyze_full_file)
        self.btn_savecsv.clicked.connect(self.on_save_csv)
        self.spin_start.valueChanged.connect(self.on_start_changed)
        self.spin_len.valueChanged.connect(self.on_len_changed)
        # odświeżanie podglądu
        self.combo_thr_mode.currentIndexChanged.connect(lambda _v: self.update_plot())
        self.spin_thr_factor.valueChanged.connect(lambda _v: self.update_plot())
        self.spin_thr_abs.valueChanged.connect(lambda _v: self.update_plot())
        self.spin_thr_pct.valueChanged.connect(lambda _v: self.update_plot())
        self.spin_overlap.valueChanged.connect(lambda _v: self.update_plot())
        self.spin_merge.valueChanged.connect(lambda _v: self.update_plot())
        self.spin_refractory.valueChanged.connect(lambda _v: self.update_plot())

        # Bufory wyników
        self.detections_fsk: list[tuple[int,int]] = []
        self.results_fsk: list[dict] = []
        self.results_spikes: list[dict] = []

        self.update_plot()

    # ---------- time-base utils ----------

    def update_timebase(self):
        if not self.filepath:
            self.tb_epoch = None; self.tb_source = None; self.tb_iso = None
            self.lbl_tb_src.setText("Źródło: —"); self.lbl_tb_iso.setText("Start: —")
            return
        epoch, src, iso = file_timebase(self.filepath)
        self.tb_epoch = epoch; self.tb_source = src; self.tb_iso = iso
        self.lbl_tb_src.setText(f"Źródło: {src or '—'}")
        self.lbl_tb_iso.setText(f"Start: {iso or '—'}")

    def abs_iso_from_offset(self, offset_s: float) -> str:
        """offset_s: sekundy od początku pliku → ISO wg time-base."""
        if self.tb_epoch is None: return ""
        ts = self.tb_epoch + float(offset_s)
        return datetime.fromtimestamp(ts).isoformat(timespec='milliseconds')

    # ---------- pomocnicze ----------

    def threshold_value(self, env: np.ndarray) -> float:
        mode = self.combo_thr_mode.currentText()
        if mode == "× mediana":
            med = np.median(env); fac = self.spin_thr_factor.value()
            return (med*fac) if med > 1e-12 else fac
        elif mode == "Stały":
            return float(self.spin_thr_abs.value())
        else:
            p = float(self.spin_thr_pct.value()); p = min(max(p, 50.0), 99.99)
            return float(np.percentile(env, p))

    def get_window_samples(self):
        if self.audio is None or self.fs is None:
            return np.array([], dtype=np.float64), 0, 0
        total_n = len(self.audio)
        n0 = int(round(self.window_start_sec * self.fs))
        n1 = int(round((self.window_start_sec + self.window_sec) * self.fs))
        n0 = max(0, n0); n1 = min(total_n, n1)
        return self.audio[n0:n1], n0, n1

    def set_table_headers_for_mode(self, mode: str):
        if mode == "Dekoder FSK":
            headers = ["t_start [s]","t_end [s]","długość [ms]","slot","expected","pol","BER",
                       "f0/f1 [Hz]","offset [ms]","BITY (24)","HEX (MSB / LSBflip)","margin"]
        else:
            headers = ["onset [s]","peak_env","prev_dt [s]","next_dt [s]","cluster_count_1s","valid_spike","period_ok"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    # ---------- handlery ----------

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Wybierz plik audio", "",
            "Audio files (*.wav *.flac *.ogg *.aiff *.aif *.aifc *.au);;All files (*)"
        )
        if not path: return
        try:
            x, fs = sf.read(path, always_2d=False)
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Nie udało się wczytać pliku:\n{e}")
            return
        self.audio = ensure_mono(np.asarray(x, dtype=np.float64))
        self.fs = float(fs); self.filepath = path
        if self.fs < 120_000.0:
            QMessageBox.warning(self, "Uwaga: próbkowanie",
                                f"Plik ma fs={self.fs:.1f} Hz. Do pasma 48–52 kHz wymagane jest ≥ 120 kHz.")
        self.window_start_sec = 0.0; self.spin_start.setValue(0.0)
        self.spin_len.setValue(min(self.window_sec, len(self.audio)/self.fs, 10.0))
        self.results_fsk.clear(); self.results_spikes.clear(); self.detections_fsk.clear()
        self.update_timebase()
        self.refresh_table(); self.update_plot()

    def on_mode_changed(self, idx: int):
        self.mode = self.modes[idx]; self.set_table_headers_for_mode(self.mode)
        self.refresh_table(); self.update_plot()

    def on_prev(self):
        if self.audio is None: return
        self.window_start_sec = max(0.0, self.window_start_sec - self.spin_len.value())
        self.spin_start.setValue(self.window_start_sec); self.update_plot()

    def on_next(self):
        if self.audio is None: return
        total = len(self.audio)/self.fs
        self.window_start_sec = min(max(0.0, total - self.spin_len.value()),
                                    self.window_start_sec + self.spin_len.value())
        self.spin_start.setValue(self.window_start_sec); self.update_plot()

    def on_start_changed(self, v: float):
        self.window_start_sec = float(v); self.update_plot()

    def on_len_changed(self, v: float):
        self.window_sec = float(v); self.update_plot()

    def on_analyze(self):
        if self.audio is None:
            QMessageBox.information(self, "Brak pliku", "Najpierw wczytaj plik audio."); return
        try:
            if self.mode == "Dekoder FSK":
                self.run_detection_and_decode_fsk()
            else:
                self.run_detection_spikes()
        except Exception as e:
            QMessageBox.critical(self, "Błąd analizy", str(e))

    def on_save_csv(self):
        if self.mode == "Dekoder FSK":
            if not self.results_fsk:
                QMessageBox.information(self, "Brak danych", "Nie ma wyników do zapisania."); return
            path, _ = QFileDialog.getSaveFileName(self, "Zapisz wyniki CSV (okno FSK)",
                                                  "wyniki_okno_fsk.csv", "CSV (*.csv)")
            if not path: return
            try:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f, delimiter=';')
                    w.writerow(["frame_start_s","frame_start_abs_iso","slot_idx2s","expected_label","polarity","errors","ber",
                                "t_start_s","t_end_s","abs_start_iso","abs_end_iso",
                                "duration_ms","f0_Hz","f1_Hz","offset_ms","bits_24","hex_msb","hex_lsbflip","margin",
                                "file_timebase_source","file_timebase_iso"])
                    for row in self.results_fsk:
                        abs_start = self.abs_iso_from_offset(row['t_start_s'])
                        abs_end   = self.abs_iso_from_offset(row['t_end_s'])
                        w.writerow([
                            f"{self.window_start_sec:.6f}",
                            self.abs_iso_from_offset(self.window_start_sec),
                            row.get('slot_idx2s', ""), row.get('expected_label', ""),
                            row.get('polarity', ""),
                            "" if row.get('errors') is None else int(row['errors']),
                            "" if row.get('ber') is None else f"{row['ber']:.6f}",
                            f"{row['t_start_s']:.6f}", f"{row['t_end_s']:.6f}",
                            abs_start, abs_end,
                            f"{row['duration_ms']:.3f}",
                            f"{row['f0']:.2f}", f"{row['f1']:.2f}",
                            "" if math.isnan(row['offset_ms']) else f"{row['offset_ms']:.3f}",
                            row['bits'], row['hex_msb'], row['hex_lsbflip'], f"{row['margin']:.3f}",
                            self.tb_source or "", self.tb_iso or ""
                        ])
                QMessageBox.information(self, "Zapisano", f"Wyniki zapisane do: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Błąd zapisu", str(e))
        else:
            if not self.results_spikes:
                QMessageBox.information(self, "Brak danych", "Nie ma wyników do zapisania."); return
            path, _ = QFileDialog.getSaveFileName(self, "Zapisz wyniki CSV (okno – impulsy)",
                                                  "spikes_window.csv", "CSV (*.csv)")
            if not path: return
            try:
                # szczegóły
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f, delimiter=';')
                    w.writerow(["frame_start_s","frame_start_abs_iso","onset_s","abs_time_iso","peak_env","prev_dt_s","next_dt_s",
                                "cluster_count_1s","valid_spike","period_ok",
                                "thr_mode","thr_value","refractory_ms","file_timebase_source","file_timebase_iso"])
                    thr_mode = self.combo_thr_mode.currentText()
                    samples, n0, n1 = self.get_window_samples()
                    xf = apply_bandpass(samples, self.fs, self.low_hz, self.high_hz) if samples.size>0 else samples
                    env = analytic_envelope(xf) if samples.size>0 else np.array([])
                    thr_val = self.threshold_value(env) if env.size>0 else float('nan')
                    refr = self.spin_refractory.value()
                    for r in self.results_spikes:
                        w.writerow([
                            f"{self.window_start_sec:.6f}",
                            self.abs_iso_from_offset(self.window_start_sec),
                            f"{r['onset_s']:.6f}",
                            self.abs_iso_from_offset(r['onset_s']),
                            f"{r['peak_env']:.6g}",
                            "" if np.isnan(r['prev_dt_s']) else f"{r['prev_dt_s']:.6f}",
                            "" if np.isnan(r['next_dt_s']) else f"{r['next_dt_s']:.6f}",
                            int(r['cluster_count_1s']), int(bool(r['valid_spike'])), int(bool(r['period_ok'])),
                            thr_mode, f"{thr_val:.6g}", f"{refr:.3f}",
                            self.tb_source or "", self.tb_iso or ""
                        ])
                # agregat
                path_agg = (path[:-4] + "_agg.csv") if path.lower().endswith(".csv") else (path + "_agg.csv")
                agg = self._aggregate_spikes(self.results_spikes)
                with open(path_agg, "w", newline="", encoding="utf-8") as f2:
                    w2 = csv.writer(f2, delimiter=';')
                    w2.writerow(["frame_start_s","frame_start_abs_iso","thr_mode","thr_value","refractory_ms",
                                 "spikes_total","spikes_valid","spikes_invalid","period_ok_share",
                                 "mean_prev_dt_s","median_prev_dt_s","std_prev_dt_s",
                                 "mean_next_dt_s","median_next_dt_s","std_next_dt_s",
                                 "mean_peak_env","median_peak_env",
                                 "file_timebase_source","file_timebase_iso"])
                    w2.writerow([
                        f"{self.window_start_sec:.6f}", self.abs_iso_from_offset(self.window_start_sec),
                        self.combo_thr_mode.currentText(), f"{thr_val:.6g}", f"{refr:.3f}",
                        agg['spikes_total'], agg['spikes_valid'], agg['spikes_invalid'],
                        self._fmt_float(agg['period_ok_share']),
                        self._fmt_float(agg['mean_prev_dt_s']), self._fmt_float(agg['median_prev_dt_s']), self._fmt_float(agg['std_prev_dt_s']),
                        self._fmt_float(agg['mean_next_dt_s']), self._fmt_float(agg['median_next_dt_s']), self._fmt_float(agg['std_next_dt_s']),
                        self._fmt_float(agg['mean_peak_env']), self._fmt_float(agg['median_peak_env']),
                        self.tb_source or "", self.tb_iso or ""
                    ])
                QMessageBox.information(self, "Zapisano",
                                        f"Wyniki zapisane do:\n• {path}\n• {path_agg}")
            except Exception as e:
                QMessageBox.critical(self, "Błąd zapisu", str(e))

    def on_analyze_full_file(self):
        if self.audio is None or self.fs is None:
            QMessageBox.information(self, "Brak pliku", "Najpierw wczytaj plik audio."); return
        if self.mode == "Dekoder FSK":
            self.fullscan_fsk()
        else:
            self.fullscan_spikes()

    # ---------- FSK ----------

    def find_segments(self, mask: np.ndarray, fs: float, extra_overlap_ms: float, merge_gap_ms: float):
        if mask.size == 0: return []
        diff = np.diff(mask.astype(np.int8), prepend=0)
        starts = np.flatnonzero(diff == 1); ends = np.flatnonzero(diff == -1) - 1
        if mask[0]: starts = np.r_[0, starts]
        if mask[-1]: ends = np.r_[ends, mask.size - 1]
        add = int(round((extra_overlap_ms/1000.0)*fs))
        starts = np.maximum(starts - add, 0); ends = np.minimum(ends + add, mask.size - 1)
        merged = []
        if len(starts) > 0:
            s0 = int(starts[0]); e0 = int(ends[0])
            max_gap = int(round((merge_gap_ms/1000.0)*fs))
            for s,e in zip(starts[1:], ends[1:]):
                if int(s) - e0 - 1 <= max_gap: e0 = max(e0, int(e))
                else: merged.append((s0,e0)); s0,e0 = int(s), int(e)
            merged.append((s0,e0))
        return merged

    def _analyze_window_core_fsk(self, n0: int, n1: int, extra_overlap_ms: float, merge_gap_ms: float):
        xw = self.audio[n0:n1]
        if xw.size == 0: return [], []
        try: xf = apply_bandpass(xw, self.fs, self.low_hz, self.high_hz)
        except ValueError: xf = xw
        env = analytic_envelope(xf)
        thr_val = self.threshold_value(env)
        mask = env >= thr_val
        segs = self.find_segments(mask, fs=self.fs, extra_overlap_ms=extra_overlap_ms, merge_gap_ms=merge_gap_ms)
        min_len_n = int(round((self.min_packet_ms/1000.0)*self.fs))
        segs = [(s,e) for (s,e) in segs if (e - s + 1) >= min_len_n]
        results = []
        for (s_idx, e_idx) in segs:
            seg = xf[s_idx:e_idx+1]
            d = decode_fsk_segment(seg, self.fs, 49_600.0, 50_600.0,
                                   autocal=self.chk_autocal.isChecked(),
                                   bit_ms=self.bit_ms, gap_ms=self.gap_ms, n_bits=self.n_bits,
                                   search_step_ms=0.25)
            t_start = (n0 + s_idx)/self.fs; t_end = (n0 + e_idx)/self.fs
            duration_ms = (t_end - t_start)*1000.0
            if d.get('ok', False):
                results.append({
                    't_start_s': t_start, 't_end_s': t_end, 'duration_ms': duration_ms,
                    'f0': d['f0'], 'f1': d['f1'], 'offset_ms': d['start_offset_ms'],
                    'bits': d['bits'], 'hex_msb': d['hex_msb'], 'hex_lsbflip': d['hex_lsbflip'],
                    'margin': d['margin']
                })
            else:
                results.append({
                    't_start_s': t_start, 't_end_s': t_end, 'duration_ms': duration_ms,
                    'f0': d.get('f0', 49_600.0), 'f1': d.get('f1', 50_600.0),
                    'offset_ms': float('nan'), 'bits': "(nie zdekodowano)",
                    'hex_msb': "", 'hex_lsbflip': "", 'margin': 0.0
                })
        return results, segs

    def _label_results_with_expected(self, results: list[dict], frame_start_s: float):
        for r in results:
            mid = (r['t_start_s'] + r['t_end_s'])/2.0
            rel = mid - frame_start_s
            r['slot_idx2s'] = None; r['expected_label'] = ""; r['polarity'] = ""; r['errors'] = None; r['ber'] = None
            if 0.0 <= rel < 8.0:
                slot_idx = int(rel // 2.0) + 1
                r['slot_idx2s'] = slot_idx
                label = SLOT_TO_LABEL.get(slot_idx, "")
                r['expected_label'] = label
                exp_bits = EXPECTED_PATTERNS.get(label, "")
                if exp_bits and len(r.get('bits', "")) == len(exp_bits) and not r['bits'].startswith("("):
                    comp = compare_to_expected(r['bits'], exp_bits)
                    r['polarity'] = comp['polarity'] or ""; r['errors'] = comp['errors']; r['ber'] = comp['ber']
        return results

    def run_detection_and_decode_fsk(self):
        samples, n0, n1 = self.get_window_samples()
        if samples.size == 0: return
        results, segs = self._analyze_window_core_fsk(n0, n1, self.spin_overlap.value(), self.spin_merge.value())
        results = self._label_results_with_expected(results, self.window_start_sec)
        self.detections_fsk = segs; self.results_fsk = results
        self.refresh_table(); self.update_plot()

    def fullscan_fsk(self):
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz wyniki CSV (cały plik – FSK)",
                                              "wyniki_fullscan.csv", "CSV (*.csv)")
        if not path: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            total = len(self.audio); frame_len = int(round(10.0*self.fs))
            start = 0; frame_idx = 0; saved = 0
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter=';')
                w.writerow(["frame_idx10s","frame_start_s","frame_start_abs_iso",
                            "slot_idx2s","expected_label","polarity","errors","ber",
                            "t_start_s","t_end_s","abs_start_iso","abs_end_iso","duration_ms",
                            "f0_Hz","f1_Hz","offset_ms","bits_24","hex_msb","hex_lsbflip","margin",
                            "file_timebase_source","file_timebase_iso"])
                while start < total:
                    end = min(total, start + frame_len)
                    results, _ = self._analyze_window_core_fsk(start, end, self.spin_overlap.value(), self.spin_merge.value())
                    frame_start_s = start / self.fs
                    results = self._label_results_with_expected(results, frame_start_s)
                    frame_start_iso = self.abs_iso_from_offset(frame_start_s)
                    for r in results:
                        w.writerow([
                            frame_idx, f"{frame_start_s:.6f}", frame_start_iso,
                            "" if r.get('slot_idx2s') is None else r['slot_idx2s'],
                            r.get('expected_label', ""), r.get('polarity', ""),
                            "" if r.get('errors') is None else int(r['errors']),
                            "" if r.get('ber') is None else f"{r['ber']:.6f}",
                            f"{r['t_start_s']:.6f}", f"{r['t_end_s']:.6f}",
                            self.abs_iso_from_offset(r['t_start_s']),
                            self.abs_iso_from_offset(r['t_end_s']),
                            f"{r['duration_ms']:.3f}",
                            f"{r['f0']:.2f}", f"{r['f1']:.2f}",
                            "" if math.isnan(r['offset_ms']) else f"{r['offset_ms']:.3f}",
                            r['bits'], r['hex_msb'], r['hex_lsbflip'], f"{r['margin']:.3f}",
                            self.tb_source or "", self.tb_iso or ""
                        ])
                        saved += 1
                    start += frame_len; frame_idx += 1
            QMessageBox.information(self, "Zapisano", f"FSK: zapisano {saved} wierszy do {path}")
        except Exception as e:
            QMessageBox.critical(self, "Błąd analizy FSK", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    # ---------- Impulsy ----------

    def _nanstats(self, arr_like):
        a = np.asarray(arr_like, dtype=float)
        if a.size == 0: return (np.nan, np.nan, np.nan)
        return (np.nanmean(a), np.nanmedian(a), np.nanstd(a, ddof=0))

    def _fmt_float(self, x, nd=6):
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))): return ""
        return f"{x:.{nd}f}"

    def _aggregate_spikes(self, metrics: list[dict]) -> dict:
        if not metrics:
            return {'spikes_total': 0, 'spikes_valid': 0, 'spikes_invalid': 0,
                    'period_ok_share': np.nan,
                    'mean_prev_dt_s': np.nan, 'median_prev_dt_s': np.nan, 'std_prev_dt_s': np.nan,
                    'mean_next_dt_s': np.nan, 'median_next_dt_s': np.nan, 'std_next_dt_s': np.nan,
                    'mean_peak_env': np.nan, 'median_peak_env': np.nan}
        total = len(metrics); valid = sum(1 for m in metrics if bool(m['valid_spike']))
        invalid = total - valid
        period_ok_share = (sum(1 for m in metrics if bool(m['period_ok'])) / total) if total else np.nan
        prev_list = [m['prev_dt_s'] for m in metrics]; next_list = [m['next_dt_s'] for m in metrics]
        peak_list = [m['peak_env'] for m in metrics]
        mn, md, sd = self._nanstats(prev_list); mn2, md2, sd2 = self._nanstats(next_list)
        mnp, mdp, _ = self._nanstats(peak_list)
        return {'spikes_total': total, 'spikes_valid': valid, 'spikes_invalid': invalid,
                'period_ok_share': period_ok_share,
                'mean_prev_dt_s': mn, 'median_prev_dt_s': md, 'std_prev_dt_s': sd,
                'mean_next_dt_s': mn2, 'median_next_dt_s': md2, 'std_next_dt_s': sd2,
                'mean_peak_env': mnp, 'median_peak_env': mdp}

    def _analyze_window_core_spikes(self, n0: int, n1: int):
        xw = self.audio[n0:n1]
        if xw.size == 0: return [], np.array([]), np.array([])
        try: xf = apply_bandpass(xw, self.fs, self.low_hz, self.high_hz)
        except ValueError: xf = xw
        env = analytic_envelope(xf)
        thr_val = self.threshold_value(env)
        refr = self.spin_refractory.value()
        onsets = detect_spikes_onsets(env, self.fs, thr_val, refractory_ms=refr)
        metrics = spikes_metrics(onsets, env, self.fs, peak_win_ms=self.peak_win_ms,
                                 period_s=self.period_s, period_tol=self.period_tol)
        # look‑ahead
        if len(metrics) > 0:
            last_onset_abs = (n0/self.fs) + metrics[-1]['onset_s']
            need_s = self.period_s*(1.0+self.period_tol) + 0.1
            n2 = min(len(self.audio), n1 + int(round(need_s*self.fs)))
            if n2 > n1:
                tail = self.audio[n1:n2]
                try: xt = apply_bandpass(tail, self.fs, self.low_hz, self.high_hz)
                except ValueError: xt = tail
                env_t = analytic_envelope(xt)
                on_tail = detect_spikes_onsets(env_t, self.fs, thr_val, refractory_ms=refr)
                on_tail_abs = (n1 + on_tail) / self.fs
                metrics = patch_last_spike_crossframe(metrics, last_onset_abs, on_tail_abs,
                                                      self.period_s, self.period_tol)
        # onset na absolutne (sekundy od startu pliku)
        for m in metrics:
            m['onset_s'] = (n0/self.fs) + m['onset_s']
        return metrics, env, onsets

    def run_detection_spikes(self):
        samples, n0, n1 = self.get_window_samples()
        if samples.size == 0: return
        metrics, env, onsets = self._analyze_window_core_spikes(n0, n1)
        self.results_spikes = metrics
        self.refresh_table(); self.update_plot()

    def fullscan_spikes(self):
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz wyniki CSV (cały plik – impulsy)",
                                              "spikes_fullscan.csv", "CSV (*.csv)")
        if not path: return
        path_agg = (path[:-4] + "_agg.csv") if path.lower().endswith(".csv") else (path + "_agg.csv")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            total = len(self.audio); frame_len = int(round(10.0*self.fs))
            start = 0; frame_idx = 0; saved = 0

            prev_metrics = None; prev_frame_idx = None; prev_frame_start_s = None
            prev_thr_mode = None; prev_thr_val = None; prev_refr = None

            with open(path, "w", newline="", encoding="utf-8") as f, \
                 open(path_agg, "w", newline="", encoding="utf-8") as fagg:
                w = csv.writer(f, delimiter=';'); wagg = csv.writer(fagg, delimiter=';')
                w.writerow(["frame_idx10s","frame_start_s","frame_start_abs_iso",
                            "spike_idx","onset_s","abs_time_iso","peak_env","prev_dt_s","next_dt_s",
                            "cluster_count_1s","valid_spike","period_ok",
                            "thr_mode","thr_value","refractory_ms","file_timebase_source","file_timebase_iso"])
                wagg.writerow(["frame_idx10s","frame_start_s","frame_start_abs_iso","thr_mode","thr_value","refractory_ms",
                               "spikes_total","spikes_valid","spikes_invalid","period_ok_share",
                               "mean_prev_dt_s","median_prev_dt_s","std_prev_dt_s",
                               "mean_next_dt_s","median_next_dt_s","std_next_dt_s",
                               "mean_peak_env","median_peak_env",
                               "file_timebase_source","file_timebase_iso"])
                while start < total:
                    end = min(total, start + frame_len)
                    xw = self.audio[start:end]
                    try: xf = apply_bandpass(xw, self.fs, self.low_hz, self.high_hz)
                    except ValueError: xf = xw
                    env = analytic_envelope(xf)
                    thr_mode = self.combo_thr_mode.currentText()
                    thr_val = self.threshold_value(env)
                    refr = self.spin_refractory.value()
                    on_local = detect_spikes_onsets(env, self.fs, thr_val, refractory_ms=refr)
                    mets_local = spikes_metrics(on_local, env, self.fs, peak_win_ms=self.peak_win_ms,
                                                period_s=self.period_s, period_tol=self.period_tol)
                    frame_start_s = start / self.fs
                    # dopnij poprzednią ramkę
                    if prev_metrics is not None and len(prev_metrics) > 0:
                        last_abs_prev = prev_metrics[-1]['onset_s']
                        on_abs_cur = frame_start_s + (on_local / self.fs)
                        prev_metrics = patch_last_spike_crossframe(prev_metrics, last_abs_prev, on_abs_cur,
                                                                   self.period_s, self.period_tol)
                        for i, m in enumerate(prev_metrics):
                            w.writerow([
                                prev_frame_idx, f"{prev_frame_start_s:.6f}",
                                self.abs_iso_from_offset(prev_frame_start_s),
                                i, f"{m['onset_s']:.6f}", self.abs_iso_from_offset(m['onset_s']),
                                f"{m['peak_env']:.6g}",
                                "" if np.isnan(m['prev_dt_s']) else f"{m['prev_dt_s']:.6f}",
                                "" if np.isnan(m['next_dt_s']) else f"{m['next_dt_s']:.6f}",
                                int(m['cluster_count_1s']), int(bool(m['valid_spike'])), int(bool(m['period_ok'])),
                                prev_thr_mode, f"{prev_thr_val:.6g}", f"{prev_refr:.3f}",
                                self.tb_source or "", self.tb_iso or ""
                            ])
                            saved += 1
                        agg_prev = self._aggregate_spikes(prev_metrics)
                        wagg.writerow([
                            prev_frame_idx, f"{prev_frame_start_s:.6f}",
                            self.abs_iso_from_offset(prev_frame_start_s),
                            prev_thr_mode, f"{prev_thr_val:.6g}", f"{prev_refr:.3f}",
                            agg_prev['spikes_total'], agg_prev['spikes_valid'], agg_prev['spikes_invalid'],
                            self._fmt_float(agg_prev['period_ok_share']),
                            self._fmt_float(agg_prev['mean_prev_dt_s']), self._fmt_float(agg_prev['median_prev_dt_s']), self._fmt_float(agg_prev['std_prev_dt_s']),
                            self._fmt_float(agg_prev['mean_next_dt_s']), self._fmt_float(agg_prev['median_next_dt_s']), self._fmt_float(agg_prev['std_next_dt_s']),
                            self._fmt_float(agg_prev['mean_peak_env']), self._fmt_float(agg_prev['median_peak_env']),
                            self.tb_source or "", self.tb_iso or ""
                        ])
                        prev_metrics = None

                    # przygotuj bieżącą
                    for m in mets_local:
                        m['onset_s'] = frame_start_s + m['onset_s']
                    prev_metrics = mets_local
                    prev_frame_idx = frame_idx
                    prev_frame_start_s = frame_start_s
                    prev_thr_mode = thr_mode
                    prev_thr_val = thr_val
                    prev_refr = refr

                    start += frame_len; frame_idx += 1

                # ostatnia ramka
                if prev_metrics is not None:
                    for i, m in enumerate(prev_metrics):
                        w.writerow([
                            prev_frame_idx, f"{prev_frame_start_s:.6f}",
                            self.abs_iso_from_offset(prev_frame_start_s),
                            i, f"{m['onset_s']:.6f}", self.abs_iso_from_offset(m['onset_s']),
                            f"{m['peak_env']:.6g}",
                            "" if np.isnan(m['prev_dt_s']) else f"{m['prev_dt_s']:.6f}",
                            "" if np.isnan(m['next_dt_s']) else f"{m['next_dt_s']:.6f}",
                            int(m['cluster_count_1s']), int(bool(m['valid_spike'])), int(bool(m['period_ok'])),
                            prev_thr_mode, f"{prev_thr_val:.6g}", f"{prev_refr:.3f}",
                            self.tb_source or "", self.tb_iso or ""
                        ])
                        saved += 1
                    agg_prev = self._aggregate_spikes(prev_metrics)
                    wagg.writerow([
                        prev_frame_idx, f"{prev_frame_start_s:.6f}",
                        self.abs_iso_from_offset(prev_frame_start_s),
                        prev_thr_mode, f"{prev_thr_val:.6g}", f"{prev_refr:.3f}",
                        agg_prev['spikes_total'], agg_prev['spikes_valid'], agg_prev['spikes_invalid'],
                        self._fmt_float(agg_prev['period_ok_share']),
                        self._fmt_float(agg_prev['mean_prev_dt_s']), self._fmt_float(agg_prev['median_prev_dt_s']), self._fmt_float(agg_prev['std_prev_dt_s']),
                        self._fmt_float(agg_prev['mean_next_dt_s']), self._fmt_float(agg_prev['median_next_dt_s']), self._fmt_float(agg_prev['std_next_dt_s']),
                        self._fmt_float(agg_prev['mean_peak_env']), self._fmt_float(agg_prev['median_peak_env']),
                        self.tb_source or "", self.tb_iso or ""
                    ])

            QMessageBox.information(self, "Zapisano",
                                    f"Impulsy: zapisano {saved} rekordów do:\n• {path}\n• {path_agg}")
        except Exception as e:
            QMessageBox.critical(self, "Błąd analizy impulsów", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    # ---------- Tabela i rysowanie ----------

    def refresh_table(self):
        mode = self.mode
        self.table.setRowCount(0)
        if mode == "Dekoder FSK":
            for row in self.results_fsk:
                r = self.table.rowCount(); self.table.insertRow(r)
                self.table.setItem(r, 0, QTableWidgetItem(f"{row['t_start_s']:.6f}"))
                self.table.setItem(r, 1, QTableWidgetItem(f"{row['t_end_s']:.6f}"))
                self.table.setItem(r, 2, QTableWidgetItem(f"{row['duration_ms']:.3f}"))
                self.table.setItem(r, 3, QTableWidgetItem("" if row.get('slot_idx2s') is None else str(row['slot_idx2s'])))
                self.table.setItem(r, 4, QTableWidgetItem(row.get('expected_label',"")))
                self.table.setItem(r, 5, QTableWidgetItem(row.get('polarity',"")))
                ber = row.get('ber', None)
                self.table.setItem(r, 6, QTableWidgetItem("" if ber is None else f"{ber:.4f}"))
                self.table.setItem(r, 7, QTableWidgetItem(f"{row['f0']:.1f} / {row['f1']:.1f}"))
                self.table.setItem(r, 8, QTableWidgetItem("" if math.isnan(row['offset_ms']) else f"{row['offset_ms']:.3f}"))
                self.table.setItem(r, 9, QTableWidgetItem(row['bits']))
                self.table.setItem(r,10, QTableWidgetItem(f"{row['hex_msb'] or '—'} / {row['hex_lsbflip'] or '—'}"))
                self.table.setItem(r,11, QTableWidgetItem(f"{row['margin']:.3f}"))
        else:
            for row in self.results_spikes:
                r = self.table.rowCount(); self.table.insertRow(r)
                self.table.setItem(r, 0, QTableWidgetItem(f"{row['onset_s']:.6f}"))
                self.table.setItem(r, 1, QTableWidgetItem(f"{row['peak_env']:.6g}"))
                self.table.setItem(r, 2, QTableWidgetItem("" if np.isnan(row['prev_dt_s']) else f"{row['prev_dt_s']:.6f}"))
                self.table.setItem(r, 3, QTableWidgetItem("" if np.isnan(row['next_dt_s']) else f"{row['next_dt_s']:.6f}"))
                self.table.setItem(r, 4, QTableWidgetItem(str(int(row['cluster_count_1s']))))
                self.table.setItem(r, 5, QTableWidgetItem("1" if row['valid_spike'] else "0"))
                self.table.setItem(r, 6, QTableWidgetItem("1" if row['period_ok'] else "0"))

    def update_plot(self):
        axS = self.canvas.ax_spec; axT = self.canvas.ax_time
        axS.clear(); axT.clear()

        samples, n0, n1 = self.get_window_samples()
        if samples.size == 0:
            axS.set_title("Brak danych — wczytaj plik audio")
            axT.set_title("Sygnał po filtrze i obwiednia (podgląd)")
            self.canvas.draw_idle(); return

        try: xf = apply_bandpass(samples, self.fs, self.low_hz, self.high_hz)
        except Exception: xf = samples

        env = analytic_envelope(xf)
        thr_val = self.threshold_value(env)

        # Spektrogram
        nper = 1024; nover = nper//2
        f, t, Sxx = signal.spectrogram(xf, fs=self.fs, nperseg=nper, noverlap=nover,
                                       detrend=False, scaling='spectrum', mode='magnitude')
        Sxx_db = db_from_lin(Sxx, floor=1e-16)
        band = (f >= 40_000.0) & (f <= 60_000.0)
        if np.any(band):
            f_plot = f[band]; S_plot = Sxx_db[band,:]; y_units='kHz'; y_scale=1/1000.0
        else:
            f_plot = f; S_plot = Sxx_db; y_units='Hz'; y_scale=1.0
        t_abs = self.window_start_sec + t
        extent = [ t_abs.min() if t_abs.size else self.window_start_sec,
                   t_abs.max() if t_abs.size else (self.window_start_sec + self.window_sec),
                   (f_plot.min()*y_scale) if f_plot.size else 0.0,
                   (f_plot.max()*y_scale) if f_plot.size else 0.0 ]
        if S_plot.size>0: axS.imshow(S_plot[::-1,:], extent=extent, aspect='auto', origin='upper')
        axS.set_ylabel(f"Częstotliwość [{y_units}]"); axS.set_xlabel("Czas [s]")
        axS.set_title(f"Spektrogram (nperseg=1024), okno {self.window_sec:.1f} s")

        if self.mode == "Dekoder FSK":
            y0, y1 = axS.get_ylim(); yb, yh = (min(y0,y1), max(y0,y1)); full_h = (yh-yb) if yh>yb else 1.0
            bit_s = self.bit_ms/1000.0; gap_s = self.gap_ms/1000.0; sym_s = bit_s + gap_s
            color0=(0.2,0.4,0.9,0.18); color1=(0.95,0.55,0.2,0.18); edge0=(0.2,0.4,0.9,0.9); edge1=(0.95,0.55,0.2,0.9)
            win_start=self.window_start_sec; win_end=self.window_start_sec+self.window_sec
            for row in self.results_fsk:
                bits=row.get('bits',""); if not bits or bits.startswith("("): continue
                offset=(row.get('offset_ms',0.0) or 0.0)/1000.0; t0=row['t_start_s']+offset
                for k,b in enumerate(bits):
                    tb=t0+k*sym_s; te=tb+bit_s
                    if te<win_start or tb>win_end: continue
                    x=max(tb,win_start); w=max(0.0,min(te,win_end)-x)
                    if w<=0: continue
                    is1=(b=='1'); fc=color1 if is1 else color0; ec=edge1 if is1 else edge0
                    rect=Rectangle((x,yb),w,full_h,facecolor=fc,edgecolor=ec,linewidth=0.8)
                    axS.add_patch(rect)
                    axS.text(x+w/2.0,yh,b,ha='center',va='bottom',fontsize=8,alpha=0.9,clip_on=True)
        else:
            refr = self.spin_refractory.value()
            on_local = detect_spikes_onsets(env, self.fs, thr_val, refractory_ms=refr)
            y0, y1 = axS.get_ylim(); yb, yh = (min(y0,y1), max(y0,y1)); full_h=(yh-yb) if yh>yb else 1.0
            cluster_horizon = self.period_s*(1.0+self.period_tol)
            for o in on_local:
                t0 = self.window_start_sec + (o/self.fs)
                cnt = int(np.sum(((on_local/self.fs + self.window_start_sec) >= t0) &
                                 ((on_local/self.fs + self.window_start_sec) <  t0 + cluster_horizon)))
                if cnt > 1:
                    rect = Rectangle((t0,yb), cluster_horizon, full_h, facecolor=(1.0,0.2,0.2,0.15),
                                     edgecolor=(0.8,0.0,0.0,0.7), linewidth=0.8)
                    axS.add_patch(rect)
                axS.axvline(t0, linestyle='-', linewidth=1.0, alpha=0.9)

        # Czasowy: sygnał + obwiednia + próg
        max_pts=5000; step=max(1,int(np.ceil(xf.size/max_pts)))
        t_sig=self.window_start_sec + (np.arange(0,xf.size,step)/self.fs)
        axT.plot(t_sig, xf[::step], linewidth=0.6, alpha=0.6, label="Sygnał (po filtrze)")
        axT.plot(t_sig, env[::step], linewidth=1.0, alpha=0.9, label="Obwiednia")
        axT.axhline(thr_val, linestyle='--', linewidth=1.0, label=f"Próg ({self.combo_thr_mode.currentText()})")

        if self.mode == "Dekoder FSK":
            mask = env >= thr_val
            segs_prev = self.find_segments(mask, fs=self.fs, extra_overlap_ms=self.spin_overlap.value(), merge_gap_ms=self.spin_merge.value())
            min_len_n = int(round((self.min_packet_ms/1000.0)*self.fs))
            segs_prev = [(s,e) for (s,e) in segs_prev if (e - s + 1) >= min_len_n]
            y0t,y1t=axT.get_ylim(); ybt,yht=(min(y0t,y1t), max(y0t,y1t)); htime=(yht-ybt) if yht>ybt else 1.0
            for (s,e) in segs_prev:
                t_s=self.window_start_sec + (s/self.fs); t_e=self.window_start_sec + (e/self.fs)
                rect=Rectangle((t_s,ybt),(t_e-t_s),htime,facecolor=(0.1,0.8,0.1,0.15),
                               edgecolor=(0.1,0.6,0.1,0.8),linewidth=0.8)
                axT.add_patch(rect)
        else:
            refr = self.spin_refractory.value()
            on_local = detect_spikes_onsets(env, self.fs, thr_val, refractory_ms=refr)
            y0t,y1t=axT.get_ylim(); ybt,yht=(min(y0t,y1t), max(y0t,y1t)); htime=(yht-ybt) if yht>ybt else 1.0
            cluster_horizon = self.period_s*(1.0+self.period_tol)
            for o in on_local:
                t0=self.window_start_sec + (o/self.fs)
                cnt=int(np.sum(((on_local/self.fs + self.window_start_sec) >= t0) &
                               ((on_local/self.fs + self.window_start_sec) <  t0 + cluster_horizon)))
                if cnt > 1:
                    rect=Rectangle((t0,ybt), cluster_horizon, htime, facecolor=(1.0,0.2,0.2,0.15),
                                   edgecolor=(0.8,0.0,0.0,0.7), linewidth=0.8)
                    axT.add_patch(rect)
                axT.axvline(t0, linestyle='-', linewidth=1.0, alpha=0.9)

        axT.set_xlabel("Czas [s]"); axT.set_ylabel("Amplituda")
        axT.set_title("Sygnał po filtrze (zdecy­mowany), obwiednia i próg")
        axT.legend(loc='upper right', fontsize=8)
        self.canvas.draw_idle()

# -----------------------

def main():
    app = QApplication(sys.argv)
    w = DecoderApp(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
