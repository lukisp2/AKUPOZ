

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, hilbert, find_peaks, get_window
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

APP_TITLE = "Burst Analyzer GUI"

def read_wav(path):
    sr, x = wavfile.read(path)
    if x.dtype.kind in ("i", "u"):
        max_val = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / max_val
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    return sr, x

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    if not (0 < lowcut < highcut < nyq):
        raise ValueError(f"Niepoprawne pasmo filtru: low={lowcut}, high={highcut}, fs={fs}")
    low = lowcut / nyq
    high = highcut / nyq
    from scipy.signal import butter
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(x, fs, lowcut, highcut, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, x)
    return y

def envelope_hilbert(x):
    analytic = hilbert(x)
    env = np.abs(analytic)
    return env

def find_bursts(env, fs, thresh_factor, min_sep_ms, min_width_ms):
    thr = np.median(env) * thresh_factor
    min_distance = int((min_sep_ms / 1000.0) * fs)
    width = int((min_width_ms / 1000.0) * fs)
    if min_distance < 1:
        min_distance = 1
    if width < 1:
        width = 1
    peaks, props = find_peaks(env, height=thr, distance=min_distance, width=width)
    return peaks, props, thr

def segment_burst(env, x, fs, peak_idx, pad_ms):
    pad = int((pad_ms/1000.0)*fs)
    N = len(env)
    local_thr = 0.5 * env[peak_idx]
    left = peak_idx
    while left > 0 and env[left] > local_thr:
        left -= 1
    right = peak_idx
    while right < N-1 and env[right] > local_thr:
        right += 1
    left = max(0, left - pad)
    right = min(N, right + pad)
    return left, right

def dominant_freq(segment, fs):
    seg = segment - np.mean(segment)
    N = len(seg)
    if N < 8:
        return np.nan
    win = get_window("hann", N, fftbins=True)
    segw = seg * win
    nfft = int(2**np.ceil(np.log2(N)))
    spec = np.fft.rfft(segw, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
    mag = np.abs(spec)
    if len(mag) > 1:
        mag[0] = 0.0
    peak_bin = int(np.argmax(mag))
    f0 = float(freqs[peak_bin])
    return f0

def decimate_for_plot(y, max_points=200000):
    N = len(y)
    if N <= max_points:
        return y
    step = int(np.ceil(N / max_points))
    return y[::step]

class BurstAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1400x900")
        self.minsize(1200, 800)

        # Data
        self.fs = None
        self.x = None
        self.xf = None
        self.env = None
        self.peaks = np.array([], dtype=int)
        self.props = {}
        self.thr = None
        self.peak_times = np.array([])
        self.ibi = np.array([])
        self.rep_freqs = np.array([])
        self.burst_bounds = []
        self.intraburst_freqs = np.array([])
        self.summary = {}
        self.per_burst_df = pd.DataFrame()

        self._build_ui()

    def _build_ui(self):
        # ---- Top controls ----
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        # File chooser
        self.wav_path_var = tk.StringVar()
        ttk.Label(top, text="Plik WAV:").grid(row=0, column=0, sticky="w")
        self.wav_entry = ttk.Entry(top, textvariable=self.wav_path_var, width=70)
        self.wav_entry.grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(top, text="Wybierz...", command=self.choose_file).grid(row=0, column=2, padx=4)

        # Parameters
        param_frame = ttk.LabelFrame(top, text="Parametry")
        param_frame.grid(row=1, column=0, columnspan=3, sticky="we", pady=6)

        self.bp_low_var = tk.DoubleVar(value=300.0)
        self.bp_high_var = tk.DoubleVar(value=3000.0)
        self.bp_order_var = tk.IntVar(value=4)
        self.thresh_factor_var = tk.DoubleVar(value=5.0)
        self.min_sep_ms_var = tk.DoubleVar(value=10.0)
        self.min_width_ms_var = tk.DoubleVar(value=2.0)
        self.burst_pad_ms_var = tk.DoubleVar(value=3.0)

        row = 0
        ttk.Label(param_frame, text="BP low [Hz]").grid(row=row, column=0, padx=4, pady=2, sticky="w")
        ttk.Entry(param_frame, textvariable=self.bp_low_var, width=10).grid(row=row, column=1, padx=4)
        ttk.Label(param_frame, text="BP high [Hz]").grid(row=row, column=2, padx=4, pady=2, sticky="w")
        ttk.Entry(param_frame, textvariable=self.bp_high_var, width=10).grid(row=row, column=3, padx=4)
        ttk.Label(param_frame, text="Order").grid(row=row, column=4, padx=4, pady=2, sticky="w")
        ttk.Entry(param_frame, textvariable=self.bp_order_var, width=6).grid(row=row, column=5, padx=4)

        row += 1
        ttk.Label(param_frame, text="Thresh × median(env)").grid(row=row, column=0, padx=4, sticky="w")
        ttk.Entry(param_frame, textvariable=self.thresh_factor_var, width=10).grid(row=row, column=1, padx=4)
        ttk.Label(param_frame, text="Min odstęp [ms]").grid(row=row, column=2, padx=4, sticky="w")
        ttk.Entry(param_frame, textvariable=self.min_sep_ms_var, width=10).grid(row=row, column=3, padx=4)
        ttk.Label(param_frame, text="Min szer. [ms]").grid(row=row, column=4, padx=4, sticky="w")
        ttk.Entry(param_frame, textvariable=self.min_width_ms_var, width=10).grid(row=row, column=5, padx=4)

        row += 1
        ttk.Label(param_frame, text="Pad FFT [ms]").grid(row=row, column=0, padx=4, sticky="w")
        ttk.Entry(param_frame, textvariable=self.burst_pad_ms_var, width=10).grid(row=row, column=1, padx=4)

        # Action buttons
        btn_frame = ttk.Frame(top)
        btn_frame.grid(row=2, column=0, columnspan=3, sticky="we", pady=6)
        ttk.Button(btn_frame, text="Analizuj", command=self.analyze).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Zapisz wyniki CSV", command=self.save_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Zapisz wykresy PNG", command=self.save_pngs).pack(side=tk.LEFT, padx=4)

        # ---- Middle: Stats + List ----
        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        stats = ttk.LabelFrame(mid, text="Podsumowanie")
        stats.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self.stats_vars = {
            "fs": tk.StringVar(value="-"),
            "num_bursts": tk.StringVar(value="-"),
            "mean_ibi_ms": tk.StringVar(value="-"),
            "median_ibi_ms": tk.StringVar(value="-"),
            "mean_rep_hz": tk.StringVar(value="-"),
            "median_rep_hz": tk.StringVar(value="-"),
            "median_intra_hz": tk.StringVar(value="-"),
        }

        def add_stat(row, label, var):
            ttk.Label(stats, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=1)
            ttk.Label(stats, textvariable=var).grid(row=row, column=1, sticky="w", padx=4, pady=1)

        add_stat(0, "fs [Hz]:", self.stats_vars["fs"])
        add_stat(1, "Liczba burstów:", self.stats_vars["num_bursts"])
        add_stat(2, "Średni IBI [ms]:", self.stats_vars["mean_ibi_ms"])
        add_stat(3, "Mediana IBI [ms]:", self.stats_vars["median_ibi_ms"])
        add_stat(4, "Średnia rep. [Hz]:", self.stats_vars["mean_rep_hz"])
        add_stat(5, "Mediana rep. [Hz]:", self.stats_vars["median_rep_hz"])
        add_stat(6, "Mediana f0 (intra) [Hz]:", self.stats_vars["median_intra_hz"])

        # List of bursts
        burst_list_frame = ttk.LabelFrame(mid, text="Bursty")
        burst_list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        columns = ("idx", "time_s", "f0_hz")
        self.tree = ttk.Treeview(burst_list_frame, columns=columns, show="headings", height=8)
        self.tree.heading("idx", text="Idx")
        self.tree.heading("time_s", text="Czas [s]")
        self.tree.heading("f0_hz", text="f0 [Hz]")
        self.tree.column("idx", width=60, anchor="center")
        self.tree.column("time_s", width=120, anchor="center")
        self.tree.column("f0_hz", width=120, anchor="center")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_burst)

        vsb = ttk.Scrollbar(burst_list_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill="y")
        self.tree.configure(yscrollcommand=vsb.set)

        # ---- Bottom: Plots (3 separate figures, no subplots) ----
        plot_area = ttk.Frame(self)
        plot_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Figure 1: waveform + envelope + peaks
        self.fig1 = Figure(figsize=(5, 3), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=plot_area)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar1 = NavigationToolbar2Tk(self.canvas1, plot_area, pack_toolbar=False)
        self.toolbar1.update()
        self.toolbar1.pack(side=tk.TOP, fill=tk.X)

        # Figure 2: IBI histogram
        self.fig2 = Figure(figsize=(5, 3), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=plot_area)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar2 = NavigationToolbar2Tk(self.canvas2, plot_area, pack_toolbar=False)
        self.toolbar2.update()
        self.toolbar2.pack(side=tk.TOP, fill=tk.X)

        # Figure 3: FFT of selected burst
        self.fig3 = Figure(figsize=(5, 3), dpi=100)
        self.ax3 = self.fig3.add_subplot(111)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=plot_area)
        self.canvas3.draw()
        self.canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar3 = NavigationToolbar2Tk(self.canvas3, plot_area, pack_toolbar=False)
        self.toolbar3.update()
        self.toolbar3.pack(side=tk.TOP, fill=tk.X)

        # Initial titles
        self.ax1.set_title("Sygnał (po filtracji) + obwiednia + piki")
        self.ax1.set_xlabel("Czas [s]")
        self.ax1.set_ylabel("Amplituda / Obwiednia")
        self.ax2.set_title("Histogram odstępów między burstami (IBI)")
        self.ax2.set_xlabel("IBI [ms]")
        self.ax2.set_ylabel("Liczność")
        self.ax3.set_title("FFT wybranego bursta")
        self.ax3.set_xlabel("Częstotliwość [Hz]")
        self.ax3.set_ylabel("Magnituda")

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Wybierz plik WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if path:
            self.wav_path_var.set(path)

    def analyze(self):
        path = self.wav_path_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror(APP_TITLE, "Wskaż istniejący plik WAV.")
            return
        try:
            fs, x = read_wav(path)
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Błąd odczytu WAV: {e}")
            return

        # Params
        try:
            low = float(self.bp_low_var.get())
            high = float(self.bp_high_var.get())
            order = int(self.bp_order_var.get())
            thr_fac = float(self.thresh_factor_var.get())
            min_sep = float(self.min_sep_ms_var.get())
            min_width = float(self.min_width_ms_var.get())
            pad_ms = float(self.burst_pad_ms_var.get())
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Błąd parametrów: {e}")
            return

        try:
            xf = apply_bandpass(x, fs, low, high, order)
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Błąd filtru: {e}")
            return

        env = envelope_hilbert(xf)
        peaks, props, thr = find_bursts(env, fs, thr_fac, min_sep, min_width)

        peak_times = peaks / fs
        ibi = np.diff(peak_times) if len(peak_times) > 1 else np.array([])
        rep_freqs = 1.0 / ibi if len(ibi) else np.array([])

        # Segment & intra-burst f0
        burst_bounds = []
        intraburst_freqs = []
        for p in peaks:
            L, R = segment_burst(env, xf, fs, int(p), pad_ms)
            burst_bounds.append((L, R))
            f0 = dominant_freq(xf[L:R], fs)
            intraburst_freqs.append(f0)
        intraburst_freqs = np.array(intraburst_freqs)

        # Summaries
        summary = {
            "fs_Hz": fs,
            "num_bursts": int(len(peaks)),
            "mean_inter_burst_interval_ms": float(np.mean(ibi)*1000.0) if len(ibi) else np.nan,
            "median_inter_burst_interval_ms": float(np.median(ibi)*1000.0) if len(ibi) else np.nan,
            "mean_repetition_frequency_Hz": float(np.mean(rep_freqs)) if len(rep_freqs) else np.nan,
            "median_repetition_frequency_Hz": float(np.median(rep_freqs)) if len(rep_freqs) else np.nan,
            "median_intraburst_frequency_Hz": float(np.nanmedian(intraburst_freqs)) if len(intraburst_freqs) else np.nan
        }

        per_burst = pd.DataFrame({
            "burst_index": np.arange(len(peaks)),
            "peak_time_s": peak_times.astype(float),
            "intraburst_f0_Hz": intraburst_freqs.astype(float)
        })

        # Store
        self.fs, self.x, self.xf, self.env = fs, x, xf, env
        self.peaks, self.props, self.thr = peaks, props, thr
        self.peak_times, self.ibi, self.rep_freqs = peak_times, ibi, rep_freqs
        self.burst_bounds, self.intraburst_freqs = burst_bounds, intraburst_freqs
        self.summary, self.per_burst_df = summary, per_burst

        # Update GUI
        self.update_stats()
        self.populate_bursts()
        self.update_plots()
        if len(per_burst):
            self.select_tree_index(0)

    def update_stats(self):
        s = self.summary
        self.stats_vars["fs"].set(f"{s.get('fs_Hz', np.nan):.0f}")
        self.stats_vars["num_bursts"].set(f"{s.get('num_bursts', np.nan)}")
        self.stats_vars["mean_ibi_ms"].set(self._fmt_float(s.get('mean_inter_burst_interval_ms', np.nan)))
        self.stats_vars["median_ibi_ms"].set(self._fmt_float(s.get('median_inter_burst_interval_ms', np.nan)))
        self.stats_vars["mean_rep_hz"].set(self._fmt_float(s.get('mean_repetition_frequency_Hz', np.nan)))
        self.stats_vars["median_rep_hz"].set(self._fmt_float(s.get('median_repetition_frequency_Hz', np.nan)))
        self.stats_vars["median_intra_hz"].set(self._fmt_float(s.get('median_intraburst_frequency_Hz', np.nan)))

    def populate_bursts(self):
        # Clear
        for it in self.tree.get_children():
            self.tree.delete(it)
        # Insert
        for i in range(len(self.peaks)):
            t = float(self.peak_times[i])
            f0 = float(self.intraburst_freqs[i]) if i < len(self.intraburst_freqs) else np.nan
            self.tree.insert("", "end", values=(i, f"{t:.6f}", self._fmt_float(f0)))

    def update_plots(self):
        # Plot 1: waveform + envelope + peaks (full duration, decimated for speed)
        self.ax1.cla()
        if self.xf is None or self.env is None:
            self.ax1.set_title("Sygnał (po filtracji) + obwiednia + piki")
        else:
            fs = self.fs
            t = np.arange(len(self.xf)) / fs
            t_plot = decimate_for_plot(t)
            x_plot = decimate_for_plot(self.xf)
            e_plot = decimate_for_plot(self.env)
            self.ax1.plot(t_plot, x_plot, label="bandpass")
            self.ax1.plot(t_plot, e_plot, label="envelope")
            if len(self.peaks):
                self.ax1.plot(self.peak_times, self.env[self.peaks], marker="o", linestyle="None", label="peaks")
            if self.thr is not None:
                self.ax1.axhline(self.thr, linestyle="--", label="threshold")
            self.ax1.set_xlabel("Czas [s]")
            self.ax1.set_ylabel("Amplituda / Obwiednia")
            self.ax1.legend()
        self.ax1.set_title("Sygnał (po filtracji) + obwiednia + piki")
        self.canvas1.draw()

        # Plot 2: histogram IBI
        self.ax2.cla()
        if len(self.ibi):
            self.ax2.hist(self.ibi * 1000.0, bins=20)
        self.ax2.set_title("Histogram odstępów między burstami (IBI)")
        self.ax2.set_xlabel("IBI [ms]")
        self.ax2.set_ylabel("Liczność")
        self.canvas2.draw()

        # Plot 3: FFT first or selected burst
        self.update_fft_plot(selected_index=None)

    def on_select_burst(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        item = sel[0]
        vals = self.tree.item(item, "values")
        if not vals:
            return
        idx = int(vals[0])
        self.update_fft_plot(selected_index=idx)

    def select_tree_index(self, idx):
        children = self.tree.get_children()
        if not children:
            return
        idx = max(0, min(idx, len(children)-1))
        iid = children[idx]
        self.tree.selection_set(iid)
        self.tree.see(iid)
        self.on_select_burst(None)

    def update_fft_plot(self, selected_index=None):
        self.ax3.cla()
        if self.xf is None or not self.burst_bounds:
            self.ax3.set_title("FFT wybranego bursta")
            self.ax3.set_xlabel("Częstotliwość [Hz]")
            self.ax3.set_ylabel("Magnituda")
            self.canvas3.draw()
            return

        if selected_index is None:
            sel = self.tree.selection()
            if sel:
                item = sel[0]
                vals = self.tree.item(item, "values")
                if vals:
                    selected_index = int(vals[0])
            else:
                selected_index = 0

        selected_index = max(0, min(selected_index, len(self.burst_bounds)-1))
        L, R = self.burst_bounds[selected_index]
        seg = self.xf[L:R]
        fs = self.fs
        f0 = dominant_freq(seg, fs)
        # FFT plot
        N = len(seg)
        if N > 16:
            from scipy.signal import get_window
            win = get_window("hann", N, fftbins=True)
            nfft = int(2**np.ceil(np.log2(N)))
            spec = np.abs(np.fft.rfft(seg * win, n=nfft))
            freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
            if len(spec) > 1:
                spec[0] = 0.0
            self.ax3.plot(freqs, spec)
            self.ax3.set_xlim(0, fs/2.0)
        self.ax3.set_title(f"FFT bursta #{selected_index} (dominująca ≈ {self._fmt_float(f0)} Hz)")
        self.ax3.set_xlabel("Częstotliwość [Hz]")
        self.ax3.set_ylabel("Magnituda")
        self.canvas3.draw()

    def save_csv(self):
        if self.per_burst_df is None or self.per_burst_df.empty:
            messagebox.showwarning(APP_TITLE, "Brak wyników do zapisania. Najpierw uruchom analizę.")
            return
        path = filedialog.asksaveasfilename(
            title="Zapisz CSV z wynikami",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialfile="burst_results.csv"
        )
        if not path:
            return
        try:
            # zapisujemy per-burst i summary do dwóch arkuszy CSV? CSV ma jeden arkusz,
            # więc zapisujemy dwa pliki: _per_burst.csv i _summary.json dla prostoty.
            base, ext = os.path.splitext(path)
            per_burst_path = base + "_per_burst.csv"
            summary_path = base + "_summary.json"
            self.per_burst_df.to_csv(per_burst_path, index=False)
            with open(summary_path, "w", encoding="utf-8") as f:
                import json
                json.dump(self.summary, f, ensure_ascii=False, indent=2)
            messagebox.showinfo(APP_TITLE, f"Zapisano:\n{per_burst_path}\n{summary_path}")
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Błąd zapisu CSV: {e}")

    def save_pngs(self):
        if self.xf is None:
            messagebox.showwarning(APP_TITLE, "Brak wykresów do zapisania. Najpierw uruchom analizę.")
            return
        path = filedialog.asksaveasfilename(
            title="Wybierz bazową nazwę pliku PNG",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")],
            initialfile="burst_plots.png"
        )
        if not path:
            return
        base, ext = os.path.splitext(path)
        try:
            self.fig1.savefig(base + "_signal_envelope.png", dpi=150, bbox_inches="tight")
            self.fig2.savefig(base + "_ibi_hist.png", dpi=150, bbox_inches="tight")
            self.fig3.savefig(base + "_fft.png", dpi=150, bbox_inches="tight")
            messagebox.showinfo(APP_TITLE, "Wykresy zapisane.")
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Błąd zapisu PNG: {e}")

    @staticmethod
    def _fmt_float(x):
        try:
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                return "-"
            return f"{float(x):.3f}"
        except Exception:
            return "-"

def main():
    app = BurstAnalyzerApp()
    app.mainloop()

if __name__ == "__main__":
    main()