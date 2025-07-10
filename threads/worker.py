from PySide6.QtCore import QObject, Signal, Slot
import numpy as np

# -------------> Import funkcji do konwersji
from plots.plot_utils import frequency_to_note


class AnalysisWorker(QObject):
    """
    Wykonuje ciężkie obliczenia analityczne w osobnym wątku,
    aby nie blokować głównego wątku GUI.
    """
    # -------------> Sygnał emitowany po zakończeniu analizy
    results_ready = Signal(dict)

    @Slot(np.ndarray, int)
    def run_analysis(self, samples, fs):
        """
        Główna metoda robocza. Przyjmuje surowe próbki i wykonuje analizę.
        """
        if samples.size == 0:
            self.results_ready.emit({})
            return

        # -------------> Analiza w dziedzinie czasu
        mono_samples = samples.mean(axis=1) if samples.ndim > 1 else samples
        rms = np.sqrt(np.mean(mono_samples ** 2))
        peak = np.max(np.abs(mono_samples))

        # -------------> Analiza w dziedzinie częstotliwości
        N = len(mono_samples)
        if N < 2:
            self.results_ready.emit({})
            return

        windowed_samples = mono_samples * np.hanning(N)
        yf = np.abs(np.fft.rfft(windowed_samples))
        xf = np.fft.rfftfreq(N, 1 / fs)
        yf_db = 20 * np.log10(yf + 1e-12)

        dominant_freq = 0
        note = None
        if len(yf) > 1:
            dominant_freq_idx = np.argmax(yf[1:]) + 1
            dominant_freq = xf[dominant_freq_idx]
            note = frequency_to_note(dominant_freq)

        results = {
            'samples': samples,  # -------------> Przekazujemy oryginalne próbki
            'rms': rms,
            'peak': peak,
            'yf_db': yf_db,
            'xf': xf,
            'dominant_freq': dominant_freq,
            'note': note
        }

        self.results_ready.emit(results)