import numpy as np
import matplotlib.pyplot as plt
import math


def frequency_to_note(frequency):
    """
    Konwertuje częstotliwość w Hz na najbliższą nutę muzyczną.
    """
    if frequency <= 0:
        return None
    A4_FREQ = 440.0
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    num_semitones = 12 * math.log2(frequency / A4_FREQ)
    num_semitones = round(num_semitones)
    note_index_from_a4 = 9 + num_semitones
    octave = 4 + (note_index_from_a4 // 12)
    note_index = note_index_from_a4 % 12
    note_name = NOTE_NAMES[note_index]
    return f"{note_name}{octave}"


def plot_time_domain(ax, samples, duration, rms, peak):
    """
    Rysuje sygnał w dziedzinie czasu na podstawie dostarczonych danych.
    """
    ax.clear()
    ax.set_facecolor('black')
    if len(samples) == 0:
        ax.text(0.5, 0.5, 'Brak danych', transform=ax.transAxes, color='white', ha='center', va='center')
        return

    time_axis = np.linspace(0, duration, len(samples))
    if samples.ndim > 1 and samples.shape[1] > 1:
        ax.plot(time_axis, samples[:, 0], color='cyan', linewidth=0.8)
        ax.plot(time_axis, samples[:, 1], color='red', linewidth=0.8)
    else:
        ax.plot(time_axis, samples, color='cyan', linewidth=0.8)

    ax.grid(True, alpha=0.3, color='white')
    ax.set_xlim(0, duration)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("Sygnał w dziedzinie czasu", color='white', fontsize=12)
    ax.set_xlabel("Czas [s]", color='white')
    ax.set_ylabel("Amplituda", color='white')
    ax.tick_params(colors='white')
    ax.text(0.02, 0.98, f'RMS: {rms:.3f}\nPeak: {peak:.3f}', transform=ax.transAxes, color='yellow',
            verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))


def plot_frequency_domain(ax, xf, yf_db, dominant_freq, note, fs):
    """
    Rysuje widmo częstotliwościowe na podstawie dostarczonych danych.
    """
    ax.clear()
    ax.set_facecolor('black')
    if len(xf) == 0:
        ax.text(0.5, 0.5, 'Brak danych', transform=ax.transAxes, color='white', ha='center', va='center')
        return

    ax.plot(xf, yf_db, color='magenta', linewidth=0.8)
    ax.grid(True, alpha=0.3, color='white')
    ax.set_xlim(0, min(fs / 2, 8000))
    if yf_db.size > 0:
        # -------------> Filtrujemy wartości -inf, aby uniknąć błędów
        finite_yf_db = yf_db[np.isfinite(yf_db)]
        if finite_yf_db.size > 0:
            ax.set_ylim(np.min(finite_yf_db) - 10, np.max(finite_yf_db) + 10)

    ax.set_title("Widmo częstotliwościowe", color='white', fontsize=12)
    ax.set_xlabel("Częstotliwość [Hz]", color='white')
    ax.set_ylabel("Amplituda [dB]", color='white')
    ax.tick_params(colors='white')

    info_text = f'Dominująca częstotliwość: {dominant_freq:.0f} Hz'
    if note:
        info_text += f'\nNajbliższa nuta: {note}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, color='yellow',
            verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))


def setup_plot_style():
    """Konfiguruje globalny styl wykresów."""
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = 'black'
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['savefig.facecolor'] = 'black'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'


def plot_spectrogram(fig, ax, samples, fs):
    """Rysuje spektrogram, uśredniając sygnał stereo do mono."""
    ax.clear()
    ax.set_facecolor('black')
    if len(samples) == 0:
        ax.text(0.5, 0.5, 'Brak danych', transform=ax.transAxes, color='white', ha='center', va='center')
        return

    if samples.ndim > 1 and samples.shape[1] > 1:
        samples_mono = samples.mean(axis=1)
    else:
        samples_mono = samples if samples.ndim == 1 else samples.flatten()

    Pxx, freqs, bins, im = ax.specgram(samples_mono, NFFT=1024, Fs=fs, noverlap=512, cmap='viridis')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Intensywność [dB]', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    ax.set_title("Spektrogram (Mono)", color='white', fontsize=12)
    ax.set_xlabel("Czas [s]", color='white')
    ax.set_ylabel("Częstotliwość [Hz]", color='white')
    ax.tick_params(colors='white')