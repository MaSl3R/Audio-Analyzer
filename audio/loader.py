import numpy as np
from scipy.io.wavfile import read as wav_read
import wave


def normalize_audio(audio_data):
    """Normalizuje dane audio do zakresu [-1.0, 1.0]."""
    if np.issubdtype(audio_data.dtype, np.integer):
        max_val = np.iinfo(audio_data.dtype).max
        return audio_data.astype(np.float32) / max_val
    elif np.issubdtype(audio_data.dtype, np.floating):
        return np.clip(audio_data, -1.0, 1.0)
    else:
        raise ValueError("Nieobsługiwany format danych audio")


def load_wav(filepath):
    """
    Wczytuje plik WAV i zwraca znormalizowane próbki, częstotliwość próbkowania oraz słownik z metadanymi.
    Konwertuje pliki stereo do mono.
    """
    try:
        sample_rate, data = wav_read(filepath)

        with wave.open(filepath, 'rb') as wf:
            channels = wf.getnchannels()
            bit_depth = wf.getsampwidth() * 8
            duration = wf.getnframes() / wf.getframerate()

        # -------------> Jeśli plik jest stereo, uśrednij kanały do mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        samples = normalize_audio(data)

        metadata = {
            'channels': channels,
            'bit_depth': bit_depth,
            'duration': duration,
            'sample_rate': sample_rate
        }

        return samples, sample_rate, metadata

    except Exception as e:
        print(f"Błąd podczas wczytywania pliku WAV: {e}")
        raise