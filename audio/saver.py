import numpy as np
from scipy.io.wavfile import write as wav_write
import os


def save_wav(filename, samples, fs):
    """
    Zapisuje nagranie do pliku WAV
    """
    try:
        # Upewnij się, że samples to numpy array
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples)

        # Normalizuj jeśli amplituda przekracza zakres
        if np.max(np.abs(samples)) > 1.0:
            samples = samples / np.max(np.abs(samples))

        # Skaluj float32 do int16
        int_samples = np.int16(samples * 32767)

        # Upewnij się, że folder istnieje
        os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None

        wav_write(filename, fs, int_samples)
        print(f"WAV file saved: {filename}")

    except Exception as e:
        print(f"Error saving WAV file: {e}")
        raise


def get_supported_formats():
    """
    Zwraca listę wspieranych formatów.
    """
    # Zawsze zwracamy tylko WAV
    return ["WAV"]


def validate_filename(filename, format_type):
    """
    Sprawdza poprawność nazwy pliku i dodaje rozszerzenie jeśli potrzeba
    """
    if not filename:
        raise ValueError("Filename cannot be empty")

    format_type = format_type.lower()
    expected_ext = f".{format_type}"

    if not filename.endswith(expected_ext):
        filename += expected_ext

    return filename