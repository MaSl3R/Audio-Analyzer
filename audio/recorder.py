import sounddevice as sd
import numpy as np
from collections import deque
import threading


class AudioRecorder:
    def __init__(self, fs=44100, channels=1):
        self.fs = fs
        self.channels = channels

        # -------------> Technika podwójnego buforowania
        self.write_buffer = []  # -------------> Bufor, do którego pisze wątek audio
        self.read_buffer = []  # -------------> Bufor, z którego czyta wątek GUI

        self.stream = None
        self.recording = False
        self.lock = threading.Lock()

        # -------------> Bufor główny, który przechowuje całe nagranie
        self.main_buffer_chunks = []

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")

        if self.recording:
            # -------------> Ta operacja jest bezpieczna bez locka, bo tylko ten wątek modyfikuje write_buffer
            self.write_buffer.append(indata.copy())

    def start(self, duration, device_id=None):
        try:
            self.clear_buffer()
            self.recording = True

            self.stream = sd.InputStream(
                samplerate=self.fs, channels=self.channels, callback=self._callback,
                dtype=np.float32, device=device_id
            )

            self.stream.start()
            device_info = sd.query_devices(self.stream.device, 'input')
            print(f"Recording started on '{device_info['name']}' ({self.stream.channels} channel(s))")

        except Exception as e:
            print(f"Error starting recording: {e}")
            self.recording = False
            raise

    def stop(self):
        self.recording = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                print("Recording stopped")
            except Exception as e:
                print(f"Error stopping recording: {e}")
            finally:
                self.stream = None

        with self.lock:
            if self.write_buffer:
                self.main_buffer_chunks.extend(self.write_buffer)
                self.write_buffer = []

    def get_realtime_buffer(self):
        """Metoda do pobierania NOWYCH danych w czasie rzeczywistym."""
        if not self.write_buffer:
            return np.array([])

        with self.lock:
            # -------------> Zamieniamy bufory miejscami, aby bezpiecznie odczytać dane
            self.read_buffer, self.write_buffer = self.write_buffer, []
            # -------------> Dodajemy odczytane fragmenty do głównego bufora
            self.main_buffer_chunks.extend(self.read_buffer)

        if self.read_buffer:
            return np.concatenate(self.read_buffer, axis=0)
        else:
            return np.array([])

    def get_full_recording(self):
        """Metoda do pobierania całego dotychczasowego nagrania."""
        with self.lock:
            if self.write_buffer:
                self.main_buffer_chunks.extend(self.write_buffer)
                self.write_buffer = []

            if self.main_buffer_chunks:
                return np.concatenate(self.main_buffer_chunks, axis=0)
            else:
                return np.array([])

    def is_recording(self):
        return self.recording

    def get_sample_rate(self):
        return self.fs

    def clear_buffer(self):
        with self.lock:
            self.write_buffer.clear()
            self.read_buffer.clear()
            self.main_buffer_chunks.clear()