from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel,
    QComboBox, QFileDialog, QMessageBox, QProgressBar, QGroupBox,
    QDialog, QLineEdit
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, Slot
from PySide6.QtGui import QFont, QIcon, QIntValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from collections import deque
import numpy as np
import sounddevice as sd
import traceback
import os

from audio.recorder import AudioRecorder
from audio.saver import save_wav, validate_filename, get_supported_formats
from audio.loader import load_wav
from plots.plot_utils import plot_time_domain, plot_frequency_domain, setup_plot_style, plot_spectrogram
from threads.worker import AnalysisWorker

import resources_rc


class LiveAudioAnalyzer(QWidget):
    recording_started = Signal()
    recording_stopped = Signal()
    error_occurred = Signal(str)
    analysis_trigger = Signal(np.ndarray, int)

    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(":/icon.ico"))
        self.setWindowTitle("Analizator Dźwięku")
        self.setMinimumSize(1000, 700)
        self.setAcceptDrops(True)

        self.thread = QThread()
        self.worker = AnalysisWorker()
        self.worker.moveToThread(self.thread)
        self.analysis_trigger.connect(self.worker.run_analysis)
        self.worker.results_ready.connect(self.update_plots_from_results)
        self.thread.start()

        self.duration = 5
        self.is_recording = False
        self.last_samples = np.array([])
        self.input_devices = []
        self.current_fs = 44100
        self.app_mode = 'live'

        try:
            self.recorder = AudioRecorder()
            self.current_fs = self.recorder.get_sample_rate()
            print(f"✅ AudioRecorder zainicjowany (Sample rate: {self.current_fs}Hz)")
        except Exception as e:
            QMessageBox.critical(self, "Błąd inicjalizacji", f"Nie można zainicjować nagrywania audio:\n{str(e)}")
            self.recorder = None

        setup_plot_style()
        self.init_ui()
        self.recording_started.connect(self.on_recording_started)
        self.recording_stopped.connect(self.on_recording_stopped)
        self.error_occurred.connect(self.on_error)
        self._populate_device_list()
        self.update_ui_for_mode()

    def _populate_device_list(self):
        try:
            devices = sd.query_devices()
            self.input_devices = [dev for dev in devices if dev['max_input_channels'] > 0]
            self.device_combo.clear()
            if not self.input_devices:
                self.device_combo.addItem("Brak mikrofonów")
                self.device_combo.setEnabled(False)
                self.button.setEnabled(False)
                return
            for device in self.input_devices: self.device_combo.addItem(device['name'])
            try:
                default_device_name = sd.query_devices(kind='input')['name']
                self.device_combo.setCurrentText(default_device_name)
            except Exception:
                pass
        except Exception as e:
            self.error_occurred.emit(f"Błąd podczas wczytywania urządzeń audio: {e}")
            self.device_combo.addItem("Błąd wczytywania")
            self.device_combo.setEnabled(False)

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(15)
        control_panel = self.create_control_panel()
        plot_area = self.create_plot_area()
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(plot_area, 4)
        self.init_timers()

    def create_control_panel(self):
        control_widget = QWidget()
        control_widget.setMaximumWidth(250)
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_layout.setSpacing(15)
        source_group = QGroupBox("Źródło dźwięku")
        source_layout = QVBoxLayout(source_group)
        self.load_file_button = QPushButton("📂 Załaduj plik audio")
        self.load_file_button.clicked.connect(self.load_audio_file)
        source_layout.addWidget(self.load_file_button)
        self.loaded_file_label = QLabel("Aktywne źródło: Mikrofon")
        self.loaded_file_label.setWordWrap(True)
        source_layout.addWidget(self.loaded_file_label)
        self.file_info_duration_label = QLabel()
        self.file_info_samplerate_label = QLabel()
        self.file_info_channels_label = QLabel()
        self.file_info_bitdepth_label = QLabel()
        for label in [self.file_info_duration_label, self.file_info_samplerate_label, self.file_info_channels_label,
                      self.file_info_bitdepth_label]:
            source_layout.addWidget(label)
            label.setVisible(False)
        control_layout.addWidget(source_group)
        tone_group = QGroupBox("Generator tonów")
        tone_layout = QVBoxLayout(tone_group)
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Częstotliwość [Hz]:"))
        self.tone_freq_input = QLineEdit("440")
        self.tone_freq_input.setValidator(QIntValidator(20, 20000))
        freq_layout.addWidget(self.tone_freq_input)
        tone_layout.addLayout(freq_layout)
        self.play_tone_button = QPushButton("🔊 Odtwórz i analizuj ton")
        self.play_tone_button.clicked.connect(self.play_test_tone)
        tone_layout.addWidget(self.play_tone_button)
        control_layout.addWidget(tone_group)
        self.recording_group = QGroupBox("Ustawienia nagrywania")
        recording_layout = QVBoxLayout(self.recording_group)
        recording_layout.addWidget(QLabel("Wybierz mikrofon:"))
        self.device_combo = QComboBox()
        recording_layout.addWidget(self.device_combo)
        self.time_label = QLabel(f"Czas nagrania: {self.duration} s")
        recording_layout.addWidget(self.time_label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(30)
        self.slider.setValue(self.duration)
        self.slider.valueChanged.connect(self.update_duration)
        recording_layout.addWidget(self.slider)
        self.button = QPushButton("🎙️ Start analizy")
        self.button.setMinimumHeight(40)
        self.button.clicked.connect(self.toggle_stream)
        recording_layout.addWidget(self.button)
        if not self.recorder: self.button.setEnabled(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        recording_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.recording_group)
        save_group = QGroupBox("Zapis")
        save_layout = QVBoxLayout(save_group)
        self.save_button = QPushButton("💾 Zapisz dźwięk (.wav)")
        self.save_button.clicked.connect(self.save_recording)
        save_layout.addWidget(self.save_button)
        self.save_plot_button = QPushButton("🖼️ Zapisz wykresy")
        self.save_plot_button.clicked.connect(self.save_plots)
        save_layout.addWidget(self.save_plot_button)
        self.spectrogram_button = QPushButton("📊 Pokaż spektrogram")
        self.spectrogram_button.clicked.connect(self.show_spectrogram)
        save_layout.addWidget(self.spectrogram_button)
        control_layout.addWidget(save_group)
        self.status_label = QLabel("Gotowy.")
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()
        return control_widget

    def create_plot_area(self):
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 5, 0, 0)
        self.figure = Figure(facecolor='black', tight_layout=True)
        self.canvas = FigureCanvas(self.figure)

        toolbar = NavigationToolbar(self.canvas, self)
        toolbar.setStyleSheet("background-color: #333; color: white;")

        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.canvas)

        self.ax_time = self.figure.add_subplot(2, 1, 1)
        self.ax_fft = self.figure.add_subplot(2, 1, 2)
        self.update_empty_plots()
        return plot_container

    def init_timers(self):
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.trigger_analysis)
        self.stop_timer = QTimer()
        self.stop_timer.setSingleShot(True)
        self.stop_timer.timeout.connect(self.stop_recording)
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)

    def trigger_analysis(self):
        try:
            samples_to_analyze = np.array([])
            if self.is_recording:
                samples_to_analyze = self.recorder.get_full_recording()
            elif self.app_mode == 'file' and self.last_samples.size > 0:
                samples_to_analyze = self.last_samples
                self.plot_timer.stop()

            if samples_to_analyze.size > 0:
                self.analysis_trigger.emit(samples_to_analyze, self.current_fs)
        except Exception as e:
            print(f"Błąd w trigger_analysis: {e}")
            traceback.print_exc()

    @Slot(dict)
    def update_plots_from_results(self, results):
        if not results:
            self.update_empty_plots()
            return

        try:
            samples = results['samples']
            self.last_samples = samples
            duration = len(samples) / self.current_fs if self.current_fs > 0 else 0

            plot_time_domain(self.ax_time, samples, duration, results['rms'], results['peak'])
            plot_frequency_domain(self.ax_fft, results['xf'], results['yf_db'], results['dominant_freq'],
                                  results['note'], self.current_fs)

            self.canvas.draw()
        except Exception as e:
            print(f"Błąd w update_plots_from_results: {e}")
            traceback.print_exc()

    def update_empty_plots(self):
        plot_time_domain(self.ax_time, np.array([]), 0, 0, 0)
        plot_frequency_domain(self.ax_fft, np.array([]), np.array([]), 0, None, self.current_fs)
        self.canvas.draw()

    def update_duration(self, value):
        self.duration = value
        self.time_label.setText(f"Czas nagrania: {self.duration} s")

    def toggle_stream(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if self.app_mode == 'file': self.switch_to_live_mode()
        if hasattr(self.canvas.toolbar, 'home'): self.canvas.toolbar.home()
        if not self.recorder:
            self.error_occurred.emit("Recorder nie jest zainicjowany")
            return
        selected_index = self.device_combo.currentIndex()
        if selected_index < 0 or not self.input_devices:
            self.error_occurred.emit("Nie wybrano prawidłowego mikrofonu.")
            return
        device_id = self.input_devices[selected_index]['index']
        try:
            self.last_samples = np.array([])
            self.recorder.start(self.duration, device_id=device_id)
            self.is_recording = True
            self.button.setText("⏹️ Stop analizy")
            self.button.setStyleSheet("background-color: #ff4444;")
            self.update_ui_for_mode()
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, self.duration * 10)
            self.progress_bar.setValue(0)
            self.plot_timer.start(100)
            self.stop_timer.start(self.duration * 1000)
            self.progress_timer.start(100)
            self.recording_started.emit()
        except Exception as e:
            self.error_occurred.emit(f"Błąd podczas rozpoczynania nagrywania: {str(e)}")

    def stop_recording(self):
        if not self.recorder: return
        try:
            self.recorder.stop()
            self.is_recording = False
            self.plot_timer.stop()
            self.stop_timer.stop()
            self.progress_timer.stop()
            self.button.setText("🎙️ Start analizy")
            self.button.setStyleSheet("")
            self.progress_bar.setVisible(False)
            self.last_samples = self.recorder.get_full_recording()
            self.update_ui_for_mode()
            self.trigger_analysis()
            self.recording_stopped.emit()
        except Exception as e:
            self.error_occurred.emit(f"Błąd podczas zatrzymywania nagrywania: {str(e)}")

    def update_progress(self):
        if self.is_recording:
            current_value = self.progress_bar.value()
            self.progress_bar.setValue(current_value + 1)

    def update_ui_for_mode(self):
        is_live_mode = (self.app_mode == 'live')
        has_data = (self.last_samples.size > 0)
        self.device_combo.setEnabled(is_live_mode)
        self.slider.setEnabled(is_live_mode)
        self.time_label.setEnabled(is_live_mode)
        can_operate = has_data and not self.is_recording
        self.save_button.setEnabled(can_operate)
        self.save_plot_button.setEnabled(can_operate)
        self.spectrogram_button.setEnabled(can_operate)
        file_info_visible = (self.app_mode == 'file')
        for label in [self.file_info_duration_label, self.file_info_samplerate_label, self.file_info_channels_label,
                      self.file_info_bitdepth_label]:
            label.setVisible(file_info_visible)

    def process_audio_file(self, filepath):
        try:
            samples, sample_rate, metadata = load_wav(filepath)
            self.last_samples = samples
            self.current_fs = sample_rate
            self.app_mode = 'file'
            self.loaded_file_label.setText(f"<b>Aktywny plik:</b>\n{os.path.basename(filepath)}")
            self.file_info_duration_label.setText(f"<b>Długość:</b> {metadata['duration']:.2f} s")
            self.file_info_samplerate_label.setText(f"<b>Próbkowanie:</b> {metadata['sample_rate']} Hz")
            self.file_info_channels_label.setText(f"<b>Kanały:</b> {metadata['channels']}")
            self.file_info_bitdepth_label.setText(f"<b>Głębia bitowa:</b> {metadata['bit_depth']}-bit")
            self.status_label.setText("Załadowano plik.")
            self.update_ui_for_mode()
            self.plot_timer.start(100)
        except Exception as e:
            QMessageBox.critical(self, "Błąd wczytywania", f"Nie udało się wczytać pliku:\n{e}")

    def load_audio_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Otwórz plik audio", "", "Pliki WAV (*.wav)")
        if filepath:
            self.process_audio_file(filepath)

    def switch_to_live_mode(self):
        if self.is_recording: self.stop_recording()
        self.app_mode = 'live'
        self.last_samples = np.array([])
        self.current_fs = self.recorder.get_sample_rate() if self.recorder else 44100
        self.loaded_file_label.setText("<b>Aktywne źródło:</b> Mikrofon")
        for label in [self.file_info_duration_label, self.file_info_samplerate_label, self.file_info_channels_label,
                      self.file_info_bitdepth_label]:
            label.setText("")
        self.status_label.setText("Gotowy do nagrywania.")
        self.update_ui_for_mode()
        self.update_empty_plots()

    def play_test_tone(self):
        try:
            frequency = int(self.tone_freq_input.text())
            tone_duration = 1.0;
            amplitude = 0.5;
            fs = 44100
            t = np.linspace(0., tone_duration, int(fs * tone_duration), endpoint=False)
            samples = amplitude * np.sin(2 * np.pi * frequency * t)
            self.status_label.setText(f"Odtwarzanie tonu {frequency} Hz...")
            sd.play(samples.astype(np.float32), fs)
            self.app_mode = 'file'
            self.last_samples = samples
            self.current_fs = fs
            self.loaded_file_label.setText(f"<b>Aktywny sygnał:</b>\nTon testowy {frequency} Hz")
            for label in [self.file_info_duration_label, self.file_info_samplerate_label, self.file_info_channels_label,
                          self.file_info_bitdepth_label]:
                label.setText("")
            self.update_ui_for_mode()
            self.plot_timer.start(100)
            sd.wait()
            self.status_label.setText("Zakończono odtwarzanie.")
        except ValueError:
            QMessageBox.warning(self, "Błąd", "Wprowadź poprawną liczbę jako częstotliwość.")
        except Exception as e:
            QMessageBox.critical(self, "Błąd odtwarzania", f"Nie udało się odtworzyć dźwięku:\n{str(e)}")

    def save_recording(self):
        if self.last_samples.size == 0: return
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz dźwięk", "nagranie.wav", "Pliki WAV (*.wav)")
        if path:
            try:
                path = validate_filename(path, "wav")
                save_wav(path, self.last_samples, self.current_fs)
                QMessageBox.information(self, "Sukces", f"Nagranie zapisano do:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Błąd zapisu", f"Nie udało się zapisać pliku:\n{str(e)}")

    def save_plots(self):
        if self.last_samples.size == 0: return
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz wykresy jako...", "wykresy.png",
                                              "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;SVG Files (*.svg)")
        if path:
            try:
                self.figure.savefig(path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Sukces", f"Wykresy zapisano do:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Błąd zapisu", f"Nie udało się zapisać wykresów:\n{str(e)}")

    def show_spectrogram(self):
        if self.last_samples.size == 0: return
        dialog = QDialog(self)
        dialog.setWindowTitle("Spektrogram")
        dialog.setMinimumSize(800, 600)
        fig = Figure(facecolor='black')
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        plot_spectrogram(fig, ax, self.last_samples, self.current_fs)
        save_button = QPushButton("💾 Zapisz spektrogram")

        def save_action():
            path, _ = QFileDialog.getSaveFileName(dialog, "Zapisz spektrogram jako...", "spektrogram.png",
                                                  "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;SVG Files (*.svg)")
            if path:
                try:
                    fig.savefig(path, dpi=300, bbox_inches='tight')
                    QMessageBox.information(dialog, "Sukces", f"Spektrogram zapisano do:\n{path}")
                except Exception as e:
                    QMessageBox.critical(dialog, "Błąd zapisu", f"Nie udało się zapisać pliku:\n{str(e)}")

        save_button.clicked.connect(save_action)
        layout = QVBoxLayout(dialog)
        layout.addWidget(canvas)
        layout.addWidget(save_button)
        canvas.draw()
        dialog.exec()

    def on_recording_started(self):
        self.status_label.setText("Nagrywanie...")

    def on_recording_stopped(self):
        self.status_label.setText("Nagrywanie zakończone.")
        self.update_ui_for_mode()

    def on_error(self, error_message):
        self.status_label.setText("Błąd krytyczny.")
        QMessageBox.critical(self, "Błąd", error_message)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile() and url.toLocalFile().lower().endswith('.wav'):
                event.acceptProposedAction()

    def dropEvent(self, event):
        filepath = event.mimeData().urls()[0].toLocalFile()
        self.process_audio_file(filepath)

    def closeEvent(self, event):
        if self.is_recording: self.stop_recording()
        self.thread.quit()
        self.thread.wait()
        event.accept()