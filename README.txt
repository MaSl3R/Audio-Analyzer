# Audio Analyzer v1.0

Audio Analyzer is an advanced desktop application developed in Python using the PySide6 framework for comprehensive audio signal analysis. The program provides tools to visualize sound from both a live microphone input and pre-recorded `.wav` files. It is a versatile utility for hobbyists, musicians, audio engineers, and anyone interested in "seeing" sound.

The project is optimized for performance by offloading computationally intensive tasks, such as the Fast Fourier Transform (FFT), to a separate worker thread. This ensures a smooth and responsive user interface, even during real-time analysis.

---

## Key Features

- **Dual-Mode Operation:**
  - **Live Analysis:** Process audio in real-time directly from a selected microphone.
  - **File Analysis:** Load `.wav` files for detailed, offline inspection.
- **Drag & Drop Support:** Intuitively load audio files by dragging and dropping them onto the application window.
- **Advanced Visualization:**
  - **Time-Domain Plot:** Displays the signal's amplitude over time, complete with RMS and Peak value calculations.
  - **Frequency-Domain Plot (FFT):** Analyzes the signal's frequency components, revealing its tonal structure.
  - **Spectrogram:** A 3D visualization showing how the frequency spectrum evolves over time.
- **Interactive Plots:** A built-in Matplotlib navigation toolbar allows users to zoom and pan the plots for close-up analysis.
- **Intelligent Signal-Processing:**
  - **Dominant Frequency Detection:** Automatically identifies the most prominent frequency in the signal.
  - **Musical Note Recognition:** Translates the dominant frequency into the nearest musical note (e.g., 440 Hz -> A4), turning the application into a simple instrument tuner.
- **Test Tone Generator:** An integrated tool to generate and instantly analyze sine wave tones of a user-specified frequency, perfect for testing and calibration.
- **Data Export:**
  - Save the analyzed audio clip to a `.wav` file.
  - Export the generated plots and spectrograms to various image formats (PNG, JPG, SVG).

---


## License & Acknowledgements

This project is licensed under the MIT License - see the `LICENSE` file for details.

This application is built using several third-party libraries, each with its own license. The full license texts are included in the `LICENSES` directory.

- **Qt for Python (PySide6):** Licensed under the LGPL v3.
- **NumPy & SciPy:** Licensed under the BSD 3-Clause License.
- **Matplotlib:** Licensed under a PSF-style license.
- **SoundDevice:** Licensed under the MIT License.

