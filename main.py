from PySide6.QtWidgets import QApplication
import sys
from gui.main_window import LiveAudioAnalyzer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiveAudioAnalyzer()
    window.show()
    sys.exit(app.exec())
