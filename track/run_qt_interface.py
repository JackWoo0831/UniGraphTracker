"""
Create PyQt Interface for demo
"""

# system
import sys
from torch.utils.data import DataLoader

# qt
from PyQt5.QtWidgets import QApplication, QMainWindow
from qt_window import Ui_MainWindow

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = Ui_MainWindow(window)

    window.show()
    exit(app.exec_())