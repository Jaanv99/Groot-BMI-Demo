# --------------------------------------------------------------------------

# Import relevant modules:
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QMainWindow, QMessageBox, qApp

import sys

# --------------------------------------------------------------------------

# Version #1:

# # Subclass QMainWindow to customize your application's main window
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle("My App")

#         button = QPushButton("I am Groot?")

#         self.setFixedSize(QSize(400, 300))

#         # Set the central widget of the Window.
#         self.setCentralWidget(button)
        
#         button.clicked.connect(self.close)

# def main():

#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec())


# if __name__ == '__main__':
#     main()

# --------------------------------------------------------------------------

# Version #2:

# # Subclass QMainWindow to customize your application's main window
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle("My App")

#         button = QPushButton("I am Groot?")

#         self.setFixedSize(QSize(400, 300))

#         # Set the central widget of the Window.
#         self.setCentralWidget(button)
        
#         # button.clicked.connect(self.close)
        
#         button.clicked.connect(self.setProfile)
        
#         # This defers the call to open the dialog after the main event loop has started
#         # QTimer.singleShot(0, self.setProfile)
        
#     def setProfile(self):
#         if QMessageBox.question(self, "I am Groot?", "I am groot?") != QMessageBox.No:
#             qApp.quit()
#         self.hide()
        
# def main():

#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec())


# if __name__ == '__main__':
#     main()

# --------------------------------------------------------------------------