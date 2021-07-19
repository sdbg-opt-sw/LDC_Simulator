import sys
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt6.QtGui import QPixmap, QImage, QResizeEvent
from PyQt6.QtCore import Qt
from ui.main import Ui_MainWindow
import cv2
import numpy as np
from pathlib import Path


class DistortionCoefficients:
    def __init__(self, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2

        # Step of each coefficients adjustment
        self.step_k1 = 1.0e-7
        self.step_k2 = 1.0e-12
        self.step_k3 = 1.0e-16
        self.step_p1 = 0.000001
        self.step_p2 = 0.000001

    def get_distortion_coefficients(self):
        dist_coeff = np.zeros((5, 1), dtype=np.float64)
        dist_coeff[0, 0] = self.k1
        dist_coeff[1, 0] = self.k2
        dist_coeff[4, 0] = self.k3
        dist_coeff[2, 0] = self.p1
        dist_coeff[3, 0] = self.p2
        return dist_coeff

    def reset(self):
        self.k1 = 0.0
        self.k2 = 0.0
        self.k3 = 0.0
        self.p1 = 0.0
        self.p2 = 0.0


class LDCSimulatorWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(LDCSimulatorWindow, self).__init__(parent)
        self.setupUi(self)

        # Setup menu bar item
        self.actionOpen_image.triggered.connect(self.menu_open_image)

        # Image data
        self.img = None  # original
        self.img_undist = None  # un-distortion (modified)

        # Distortion coefficients
        self.dist_coeff = DistortionCoefficients()

        # Camera focal length for OpenCV
        self.focal_length = 10.0

        # The maximum and minimum value of each slider bars
        self.horizontalSlider_k1.setMaximum(500)
        self.horizontalSlider_k2.setMaximum(500)
        self.horizontalSlider_k3.setMaximum(500)
        self.horizontalSlider_p1.setMaximum(500)
        self.horizontalSlider_p2.setMaximum(500)
        self.horizontalSlider_k1.setMinimum(-500)
        self.horizontalSlider_k2.setMinimum(-500)
        self.horizontalSlider_k3.setMinimum(-500)
        self.horizontalSlider_p1.setMinimum(-500)
        self.horizontalSlider_p2.setMinimum(-500)

        self.update_distortion_parameters_ui()

        # Setup slider bars
        self.horizontalSlider_k1.valueChanged.connect(self.value_change_k1)
        self.horizontalSlider_k2.valueChanged.connect(self.value_change_k2)
        self.horizontalSlider_k3.valueChanged.connect(self.value_change_k3)
        self.horizontalSlider_p1.valueChanged.connect(self.value_change_p1)
        self.horizontalSlider_p2.valueChanged.connect(self.value_change_p2)

        # Disable distortion parameters panel at startup
        self.groupBox_distortion.setEnabled(False)

        self.action_Save.setEnabled(False)

    def value_change_k1(self):
        self.dist_coeff.k1 = self.horizontalSlider_k1.value() * self.dist_coeff.step_k1
        self.lineEdit_k1.setText(str(self.horizontalSlider_k1.value()))
        self.show_image(self.img)

    def value_change_k2(self):
        self.dist_coeff.k2 = self.horizontalSlider_k2.value() * self.dist_coeff.step_k2
        self.lineEdit_k2.setText(str(self.horizontalSlider_k2.value()))
        self.show_image(self.img)

    def value_change_k3(self):
        self.dist_coeff.k3 = self.horizontalSlider_k3.value() * self.dist_coeff.step_k3
        self.lineEdit_k3.setText(str(self.horizontalSlider_k3.value()))
        self.show_image(self.img)

    def value_change_p1(self):
        self.dist_coeff.p1 = self.horizontalSlider_p1.value() * self.dist_coeff.step_p1
        self.lineEdit_p1.setText(str(self.horizontalSlider_p1.value()))
        self.show_image(self.img)

    def value_change_p2(self):
        self.dist_coeff.p2 = self.horizontalSlider_p2.value() * self.dist_coeff.step_p2
        self.lineEdit_p2.setText(str(self.horizontalSlider_p2.value()))
        self.show_image(self.img)

    def update_distortion_parameters_ui(self):
        val = int((self.dist_coeff.k1 / self.dist_coeff.step_k1))
        self.horizontalSlider_k1.setValue(val)
        self.lineEdit_k1.setText(str(val))

        val = int((self.dist_coeff.k2 / self.dist_coeff.step_k2))
        self.horizontalSlider_k2.setValue(val)
        self.lineEdit_k2.setText(str(val))

        val = int((self.dist_coeff.k3 / self.dist_coeff.step_k3))
        self.horizontalSlider_k3.setValue(val)
        self.lineEdit_k3.setText(str(val))

        val = int((self.dist_coeff.p1 / self.dist_coeff.step_p1))
        self.horizontalSlider_p1.setValue(val)
        self.lineEdit_p1.setText(str(val))

        val = int((self.dist_coeff.p2 / self.dist_coeff.step_p2))
        self.horizontalSlider_p2.setValue(val)
        self.lineEdit_p2.setText(str(val))

    def reset_all_parameters(self):
        self.dist_coeff.reset()
        self.update_distortion_parameters_ui()

    def menu_open_image(self):
        # Popup open file dialog
        path, _ = QFileDialog.getOpenFileNames(self, 'Open an image', '',
                                               'Images (*.jpg *.jpeg *.png *.bmp);;All Files (*.*)')
        if path != ('', '') and len(path) != 0:
            filename = Path(path[0])
            try:
                # Reset all parameters before a new image open
                self.reset_all_parameters()

                # Load image data through OpenCV
                self.img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)

                # Show image on windows
                self.show_image(self.img)

                # Enable distortion parameters panel
                self.groupBox_distortion.setEnabled(True)
            except Exception as e:
                print(e)

    def show_image(self, img: np.ndarray):
        if img is not None:
            # Calculate OpenCV un-distortion
            self.calculate_undistortion(img)

            h, w, d = img.shape

            # Convert ndarray to QT image
            q_img = QImage(self.img_undist.data, w, h, w * d, QImage.Format.Format_BGR888)
            lh = self.label_image.height()
            lw = self.label_image.width()

            # Show image on QLabel
            self.label_image.setPixmap(QPixmap.fromImage(q_img).scaled(lw, lh, Qt.AspectRatioMode.KeepAspectRatio))

    def calculate_undistortion(self, img: np.ndarray):
        height, width, depth = img.shape
        camera_matrix = self.get_camera_matrix(width, height)
        distortion_coefficients = self.dist_coeff.get_distortion_coefficients()
        self.img_undist = cv2.undistort(img, camera_matrix, distortion_coefficients)

    def get_camera_matrix(self, width, height):
        cam = np.eye(3, dtype=np.float32)
        cam[0, 2] = width / 2.0
        cam[1, 2] = height / 2.0
        cam[0, 0] = self.focal_length
        cam[1, 1] = self.focal_length

        return cam

    # FIXME
    # def resizeEvent(self, event: QResizeEvent):
    #     self.show_image(self.img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LDCSimulatorWindow()
    window.show()
    sys.exit(app.exec())
