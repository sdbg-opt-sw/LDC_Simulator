import sys
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from ui.main import Ui_MainWindow
import cv2
import numpy as np
from pathlib import Path


class LDC_Simulator_Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(LDC_Simulator_Window, self).__init__(parent)
        self.setupUi(self)

        self.img = None

        self.k1 = 0.0
        self.k2 = 0.0
        self.k3 = 0.0
        self.p1 = 0.0
        self.p2 = 0.0

        self.focal_length = 10.0

        self.step_k1 = 1.0e-6
        self.step_k2 = 1.0e-12
        self.step_k3 = 1.0e-15
        self.step_p1 = 0.000001
        self.step_p2 = 0.000001

        self.actionOpen_image.triggered.connect(self.menu_open_image)
        self.update_distortion_parameters_ui()

        self.horizontalSlider_k1.valueChanged.connect(self.change_k1)
        self.horizontalSlider_k2.valueChanged.connect(self.change_k2)
        self.horizontalSlider_k3.valueChanged.connect(self.change_k3)
        self.horizontalSlider_p1.valueChanged.connect(self.change_p1)
        self.horizontalSlider_p2.valueChanged.connect(self.change_p2)

    def change_k1(self):
        self.k1 = (self.horizontalSlider_k1.value() - 50) * self.step_k1
        self.update_distortion_parameters_ui()
        self.show_image(self.img)

    def change_k2(self):
        self.k2 = (self.horizontalSlider_k2.value() - 50) * self.step_k2
        self.update_distortion_parameters_ui()
        self.show_image(self.img)

    def change_k3(self):
        self.k3 = (self.horizontalSlider_k3.value() - 50) * self.step_k3
        self.update_distortion_parameters_ui()
        self.show_image(self.img)

    def change_p1(self):
        self.p1 = (self.horizontalSlider_p1.value() - 50) * self.step_p1
        self.update_distortion_parameters_ui()
        self.show_image(self.img)

    def change_p2(self):
        self.p2 = (self.horizontalSlider_p2.value() - 50) * self.step_p2
        self.update_distortion_parameters_ui()
        self.show_image(self.img)

    def update_distortion_parameters_ui(self):
        val = int((self.k1 / self.step_k1) + 50)
        self.horizontalSlider_k1.setValue(val)
        self.lineEdit_k1.setText(str(self.k1))

        val = int((self.k2 / self.step_k2) + 50)
        self.horizontalSlider_k2.setValue(val)
        self.lineEdit_k2.setText(str(self.k2))

        val = int((self.k3 / self.step_k3) + 50)
        self.horizontalSlider_k3.setValue(val)
        self.lineEdit_k3.setText(str(self.k3))

        val = int((self.p1 / self.step_p1) + 50)
        self.horizontalSlider_p1.setValue(val)
        self.lineEdit_p1.setText(str(self.p1))

        val = int((self.p2 / self.step_p2) + 50)
        self.horizontalSlider_p2.setValue(val)
        self.lineEdit_p2.setText(str(self.p2))

    def menu_open_image(self):
        path, _ = QFileDialog.getOpenFileNames(self, 'Open an image', '',
                                               'Images (*.jpg *.jpeg *.png *.bmp);;All Files (*.*)')
        if path != ('', '') and len(path) != 0:
            filename = Path(path[0])
            try:
                self.img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.show_image(self.img)
            except Exception as e:
                print(e)

    def show_image(self, img: np.ndarray):
        h, w, d = img.shape
        img_undistort = cv2.undistort(self.img,
                                      self.get_camera_matrix(w, h),
                                      self.get_distortion_coefficients())
        q_img = QImage(img_undistort.data, w, h, w * d, QImage.Format.Format_BGR888)
        lh = self.label_image.height()
        lw = self.label_image.width()
        self.label_image.setPixmap(QPixmap.fromImage(q_img).scaled(lw, lh, Qt.AspectRatioMode.KeepAspectRatio))

    def get_distortion_coefficients(self):
        dist_coeff = np.zeros((5, 1), dtype=np.float64)
        dist_coeff[0, 0] = self.k1
        dist_coeff[1, 0] = self.k2
        dist_coeff[4, 0] = self.k3
        dist_coeff[2, 0] = self.p1
        dist_coeff[3, 0] = self.p2

        return dist_coeff

    def update_distortion_coefficients(self, k1=None, k2=None, k3=None, p1=None, p2=None):
        if k1:
            self.k1 = k1
        if k2:
            self.k2 = k2
        if k3:
            self.k3 = k3
        if p1:
            self.p1 = p1
        if p2:
            self.p2 = p2

    def get_camera_matrix(self, width, height):
        cam = np.eye(3, dtype=np.float32)
        cam[0, 2] = width / 2.0
        cam[1, 2] = height / 2.0
        cam[0, 0] = self.focal_length
        cam[1, 1] = self.focal_length

        return cam


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LDC_Simulator_Window()
    window.show()
    sys.exit(app.exec())
