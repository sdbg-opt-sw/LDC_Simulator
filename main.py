import sys
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt6.QtGui import QPixmap, QImage, QResizeEvent, QPainter, QPen, QColor, QMouseEvent
from PyQt6.QtCore import Qt
from ui.main_ui import Ui_MainWindow
import cv2
import numpy as np
from pathlib import Path
import pyperclip

version = 'v1.3'


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

    def __str__(self):
        return 'K1 = {}, K2 = {}, K3 = {}, P1 = {}, P2 = {}'.format(self.k1, self.k2, self.k3, self.p1, self.p2)

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
        self.setWindowTitle('{} ({})'.format(self.windowTitle(), version))

        # Setup menu bar item
        self.actionOpen_image.triggered.connect(self.menu_open_image)
        self.action_Save.triggered.connect(self.menu_save)
        self.actionSave_As.triggered.connect(self.menu_save_as)
        self.actionCopy_parameters.triggered.connect(self.menu_copy_parameters)

        # Image data
        self.img = None  # original
        self.img_undist = None  # un-distortion (modified)

        # Save image data path
        self.save_path = None

        # Distortion coefficients
        self.dist_coeff = DistortionCoefficients()

        # Camera focal length for OpenCV
        self.focal_length = 10.0

        self.show_grids = False
        self.grid_division = self.spinBox_division.value()
        self.grid_color = QColor(255, 0, 0)

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
        self.actionSave_As.setEnabled(False)

        self.groupBox_show_grids.clicked.connect(self.selected_show_grids)
        self.spinBox_division.valueChanged.connect(self.change_grid_division)
        self.radioButton_red.toggled.connect(self.change_grid_color)
        self.radioButton_green.toggled.connect(self.change_grid_color)
        self.radioButton_blue.toggled.connect(self.change_grid_color)
        self.label_image.mousePressEvent = self.label_mouse_press
        self.label_image.mouseReleaseEvent = self.label_mouse_release

    def label_mouse_press(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self.show_image(show_org=True)

    def label_mouse_release(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self.show_image()

    def selected_show_grids(self):
        self.show_grids = self.groupBox_show_grids.isChecked()
        self.show_image()

    def change_grid_division(self):
        self.grid_division = self.spinBox_division.value()
        self.show_image()

    def change_grid_color(self):
        btn = self.sender()
        if btn.isChecked():
            if btn.text() == 'Red':
                self.grid_color = QColor(255, 0, 0)
            elif btn.text() == 'Green':
                self.grid_color = QColor(0, 255, 0)
            elif btn.text() == 'Blue':
                self.grid_color = QColor(0, 0, 255)
        self.show_image()

    def value_change_k1(self):
        self.dist_coeff.k1 = self.horizontalSlider_k1.value() * self.dist_coeff.step_k1
        self.lineEdit_k1.setText(str(self.horizontalSlider_k1.value()))
        self.show_image()

    def value_change_k2(self):
        self.dist_coeff.k2 = self.horizontalSlider_k2.value() * self.dist_coeff.step_k2
        self.lineEdit_k2.setText(str(self.horizontalSlider_k2.value()))
        self.show_image()

    def value_change_k3(self):
        self.dist_coeff.k3 = self.horizontalSlider_k3.value() * self.dist_coeff.step_k3
        self.lineEdit_k3.setText(str(self.horizontalSlider_k3.value()))
        self.show_image()

    def value_change_p1(self):
        self.dist_coeff.p1 = self.horizontalSlider_p1.value() * self.dist_coeff.step_p1
        self.lineEdit_p1.setText(str(self.horizontalSlider_p1.value()))
        self.show_image()

    def value_change_p2(self):
        self.dist_coeff.p2 = self.horizontalSlider_p2.value() * self.dist_coeff.step_p2
        self.lineEdit_p2.setText(str(self.horizontalSlider_p2.value()))
        self.show_image()

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
        if path:
            filename = Path(path[0])
            try:
                # Reset all parameters before a new image open
                self.reset_all_parameters()

                # Load image data through OpenCV
                self.img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                print(e)

            # Show image on windows
            self.show_image()

            # Enable distortion parameters panel
            self.groupBox_distortion.setEnabled(True)
            self.action_Save.setEnabled(True)
            self.actionSave_As.setEnabled(True)

    def show_image(self, show_org=False):
        img = self.img
        if img is not None:
            h, w, d = img.shape

            if not show_org:
                # Calculate OpenCV un-distortion
                self.calculate_undistortion(img)
                data = self.img_undist.data
            else:
                data = self.img.data

            # Convert ndarray to QT image
            q_img = QImage(data, w, h, w * d, QImage.Format.Format_BGR888)
            lh = self.label_image.height()
            lw = self.label_image.width()

            # Create pixmap
            pixmap = QPixmap.fromImage(q_img).scaled(lw, lh, Qt.AspectRatioMode.KeepAspectRatio)

            if self.show_grids:
                # Draw grids
                painter = QPainter(pixmap)
                pen_color = self.grid_color
                pen_width = 1
                div = self.grid_division

                pen = QPen(pen_color, pen_width)
                painter.setPen(pen)
                for i in range(1, div):
                    painter.drawLine(0, int(i * pixmap.height() / div), pixmap.width(), int(i * pixmap.height() / div))
                    painter.drawLine(int(i * pixmap.width() / div), 0, int(i * pixmap.width() / div), pixmap.height())
                painter.end()

            # Show image on QLabel
            self.label_image.setPixmap(pixmap)

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

    def save_image(self):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 98]
        cv2.imencode('.jpg', self.img_undist, encode_param)[1].tofile(self.save_path)

    def menu_save(self):
        if self.save_path is None:
            self.menu_save_as()
        else:
            self.save_image()

    def menu_save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save File As', '', 'Jpeg (*.jpg)')
        if path:
            self.save_path = path
            self.save_image()

    def menu_copy_parameters(self):
        if self.img is not None:
            pyperclip.copy('k1 = {} / k2 = {} / k3 = {} / p1 = {} / p2 = {}'.format(
                self.horizontalSlider_k1.value(),
                self.horizontalSlider_k2.value(),
                self.horizontalSlider_k3.value(),
                self.horizontalSlider_p1.value(),
                self.horizontalSlider_p2.value()))

    # FIXME
    # def resizeEvent(self, event: QResizeEvent):
    #     self.show_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LDCSimulatorWindow()
    window.show()
    sys.exit(app.exec())
