import sys
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QResizeEvent, QPixmap, QImage, QKeySequence, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QLabel, QShortcut, QVBoxLayout

from DetectorWindow import Ui_DetectorWindow
from DetectThread import DetectorThread

""" Global parameters """
H1 = 6144
W1 = 8192
H2 = 512
W2 = 1024
crop_h = H1//H2
crop_w = W1//W2


class MyWindow(QMainWindow, Ui_DetectorWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.in_path = ""
        self.out_path = ""
        self.weight_path = ""
        self.label_in = QLabel("")
        self.label_in.setParent(self)
        self.label_out = QLabel("")
        self.label_out.setParent(self)
        QShortcut(QKeySequence(self.tr("esc")), self, self.close)
        self.pushButton_input.setShortcut('I')
        self.pushButton_output.setShortcut('O')
        self.pushButton_weightfile.setShortcut('W')
        self.pushButton_detect.setShortcut('D')
        self.setWindowTitle("Intelligent Detection System of Bridge Surface Cracks")
        self.setWindowIcon(QIcon('Icon/Icon.png'))
        self.setStyleSheet("#DetectorWindow{border-image:url(Icon/Background.png)}")

        self.groupBox_EIF.setStyleSheet("QGroupBox { color: yellow; border: 2px solid red; "
                                        "border-radius:10px; padding:2px 4px}")
        self.groupBox_EIF.setAlignment(Qt.AlignCenter)
        self.groupBox_input.setStyleSheet("QGroupBox { color: yellow; border: 2px solid red; "
                                          "border-radius:10px; padding:2px 4px}")
        self.groupBox_input.setAlignment(Qt.AlignCenter)
        self.groupBox_output.setStyleSheet("QGroupBox { color: yellow; border: 2px solid red; "
                                           "border-radius:10px; padding:2px 4px}")
        self.groupBox_output.setAlignment(Qt.AlignCenter)

        self.pushButton_input.clicked.connect(self.load_folder)
        self.pushButton_output.clicked.connect(self.output_folder)
        self.pushButton_weightfile.clicked.connect(self.weight_folder)
        self.pushButton_detect.clicked.connect(self.detect)

    def load_folder(self):
        self.in_path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if self.in_path == "":
            QMessageBox.warning(self, 'Tip', 'Please Select the Correct Folder Path!')
        else:
            self.lineEdit_inputfolder.setAlignment(Qt.AlignCenter)
            self.lineEdit_inputfolder.setText(self.in_path)

    def output_folder(self):
        self.out_path = QFileDialog.getExistingDirectory(self, "Select Output folder")
        if self.out_path == "":
            QMessageBox.warning(self, 'Tip', 'Please Select the Correct Folder Path!')
        else:
            self.lineEdit_outputfolder.setAlignment(Qt.AlignCenter)
            self.lineEdit_outputfolder.setText(self.out_path)

    def weight_folder(self):
        self.weight_path = QFileDialog.getExistingDirectory(self, "Select Model Weight folder")
        if self.weight_path == "":
            QMessageBox.warning(self, 'Tip', 'Please Select the Correct Folder Path!')
        else:
            self.lineEdit_weightfile.setAlignment(Qt.AlignCenter)
            self.lineEdit_weightfile.setText(self.weight_path)

    def show_result(self, path_in, path_out):
        img_in = QImage(path_in)
        self.label_in.setScaledContents(True)
        self.label_in.setPixmap(QPixmap.fromImage(img_in))
        self.label_in.setScaledContents(True)
        self.label_in.show()
        img_out = QImage(path_out)
        self.label_out.setScaledContents(True)
        self.label_out.setPixmap(QPixmap.fromImage(img_out))
        self.label_out.setScaledContents(True)
        self.label_out.show()

    def detect(self):
        self.DetectorThread = DetectorThread(self.in_path, self.out_path, self.weight_path)
        self.DetectorThread.start()
        self.DetectorThread.sin_out.connect(self.show_result)

    def resizeEvent(self, a0: QResizeEvent) -> None:
        super().resizeEvent(a0)
        group_box_layout_in = QVBoxLayout()
        group_box_layout_in.setContentsMargins(10, 10, 10, 10)
        group_box_layout_in.addWidget(self.label_in)
        self.groupBox_input.setLayout(group_box_layout_in)
        group_box_layout_out = QVBoxLayout()
        group_box_layout_out.setContentsMargins(10, 10, 10, 10)
        group_box_layout_out.addWidget(self.label_out)
        self.groupBox_output.setLayout(group_box_layout_out)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_window = MyWindow()
    my_window.show()
    sys.exit(app.exec_())
