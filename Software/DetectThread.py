import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from PyQt5.QtCore import QThread, pyqtSignal

from model import BCS_Net


""" Global parameters """
H1 = 6144
W1 = 8192
H2 = 512
W2 = 1024
crop_h = H1//H2
crop_w = W1//W2


def concat_prediction_and_image(prediction, image, save_path):
    concat_out = np.zeros((H1, W1), dtype=np.float32)
    temp_num = 0
    for i in range(crop_h):
        for j in range(crop_w):
            for h in range(H2):
                for w in range(W2):
                    concat_out[h + i * H2][w + j * W2] = prediction[temp_num][h][w]
            temp_num += 1
    concat_out = (1 - concat_out) * image + concat_out * 255
    cv2.imwrite(save_path, concat_out)


class DetectorThread(QThread):
    sin_out = pyqtSignal(str, str)

    def __init__(self, path1, path2, path3):
        super(DetectorThread, self).__init__()
        self.in_path = path1
        self.out_path = path2
        self.weight_path = path3

    def run(self):
        np.random.seed(42)
        tf.random.set_seed(42)
        model = BCS_Net((H2, W2, 1))
        model.summary()
        file_path = self.weight_path + f'/model_best.ckpt'
        model.load_weights(file_path)
        image_list = os.listdir(self.in_path)
        for sub_image in tqdm(image_list):
            name = sub_image.split(".")[0]
            image_path = os.path.join(self.in_path, sub_image)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            prediction_list = []
            for i in range(crop_h):
                for j in range(crop_w):
                    temp = np.zeros((H2, W2), dtype=np.float32)
                    for h in range(H2):
                        for w in range(W2):
                            temp[h][w] = image[h + i * H2][w + j * W2]
                    x = temp/255.0
                    x = np.expand_dims(x, axis=0)
                    y = model.predict(x)[0]
                    y = np.squeeze(y, axis=-1)
                    y = y > 0.5
                    y_pred = y.astype(np.int32)
                    prediction_list.append(y_pred)
            prediction_and_image_path = self.out_path + f'/{name}.jpg'
            concat_prediction_and_image(prediction_list, image, prediction_and_image_path)
            self.sin_out.emit(image_path, prediction_and_image_path)
