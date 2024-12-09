import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
import tensorflow as tf

from model import BCS_Net


""" Global parameters """
H = 512
W = 1024
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*jpg")))
    y = sorted(glob(os.path.join(path, "mask", "*bmp")))
    return x, y


def element_eval(y_true, y_pred):
    TP = (y_true * y_pred).sum()
    y_true_inverse = 1 - y_true
    FP = (y_true_inverse * y_pred).sum()
    y_pred_inverse = 1 - y_pred
    FN = (y_true * y_pred_inverse).sum()
    return TP, FP, FN


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """

    """ Loading model """
    model = BCS_Net((H, W, 1))
    model.summary()

    model.load_weights(f"weight/model_best.ckpt")

    """ Load the dataset """
    test_x, test_y = load_data("data/test")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Evaluation and Prediction """
    TP_Sum = FP_Sum = FN_Sum = 0
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extract the name """
        name = x.split("\\")[-1].split(".")[0]

        """ Reading the 2D image """
        image = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
        x = image/255.0
        x = np.expand_dims(x, axis=0)

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        """ Prediction """
        y_pred = model.predict(x)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred > 0.5
        y_pred0 = y_pred.astype(np.int32)

        """ Saving the prediction """
        y_pred1 = y_pred * 255
        cv2.imwrite(f"result/{name}.jpg", y_pred1)

        """ Flatten the array """
        mask = mask.flatten()
        y_pred = y_pred0.flatten()

        """ Calculating the metrics values """
        TP, FP, FN = element_eval(mask, y_pred)
        TP_Sum += TP
        FP_Sum += FP
        FN_Sum += FN

    """ Metrics values """
    mIOU = (TP_Sum + 1e-15) / (TP_Sum + FN_Sum + FP_Sum + 1e-15)
    mPreci = (TP_Sum + 1e-15) / (TP_Sum + FP_Sum + 1e-15)
    mRecal = (TP_Sum + 1e-15) / (TP_Sum + FN_Sum + 1e-15)
    mFmea = (2 * mPreci * mRecal + 1e-15) / (mPreci + mRecal + 1e-15)

    print(f"TP: {TP_Sum}, FP: {FP_Sum},FN: {FN_Sum}")
    print(f"mPre: {mPreci:0.4f}, mRec:{mRecal:0.4f}, mFea:{mFmea:0.4f},mIOU:{mIOU:0.4f}")
