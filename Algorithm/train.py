import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam

from model import BCS_Net

""" Global parameters """
H = 512
W = 1024
smooth = 1e-15
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class DICE_LOSS(object):
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)
        loss = (1 - ((2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)))
        return loss


class Pre(object):
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return (tf.reduce_sum((y_true * y_pred)) + smooth)/(tf.reduce_sum(y_pred) + smooth)


class Rec(object):
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return (tf.reduce_sum((y_true * y_pred)) + smooth)/(tf.reduce_sum(y_true) + smooth)


class F1(object):
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)
        return (2 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


class IOU(object):
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        intersection = tf.reduce_sum((y_true * y_pred))
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        x = (intersection + smooth) / (union + smooth)
        x = tf.cast(x, tf.float32)
        return x


def create_csv(path):
    title_list = ['epoch', 'train_loss', 'train_precision', 'train_recall', 'train_f1', 'train_iou', 'validation_loss',
                  'validation_precision', 'validation_recall', 'validation_f1', 'validation_iou']
    csv = pd.DataFrame([title_list])
    out_path = os.path.join(path, 'data.csv')
    csv.to_csv(out_path, header=False, index=False)


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*jpg")))
    y = sorted(glob(os.path.join(path, "mask", "*bmp")))
    return x, y


def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x


def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 1])
    y.set_shape([H, W, 1])
    return x, y


def tf_dataset(X, Y, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


if __name__ == "__main__":
    """GPU Checking"""
    cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
    print("GPU Available: " + f"{cuda_gpu_available}")

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    create_csv("weight")

    """ Hyperparameters """
    batch_size = 2
    initial_lr = 1e-4
    num_epochs = 120
    model_path = os.path.join("weight", "model_best.ckpt")

    """ Dataset """
    train_x, train_y = load_data("data/train")
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_data("data/validation")

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Model """
    model = BCS_Net((H, W, 1))
    model.summary()

    loss_object = DICE_LOSS()
    iou_object = IOU()
    f1_object = F1()
    precision_object = Pre()
    recall_object = Rec()
    optimizer = Adam(learning_rate=initial_lr)

    train_loss = Mean(name='train_loss')
    train_precision = Mean(name='train_precision')
    train_recall = Mean(name='train_recall')
    train_f1 = Mean(name='train_f1')
    train_iou = Mean(name='train_iou')

    val_loss = Mean(name='val_loss')
    val_precision = Mean(name='val_precision')
    val_recall = Mean(name='val_recall')
    val_f1 = Mean(name='val_f1')
    val_iou = Mean(name='val_iou')

    @tf.function
    def train_step(train_images, train_labels):
        with tf.GradientTape() as tape:
            output = model(train_images, training=True)
            loss = loss_object(train_labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        pre = precision_object(train_labels, output)
        rec = recall_object(train_labels, output)
        f1 = f1_object(train_labels, output)
        iou = iou_object(train_labels, output)
        train_loss(loss)
        train_precision(pre)
        train_recall(rec)
        train_f1(f1)
        train_iou(iou)

    @tf.function
    def val_step(val_images, val_labels):
        output = model(val_images, training=False)
        loss = loss_object(val_labels, output)
        pre = precision_object(val_labels, output)
        rec = recall_object(val_labels, output)
        f1 = f1_object(val_labels, output)
        iou = iou_object(val_labels, output)
        val_loss(loss)
        val_precision(pre)
        val_recall(rec)
        val_f1(f1)
        val_iou(iou)

    best_val_iou = 0.
    temp_num = 0
    for epoch in range(num_epochs):
        train_loss.reset_states()
        train_precision.reset_states()
        train_recall.reset_states()
        train_f1.reset_states()
        train_iou.reset_states()
        val_loss.reset_states()
        val_precision.reset_states()
        val_recall.reset_states()
        val_f1.reset_states()
        val_iou.reset_states()

        if temp_num == 3:
            new_lr = initial_lr * 0.8
            optimizer.learning_rate = new_lr
            initial_lr = new_lr
            temp_num = 0

        train_bar = tqdm(train_dataset)
        print("Epoch[{}/{}]".format(epoch + 1, num_epochs))
        for images, labels in train_bar:
            train_step(images, labels)
            train_bar.desc = "train_loss:{:.4f}, train_precision:{:.4f}, train_recall:{:.4f}, train_f1:{:.4f}, " \
                             "train_iou:{:.4f}".format(train_loss.result(),  train_precision.result(),
                                                       train_recall.result(), train_f1.result(), train_iou.result())
        val_bar = tqdm(valid_dataset)
        for images, labels in val_bar:
            val_step(images, labels)
            val_bar.desc = "validation_loss:{:.4f}, validation_precision:{:.4f}, validation_recall:{:.4f}, " \
                           "validation_f1:{:.4f}, validation_iou:{:.4f}".format(val_loss.result(),
                                                                                val_precision.result(),
                                                                                val_recall.result(), val_f1.result(),
                                                                                val_iou.result())
        Epoch = "%d" % epoch
        train_LOSS = "%f" % train_loss.result()
        train_PRECISION = "%f" % train_precision.result()
        train_RECALL = "%f" % train_recall.result()
        train_F1 = "%f" % train_f1.result()
        training_IOU = "%f" % train_iou.result()
        val_LOSS = "%f" % val_loss.result()
        val_PRECISION = "%f" % val_precision.result()
        val_RECALL = "%f" % val_recall.result()
        val_F1 = "%f" % val_f1.result()
        val_IOU = "%f" % val_iou.result()

        index_list = [Epoch, train_LOSS, train_PRECISION, train_RECALL, train_F1, training_IOU,
                      val_LOSS, val_PRECISION, val_RECALL, val_F1, val_IOU]
        data = pd.DataFrame([index_list])
        data.to_csv("files/data.csv", mode='a', header=False,
                    index=False)

        if val_iou.result() > best_val_iou:
            temp_num = 0
            best_val_iou = val_iou.result()
            model.save_weights(model_path)
        else:
            temp_num += 1
