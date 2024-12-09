from tensorflow.keras.layers import Conv2D, Input, Add, UpSampling2D, AveragePooling2D, Dense, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation, Concatenate, MaxPooling2D, MaxPool2D, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
import tensorflow as tf
import math


def HDSModule(inputs):
    """ Hybrid Down Sampling Module """

    shape = inputs.shape
    _, h, w, c = shape[0], shape[1], shape[2], shape[3]
    variable_ratio = tf.nn.sigmoid(tf.Variable(tf.random.truncated_normal([1], mean=0.0, stddev=0.05),
                                               dtype=tf.float32, trainable=True))
    branch1_fil = math.ceil(variable_ratio * c)
    branch2_fil = c - branch1_fil
    branch1_in = inputs[:, :, :, :branch1_fil]
    branch2_in = inputs[:, :, :, branch1_fil:]

    conv1 = tf.keras.Sequential([MaxPool2D(pool_size=(2, 2))])(branch1_in)
    conv2 = tf.keras.Sequential([Conv2D(filters=branch2_fil, kernel_size=3, strides=2, padding='same', use_bias=False),
                                 BatchNormalization(), Activation("relu")])(branch2_in)
    out = Concatenate()([conv1, conv2])
    out = Sequential([Conv2D(filters=c * 2, kernel_size=1, use_bias=False),
                      BatchNormalization(), Activation("relu")])(out)
    return out


def HUSModule(inputs):
    """ Hybrid Up Sampling Module """

    shape = inputs.shape
    _, h, w, c = shape[0], shape[1], shape[2], shape[3]
    variable_ratio = tf.nn.sigmoid(tf.Variable(tf.random.truncated_normal([1], mean=0.0, stddev=0.05),
                                               dtype=tf.float32, trainable=True))
    branch1_fil = math.ceil(variable_ratio * c)
    branch2_fil = c - branch1_fil
    branch1_in = inputs[:, :, :, :branch1_fil]
    branch2_in = inputs[:, :, :, branch1_fil:]

    conv1 = Sequential([UpSampling2D((2, 2), interpolation='bilinear')])(branch1_in)
    conv2 = Sequential([Conv2DTranspose(filters=branch2_fil, kernel_size=3, strides=2, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu")])(branch2_in)
    concat = Concatenate()([conv1, conv2])
    out = Sequential([Conv2D(filters=c // 2, kernel_size=1, use_bias=False),
                      BatchNormalization(), Activation("relu")])(concat)
    return out


def MFSM(inputs):
    """ Modified Feature Selection Module """

    shape = inputs.shape
    out_channel = shape[-1]
    avg_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    max_pool = MaxPooling2D(pool_size=(shape[1], shape[2]))(inputs)
    avg_pool = Sequential([Dense(out_channel // 8, activation='relu', kernel_initializer='he_normal', use_bias=False,
                                 bias_initializer='zeros'),
                           Dense(out_channel, kernel_initializer='he_normal', use_bias=False,
                                 bias_initializer='zeros')])(avg_pool)
    max_pool = Sequential([Dense(out_channel // 8, activation='relu', kernel_initializer='he_normal', use_bias=False,
                                 bias_initializer='zeros'),
                           Dense(out_channel, kernel_initializer='he_normal', use_bias=False,
                                 bias_initializer='zeros')])(max_pool)
    out1 = Add()([avg_pool, max_pool])
    out1 = Activation('sigmoid')(out1)
    out = out1 * inputs
    out = out + inputs
    out = Sequential([Conv2D(filters=out_channel, kernel_size=1, use_bias=False), BatchNormalization()])(out)
    return out


def MCA(inputs):
    """ Modified Coordinate Attention"""

    shape = inputs.shape
    _, h, w, c = shape[0], shape[1], shape[2], shape[3]

    avg_pool_h = Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(inputs)
    avg_pool_h = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(avg_pool_h)
    avg_pool_w = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(inputs)
    ct_avg = Concatenate(axis=2)([avg_pool_h, avg_pool_w])
    conv_avg_0 = tf.keras.Sequential([Conv2D(filters=c // 4, kernel_size=1, padding='same', use_bias=False),
                                      BatchNormalization(), Activation("relu")])(ct_avg)
    conv_avg_h, conv_avg_w = Lambda(lambda x: tf.split(x, num_or_size_splits=[h, w], axis=2))(conv_avg_0)
    conv_avg_h = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(conv_avg_h)

    max_pool_h = Lambda(lambda x: K.max(x, axis=2, keepdims=True))(inputs)
    max_pool_h = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(max_pool_h)
    max_pool_w = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(inputs)
    ct_max = Concatenate(axis=2)([max_pool_h, max_pool_w])
    conv_max_0 = tf.keras.Sequential([Conv2D(filters=c // 4, kernel_size=1, padding='same', use_bias=False),
                                      BatchNormalization(), Activation("relu")])(ct_max)
    conv_max_h, conv_max_w = Lambda(lambda x: tf.split(x, num_or_size_splits=[h, w], axis=2))(conv_max_0)
    conv_max_h = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(conv_max_h)

    add_h = Add()([conv_avg_h, conv_max_h])
    add_w = Add()([conv_avg_w, conv_max_w])
    out_add_h = tf.keras.Sequential([Conv2D(filters=c, kernel_size=1, padding='same', use_bias=False),
                                     BatchNormalization(), Activation("sigmoid")])(add_h)
    out_add_w = tf.keras.Sequential([Conv2D(filters=c, kernel_size=1, padding='same', use_bias=False),
                                     BatchNormalization(), Activation("sigmoid")])(add_w)
    out_add = out_add_h * out_add_w
    out = inputs * out_add
    return out


def BCS_Net(shape):
    """ Bridge Crack Segmentation Network """

    inputs = Input(shape)
    stem0 = Sequential([Conv2D(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu"),
                        Conv2D(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu")])(inputs)
    conv1 = Sequential([Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu"),
                        Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu")])(stem0)
    mp1 = HDSModule(conv1)
    conv2 = Sequential([Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu"),
                        Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu")])(mp1)
    mp2 = HDSModule(conv2)
    conv3 = Sequential([Conv2D(filters=512, kernel_size=3, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu"),
                        Conv2D(filters=512, kernel_size=3, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu")])(mp2)
    up1 = HUSModule(conv3)
    skip1 = MFSM(conv2)
    concat1 = Concatenate()([skip1, up1])
    conv4 = Sequential([Conv2D(filters=256, kernel_size=1, use_bias=False),
                        BatchNormalization(), Activation("relu")])(concat1)
    up2 = HUSModule(conv4)
    skip2 = MFSM(conv1)
    concat2 = Concatenate()([skip2, up2])
    conv5 = Sequential([Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu"),
                        Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu")])(concat2)
    mp3 = HDSModule(conv5)
    ct1 = MCA(conv4)
    ct1 = Concatenate()([ct1, mp3])
    conv6 = Sequential([Conv2D(filters=256, kernel_size=1, use_bias=False),
                        BatchNormalization(), Activation("relu")])(ct1)
    mp4 = HDSModule(conv6)
    ct2 = MCA(conv3)
    ct2 = Concatenate()([ct2, mp4])
    conv7 = Sequential([Conv2D(filters=512, kernel_size=3, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu"),
                        Conv2D(filters=512, kernel_size=3, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu")])(ct2)
    up3 = HUSModule(conv7)
    skip3 = MFSM(conv6)
    concat3 = Concatenate()([skip3, up3])
    conv8 = Sequential([Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu"),
                        Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu")])(concat3)
    up4 = HUSModule(conv8)
    skip4 = MFSM(conv5)
    concat4 = Concatenate()([skip4, up4])
    conv9 = Sequential([Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu"),
                        Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu")])(concat4)
    stem1 = Sequential([Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu"),
                        Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same', use_bias=False),
                        BatchNormalization(), Activation("relu")])(conv9)
    out = Sequential([Conv2D(filters=1, kernel_size=3, padding='same', use_bias=False), Activation("sigmoid")])(stem1)

    return Model(inputs, out)


if __name__ == '__main__':
    model = BCS_Net((512, 1024, 1))
    model.summary()
