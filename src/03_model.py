import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3

def squeeze_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filter_num = init.shape[channel_axis]
    squeeze_shape = (1, 1, filter_num)
    squeeze = GlobalAveragePooling2D()(tensor)
    squeeze = Reshape(squeeze_shape)(squeeze)
    squeeze = Dense(
        filter_num // ratio,
        activation='relu',
        kernel_initializer='he_normal',
        use_bias=False
    )(squeeze)
    squeeze = Dense(
        filter_num,
        activation='sigmoid',
        kernel_initializer='he_normal',
        use_bias=False
    )(squeeze)
    x = Multiply(name='attention_conv')([tensor, squeeze])
    return x

def add_common_layers(y):
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y

def build_model():
    inp = Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        name='input_layer'
    )
    x0 = Conv2D(
        64,
        (7,7),
        strides=(2,2),
        padding='same',
        name='stem_conv'
    )(inp)
    x0 = add_common_layers(x0)
    conv1 = Conv2D(
        32,
        (1,1),
        activation='relu',
        padding='same',
        name='feature_block_1_conv1'
    )(x0)
    conv2 = Conv2D(
        64,
        (2,2),
        activation='relu',
        padding='same',
        name='feature_block_1_conv2'
    )(x0)
    conv3 = Conv2D(
        128,
        (3,3),
        activation='relu',
        padding='same',
        name='feature_block_1_conv3'
    )(x0)
    conv4 = Conv2D(
        256,
        (5,5),
        activation='relu',
        padding='same',
        name='feature_block_1_conv4'
    )(x0)
    out = Concatenate(name='feature_block_1')([
        conv1,
        conv2,
        conv3,
        conv4
    ])
    x = Conv2D(
        128,
        (1,1),
        activation='relu',
        padding='same'
    )(out)
    x = add_common_layers(x)
    x = MaxPooling2D((2,2))(x)
    x = squeeze_block(x)
    conv5 = Conv2D(
        32,
        (1,1),
        activation='relu',
        padding='same',
        name='feature_block_2_conv1'
    )(x)
    conv6 = Conv2D(
        64,
        (2,2),
        activation='relu',
        padding='same',
        name='feature_block_2_conv2'
    )(x)
    conv7 = Conv2D(
        128,
        (3,3),
        activation='relu',
        padding='same',
        name='feature_block_2_conv3'
    )(x)
    conv8 = Conv2D(
        256,
        (5,5),
        activation='relu',
        padding='same',
        name='final_feature_conv'
    )(x)
    out2 = Concatenate(name='feature_block_2')([
        conv5,
        conv6,
        conv7,
        conv8
    ])
    x = Conv2D(
        128,
        (1,1),
        activation='relu',
        padding='same'
    )(out2)
    x = add_common_layers(x)
    x = MaxPooling2D((2,2))(x)
    x = squeeze_block(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(
        3,
        activation='softmax',
        name='binary_output'
    )(x)
    model = Model(
        inputs=inp,
        outputs=output,
        name='SE_SPNet'
    )
    return model