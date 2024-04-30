import math
import tensorflow as tf
from keras.layers import *
from keras import regularizers
from keras import backend as K
from modles.upsampling import *

tf.reset_default_graph()


def SE(x, name, ratio=8):
    print("SE_block")
    out_dim = int(x.shape[-1])
    squeeze = GlobalAveragePooling2D()(x)
    excitation = Dense(units=int(out_dim//ratio))(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('relu')(excitation)
    excitation = Reshape((1, 1, out_dim))(excitation)
    scale = multiply([x, excitation])
    return scale


def ECA(input, name, gamma=2, b=1, ratio=2, re_co=1):
    channels = K.int_shape(input)[-1]
    raw_input_feature = input
    t = int(abs((math.log(channels, 2) + b) / gamma))
    k = t if t % 2 else t + 1
    x_global_avg_pool = GlobalAveragePooling2D()(input)
    x = Reshape((channels, 1))(x_global_avg_pool)
    x = Conv1D(1, kernel_size=k, padding='same', name=name + 'eca_conv1')(x)
    x = Activation('sigmoid', name=name + 'eca_softmax')(x)
    x = Reshape((1, 1, channels))(x)
    output = multiply([raw_input_feature, x])
    print("hello ECA!")
    return output


def eca_attention(input, name, gamma=2, b=1, **kwargs):
    channels = K.int_shape(input)[-1]
    raw_input_feature = input
    input = input
    t = int(abs((math.log(channels, 2) + b)/gamma))
    k = t if t % 2 else t+1
    x_global_avg_pool = GlobalAveragePooling2D()(input)
    x = Reshape((channels, 1))(x_global_avg_pool)
    x = Conv1D(1, kernel_size=k, padding='same', name=name+'eca_conv1')(x)
    x = Activation('sigmoid', name=name+'eca_conv1_softmax')(x)
    x = Reshape((1, 1, channels))(x)
    output = multiply([raw_input_feature, x])
    return output


def channel_attention(input_feature, high_level_feature, ratio=2, name='cbam_block', use_cbam=False):

    if use_cbam:
        raw_input_feature = input_feature
    else:
        raw_input_feature = input_feature
        input_feature = Add()([input_feature, high_level_feature])
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid', name=name + 'cbam_feature_sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([raw_input_feature, cbam_feature])


def spatial_attention(input_feature, name='cbam_block_spatial', use_cbam=False):
    kernel_size = 7
    #  this is spatial input feature shape:  (?, 32, 32, 6)
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    if use_cbam:
        return multiply([input_feature, cbam_feature])
    return cbam_feature



def CBAM(input_feature, name, ratio=8, use_cbam=True, link_place=1):
    with tf.variable_scope(name):
        attention_feature = channel_attention(input_feature, [], name=name+'ch_at', ratio=ratio, use_cbam=True)
        attention_feature = spatial_attention(attention_feature, name=name+'sp_at', use_cbam=use_cbam)
        print("CBAM Hello")  # sequential data transmission
    return attention_feature



def PAB(input, ratio=2, name='attention_block_', re_co=0.01, channel_num=2, link_place=0):

    output_channel = int(input.shape[3])
    h = int(input.shape[1])

    low_level_feature, high_level_feature = convway(input, name=name+'convway_', re_co=re_co, channel_num=channel_num)
    high_level_feature = Add(name=name+'link2high')([input, high_level_feature])
    x = eca_attention(high_level_feature, name=name+'eca_block_')
    x = Activation('sigmoid', name=name+'activate')(x)
    x = BatchNormalization(name=name+'channel_BN')(x)
    x = UpSamplingBilinear_me(targetsize=h)(x)
    x1 = x
    sa_input = x
    x = spatial_attention(sa_input, name=name + 'SA_')
    x = multiply([x, low_level_feature])
    x1 = x1
    x = Add(name=name+'final_add')([x, x1])
    print("hello pyramid attention block")
    return x


def convway(input, upsample=True, name='convway', re_co=0.01, channel_num=1):
    output_dim = int(input.shape[3])
    l2_efficient = 0.01
    kernel_size = 1
    # Block 1
    filters = output_dim//3
    input = Conv2D(filters, kernel_size=(1, 1), use_bias=False, padding='same')(input)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_efficient),
               name=name + 'block1_conv1')(input)
    c1 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name=name + 'block1_pool')(x)
    x = BatchNormalization(name=name+'block1_BN')(x)
    x = Activation('relu', name=name+'relu_1')(x)

    # Block 2
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_efficient),
               name=name + 'block2_conv1')(x)
    c2 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name=name + 'block2_pool')(x)
    x = BatchNormalization(name=name + 'block2_BN')(x)
    x = Activation('relu', name=name+'relu_2')(x)

    # Block 3
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_efficient),
               name=name + 'block3_conv1')(x)
    c3 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name=name + 'block3_pool')(x)
    x = BatchNormalization(name=name + 'block3_BN')(x)
    x = Activation('relu', name=name + 'relu_3')(x)


    # Block 4
    x = BatchNormalization(name=name + 'block4_BN')(x)
    x = Activation('relu', name=name + 'relu_4')(x)
    c4 = x
    if upsample:

        c1 = Conv2D(filters=output_dim, kernel_size=kernel_size, padding='same', name=name + 'c1_conv_c1')(c1)
        c2 = Conv2D(filters=output_dim, kernel_size=kernel_size, padding='same', name=name + 'c1_conv_c2')(c2)
        c1 = BatchNormalization(name=name+'c1_BN')(c1)
        c2 = BatchNormalization(name=name+'c2_BN')(c2)
        c1 = UpSamplingBilinear_me(targetsize=input.shape[2])(c1)
        c2 = UpSamplingBilinear_me(targetsize=input.shape[2])(c2)
        low_level_feature = Add(name=name + 'low_level_feature')([c1, c2])
        low_level_feature = Activation('sigmoid', name=name + 'c12_sigmoid')(low_level_feature)

        c3 = Conv2D(filters=output_dim, kernel_size=kernel_size, padding='same', name=name + 'c1_conv_c3')(c3)
        c4 = Conv2D(filters=output_dim, kernel_size=kernel_size, padding='same', name=name + 'c1_conv_c4')(c4)
        c3 = BatchNormalization(name=name+'c3_BN')(c3)
        c4 = BatchNormalization(name=name+'c4_BN')(c4)
        c3 = UpSamplingBilinear_me(targetsize=input.shape[2])(c3)
        c4 = UpSamplingBilinear_me(targetsize=input.shape[2])(c4)
        high_level_feature = Add(name=name + "high_level_feature")([c3, c4])
        high_level_feature = Activation('sigmoid', name=name + 'c34_sigmoid')(high_level_feature)

    return low_level_feature, high_level_feature

def convwway1x1(input, upsample=True, name='convway', re_co=0.01, channel_num=1):
    output_dim = int(input.shape[3])
    l2_efficient = 0.01
    kernel_size = 1
    # Block 1
    filters_1 = output_dim//3
    filters_2 = output_dim //9

    low = Conv2D(filters_1, kernel_size=(1, 1), use_bias=False, padding='same')(input)
    high = Conv2D(filters_2, kernel_size=(1, 1), use_bias=False, padding='same')(input)
    input = Conv2D(output_dim, kernel_size=(1, 1), use_bias=False, padding='same')(input)
    low_level_feature = process_dim(low, output_dim)
    low_level_feature = Activation('sigmoid', name=name + 'c12_sigmoid')(low_level_feature)


    high_level_feature = process_dim(high, output_dim)
    high_level_feature = Activation('sigmoid', name=name + 'c34_sigmoid')(high_level_feature)

    return low_level_feature, high_level_feature

def process_dim(x, out_dim):
    return Conv2D(out_dim, kernel_size=(1, 1), use_bias=False, padding='same')(x)


def del_ca(input, ratio=2, name='attention_block_', re_co=0.01, channel_num=2, link_place=0):

    output_channel = int(input.shape[3])
    h = int(input.shape[1])

    low_level_feature, high_level_feature = convway(input, name=name+'convway_', re_co=re_co, channel_num=channel_num)
    high_level_feature = Add(name=name+'link2high')([input, high_level_feature])
    x = high_level_feature
    x = Activation('sigmoid', name=name+'activate')(x)
    x = BatchNormalization(name=name+'channel_BN')(x)
    x = UpSamplingBilinear_me(targetsize=h)(x)
    x1 = x
    sa_input = x
    x = spatial_attention(sa_input, name=name + 'SA_')
    x = multiply([x, low_level_feature])
    x1 = x1
    x = Add(name=name+'final_add')([x, x1])
    x = Activation('sigmoid', name=name+"block_sigmoid")(x)
    print("hello del ca block")
    return x


def del_sa(input, ratio=2, name='attention_block_', re_co=0.01, channel_num=2, link_place=0):

    output_channel = int(input.shape[3])
    h = int(input.shape[1])

    low_level_feature, high_level_feature = convway(input, name=name+'convway_', re_co=re_co, channel_num=channel_num)
    high_level_feature = Add(name=name+'link2high')([input, high_level_feature])
    x = eca_attention(high_level_feature, name=name+'eca_block_')
    x = Activation('sigmoid', name=name+'activate')(x)
    x = BatchNormalization(name=name+'channel_BN')(x)
    x = UpSamplingBilinear_me(targetsize=h)(x)
    x1 = x
    x = multiply([x, low_level_feature])
    x1 = x1
    x = Add(name=name+'final_add')([x, x1])
    x = Activation('sigmoid', name=name+"block_sigmoid")(x)
    print("hello del sa block")
    return x


def del_conv(input, ratio=2, name='attention_block_', re_co=0.01, channel_num=2, link_place=0):
    h = int(input.shape[1])
    x = eca_attention(input, name=name+'eca_block_')
    x = Activation('sigmoid', name=name+'activate')(x)
    x = BatchNormalization(name=name+'channel_BN')(x)
    x = UpSamplingBilinear_me(targetsize=h)(x)
    x1 = x
    sa_input = x
    x = spatial_attention(sa_input, name=name + 'SA_')
    x = multiply([x, input])
    x1 = x1
    x = Add(name=name+'final_add')([x, x1])
    x = Activation('sigmoid', name=name+"block_sigmoid")(x)
    print("hello del conv block")
    return x


def only_ca(input, ratio=2, name='attention_block_', re_co=0.01, channel_num=2, link_place=0):

    output_channel = int(input.shape[3])
    h = int(input.shape[1])
    x = eca_attention(input, name=name+'eca_block_')
    x = Activation('sigmoid', name=name+'activate')(x)
    x = BatchNormalization(name=name+'channel_BN')(x)
    x = UpSamplingBilinear_me(targetsize=h)(x)
    x1 = x
    x = multiply([x, input])
    x = Add(name=name+'final_add')([x, x1])
    x = Activation('sigmoid', name=name+"block_sigmoid")(x)
    print("hello only ca block")
    return x


def only_sa(input, ratio=2, name='attention_block_', re_co=0.01, channel_num=2, link_place=0):
    output_channel = int(input.shape[3])
    h = int(input.shape[1])
    x = Activation('sigmoid', name=name+'activate')(input)
    x = BatchNormalization(name=name+'channel_BN')(x)
    x = UpSamplingBilinear_me(targetsize=h)(x)
    x1 = x
    sa_input = x
    x = spatial_attention(sa_input, name=name + 'SA_')
    x = multiply([x, input])
    x = Add(name=name+'final_add')([x, x1])
    x = Activation('sigmoid', name=name+"block_sigmoid")(x)
    print("hello only sa block")
    return x

def only_fe(input, ratio=2, name='attention_block_', re_co=0.01, channel_num=2, link_place=0):

    output_channel = int(input.shape[3])
    h = int(input.shape[1])

    low_level_feature, high_level_feature = convway(input, name=name+'convway_', re_co=re_co, channel_num=channel_num)
    high_level_feature = Add(name=name+'link2high')([input, high_level_feature])
    x = Activation('sigmoid', name=name+'activate')(high_level_feature)
    x = BatchNormalization(name=name+'channel_BN')(x)
    x = UpSamplingBilinear_me(targetsize=h)(x)
    x1 = x
    x = multiply([x, low_level_feature])
    x1 = x1
    x = Add(name=name+'final_add')([x, x1])
    x = Activation('sigmoid', name=name+"block_sigmoid")(x)
    print("hello only fe block")
    return x




