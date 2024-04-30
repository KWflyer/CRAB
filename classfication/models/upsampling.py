import tensorflow as tf
from keras.layers import Layer


class UpSamplingBilinear_me(Layer):
    def __init__(self, targetsize=32):
        super(UpSamplingBilinear_me, self).__init__()
        self.targetsize = int(targetsize)

    def call(self, inputs):
        x = tf.image.resize_images(inputs, [self.targetsize, self.targetsize])
        return x
    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        assert isinstance(input_shape, list)
        shape_a = input_shape[0]
        shape_c = input_shape[-1]
        output_shape = [shape_a, self.targetsize, self.targetsize, shape_c]
        return tuple(output_shape)

class UpSamplingBilinear(Layer):
    def __init__(self, scale=4):
        self.scale = scale
        super(UpSamplingBilinear, self).__init__()

    def build(self, input_shape):
        self.shape = input_shape
        super(UpSamplingBilinear, self).build(input_shape)

    def call(self, inputs):
        new_size = self.compute_output_shape(self.shape)[-2:]
        x = K.permute_dimensions(inputs, [0, 2, 3, 1])
        x = tf.image.resize_images(x, new_size)
        x = K.permute_dimensions(x, [0, 3, 1, 2])
        return x

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[2] *= self.scale
        input_shape[3] *= self.scale
        return tuple(input_shape)


from keras import backend as K
from keras.layers import Layer
from keras.utils import conv_utils
from keras.layers import InputSpec

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (int(inputs.shape[1] * self.upsampling[0]),
                                                       int(inputs.shape[2] * self.upsampling[1])),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
