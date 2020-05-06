import tensorflow as tf

from stable_baselines.a2c.utils import conv, linear, conv_to_fc


def tic_tac_toe_cnn(scaled_images, **kwargs):
    """
    Custom CNN for Tic Tac Toe env.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer = scaled_images

    # print(kwargs)
    net_arch = kwargs['cnn_arch']

    for i, f in enumerate(net_arch[:-1], start=1):
        # print('c' + str(i), f)
        layer = activ(conv(layer, 'c' + str(i), n_filters=f, filter_size=3,
                           stride=1, pad='SAME', data_format='NCHW'))

    layer = conv_to_fc(layer)

    # print('fc1', net_arch[-1])
    # print()
    return activ(linear(layer, 'fc1', n_hidden=net_arch[-1]))
