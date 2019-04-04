import math
import tensorflow as tf
from tensorflow.python.ops import nn_ops

from layers import dropout

"""
layers copied from https://github.com/NLPLearn/QANet/blob/master/layers.py
"""

initializer = lambda: tf.contrib.layers.variance_scaling_initializer(
    factor=1.0,
    mode='FAN_AVG',
    uniform=True,
    dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(
    factor=2.0,
    mode='FAN_IN',
    uniform=False,
    dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
    x: a Tensor with shape [..., m]
    n: an integer.
    Returns:
    a Tensor with shape [..., n, m/n]
    """
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return tf.transpose(ret, [0, 2, 1, 3])


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    Args:
    x: a Tensor with shape [..., a, b]
    Returns:
    a Tensor with shape [..., ab]
    """
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor the same shape as x.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def layer_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
            "layer_norm_scale", [filters],
            regularizer=regularizer, initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "layer_norm_bias", [filters],
            regularizer=regularizer, initializer=tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result


def layer_dropout(inputs, residual, dropout):
    pred = tf.random_uniform([]) < dropout
    return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)


def residual_block(inputs, num_blocks, num_conv_layers, kernel_size, mask=None,
                   num_filters=128, input_projection=False, num_heads=8,
                   scope="res_block", is_train=True, reuse=None, bias=True, keep_prob=1.0):
    with tf.variable_scope(scope, reuse=reuse):
        if input_projection:
            inputs = conv(inputs, num_filters, name="input_projection", reuse=reuse)
        outputs = inputs
        sublayer = 1
        total_sublayers = (num_conv_layers + 2) * num_blocks
        for i in range(num_blocks):
            outputs = add_timing_signal_1d(outputs)
            outputs, sublayer = conv_block(outputs, num_conv_layers, kernel_size, num_filters,
                                           scope="encoder_block_%d"%i, reuse=reuse, is_train=is_train,
                                           keep_prob=keep_prob, sublayers=(sublayer, total_sublayers))
            outputs, sublayer = self_attention_block(outputs, num_filters, mask=mask, num_heads=num_heads,
                                                     scope="self_attention_layers%d"%i, reuse=reuse, is_train=is_train,
                                                     bias=bias, keep_prob=keep_prob, sublayers=(sublayer, total_sublayers))
        return outputs


def conv_block(inputs, num_conv_layers, kernel_size, num_filters,
               scope="conv_block", is_train=True, input_projection=False,
               reuse=None, keep_prob=1.0, sublayers=(1, 1), norm_fn=layer_norm):
    with tf.variable_scope(scope, reuse=reuse):
        if input_projection:
            inputs = conv(inputs, num_filters, name="input_projection", reuse=reuse)
        outputs = tf.expand_dims(inputs, 2)
        l, L = sublayers
        for i in range(num_conv_layers):
            residual = outputs
            outputs = norm_fn(outputs, scope="layer_norm_%d" % i, reuse=reuse)
            if i % 2 == 0:
                outputs = dropout(outputs, keep_prob, is_train=is_train, mode='')
            outputs = depthwise_separable_convolution(
                outputs,
                kernel_size=(kernel_size, 1), num_filters=num_filters,
                scope="depthwise_conv_layers_%d" % i, reuse=reuse, is_train=is_train)
            outputs = layer_dropout(outputs, residual, (1. - keep_prob) * float(l) / L)
            l += 1
        return tf.squeeze(outputs, 2), l


def self_attention_block(inputs, num_filters, mask=None, num_heads=8,
                         scope="self_attention_ffn", reuse=None, is_train=True,
                         bias=True, keep_prob=1.0, sublayers=(1, 1), norm_fn=layer_norm):
    with tf.variable_scope(scope, reuse=reuse):
        l, L = sublayers
        # Self attention
        outputs = norm_fn(inputs, scope="layer_norm_1", reuse=reuse)
        outputs = dropout(outputs, keep_prob, is_train, mode='')
        outputs = multihead_attention(outputs, num_filters,
                                      num_heads=num_heads, reuse=reuse,
                                      mask=mask, is_train=is_train, bias=bias, keep_prob=keep_prob)
        residual = layer_dropout(outputs, inputs, (1. - keep_prob) * float(l) / L)
        l += 1
        # Feed-forward
        outputs = norm_fn(residual, scope="layer_norm_2", reuse=reuse)
        outputs = dropout(outputs, keep_prob, is_train, mode='')
        outputs = conv(outputs, num_filters, True, tf.nn.relu, name="FFN_1", reuse=reuse)
        outputs = conv(outputs, num_filters, True, None, name="FFN_2", reuse=reuse)
        outputs = layer_dropout(outputs, residual, (1. - keep_prob) * float(l) / L)
        l += 1
        return outputs, l


def multihead_attention(queries, units, num_heads,
                        memory=None,
                        scope="Multi_Head_Attention",
                        reuse=None,
                        mask=None,
                        seq_len_q=None,
                        seq_len_m=None,
                        is_train=True,
                        bias=True,
                        keep_prob=1.0):
    with tf.variable_scope(scope, reuse=reuse):
        # Self attention
        if memory is None:
            memory = queries
            seq_len_m = seq_len_q

        memory = conv(memory, 2 * units, name="memory_projection", reuse=reuse)
        query = conv(queries, units, name="query_projection", reuse=reuse)
        Q = split_last_dimension(query, num_heads)
        K, V = [split_last_dimension(tensor, num_heads) for tensor in tf.split(memory, 2, axis=2)]

        # scaling
        key_depth_per_head = units // num_heads
        Q *= key_depth_per_head ** -0.5
        x = dot_product_attention(Q, K, V,
                                  bias=bias,
                                  mask=mask,
                                  is_train=is_train,
                                  scope="dot_product_attention",
                                  reuse=reuse,
                                  keep_prob=keep_prob)
        return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          mask=None,
                          is_train=True,
                          scope=None,
                          reuse=None,
                          keep_prob=1.0):
    """dot-product attention.
    Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    is_training: a bool of training
    scope: an optional string
    Returns:
    A Tensor.
    """
    with tf.variable_scope(scope, default_name="dot_product_attention", reuse=reuse):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        if bias:
            b = tf.get_variable("bias",
                                logits.get_shape()[-1],
                                regularizer=regularizer,
                                initializer=tf.zeros_initializer())
            logits += b
        if mask is not None:
            shapes = [x if x != None else -1 for x in logits.shape.as_list()]
            mask = tf.reshape(mask, [shapes[0], 1, 1, shapes[-1]])
            # mask = tf.reshape(mask, [tf.shape(logits)[0], 1, 1, tf.shape(logits)[-1]])
            logits = mask_logits(logits, mask)
        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        weights = dropout(weights, keep_prob, is_train, mode='')
        return tf.matmul(weights, v)


def conv(inputs, output_size, bias=None, activation=None, kernel_size=1, name="conv", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1, kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, 1, output_size]
            strides = [1, 1, 1, 1]
        else:
            filter_shape = [kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable(
            "kernel_",
            filter_shape,
            dtype=tf.float32,
            regularizer=regularizer,
            initializer=initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable(
                "bias_",
                bias_shape,
                regularizer=regularizer,
                initializer=tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def depthwise_separable_convolution(inputs, kernel_size, num_filters,
                                    scope="depthwise_separable_convolution",
                                    bias=True, is_train=True, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        shapes = inputs.shape.as_list()
        depthwise_filter = tf.get_variable("depthwise_filter",
                                           (kernel_size[0], kernel_size[1], shapes[-1], 1),
                                           dtype=tf.float32,
                                           regularizer=regularizer,
                                           initializer=initializer_relu())
        pointwise_filter = tf.get_variable("pointwise_filter",
                                           (1, 1, shapes[-1], num_filters),
                                           dtype=tf.float32,
                                           regularizer=regularizer,
                                           initializer=initializer_relu())
        outputs = tf.nn.separable_conv2d(inputs,
                                         depthwise_filter,
                                         pointwise_filter,
                                         strides=(1, 1, 1, 1),
                                         padding="SAME")
        if bias:
            b = tf.get_variable("bias",
                                outputs.shape[-1],
                                regularizer=regularizer,
                                initializer=tf.zeros_initializer())
            outputs += b
        outputs = tf.nn.relu(outputs)
        return outputs


def highway(x, size=None, activation=None,
            num_layers=2, scope="highway", keep_prob=1.0, reuse=None, is_train=False):
    with tf.variable_scope(scope, reuse):
        if size is None:
            size = x.shape.as_list()[-1]
        else:
            x = conv(x, size, name="input_projection", reuse=reuse)
        for i in range(num_layers):
            T = conv(x, size, bias=True, activation=tf.sigmoid,
                     name="gate_%d"%i, reuse=reuse)
            H = conv(x, size, bias=True, activation=activation,
                     name="activation_%d"%i, reuse=reuse)
            H = dropout(H, keep_prob=keep_prob, is_train=is_train, mode='')
            x = H * T + x * (1.0 - T)
        return x


def ndim(x):
    """Copied from keras==2.0.6
    Returns the number of axes in a tensor, as an integer.
    # Arguments
        x: Tensor or variable.
    # Returns
        Integer (scalar), number of axes.
    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.ndim(inputs)
        3
        >>> K.ndim(kvar)
        2
    ```
    """
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None


def dot(x, y):
    """Modified from keras==2.0.6
    Multiplies 2 tensors (and/or variables) and returns a *tensor*.
    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)
    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.
    # Returns
        A tensor, dot product of `x` and `y`.
    """
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(x.get_shape().as_list(), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(y.get_shape().as_list(), tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if isinstance(x, tf.SparseTensor):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out


def batch_dot(x, y, axes=None):
    """Copy from keras==2.0.6
    Batchwise dot product.
    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.
    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.
    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = tf.expand_dims(out, 1)
    return out


def optimized_trilinear_for_attention(args, c_maxlen, q_maxlen, is_train,
        input_keep_prob=1.0,
        scope='efficient_trilinear',
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=initializer()):
    assert len(args) == 2, "just use for computing attention with two input"
    arg0_shape = args[0].get_shape().as_list()
    arg1_shape = args[1].get_shape().as_list()
    if len(arg0_shape) != 3 or len(arg1_shape) != 3:
        raise ValueError("`args` must be 3 dims (batch_size, len, dimension)")
    if arg0_shape[2] != arg1_shape[2]:
        raise ValueError("the last dimension of `args` must equal")
    arg_size = arg0_shape[2]
    dtype = args[0].dtype
    dropped_args = [dropout(arg, input_keep_prob, is_train, mode='') for arg in args]
    with tf.variable_scope(scope):
        weights4arg0 = tf.get_variable(
            "linear_kernel4arg0", [arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        weights4arg1 = tf.get_variable(
            "linear_kernel4arg1", [arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        weights4mlu = tf.get_variable(
            "linear_kernel4mul", [1, 1, arg_size],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        biases = tf.get_variable(
            "linear_bias", [1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=bias_initializer)
        # shape: [batch_size, c_maxlen, q_maxlen]
        subres0 = tf.tile(dot(dropped_args[0], weights4arg0), [1, 1, q_maxlen])
        # shape: [batch_size, c_maxlen, q_maxlen]
        subres1 = tf.tile(tf.transpose(dot(dropped_args[1], weights4arg1), perm=(0, 2, 1)), [1, c_maxlen, 1])
        # shape: [batch_size, c_maxlen, q_maxlen]
        subres2 = batch_dot(dropped_args[0] * weights4mlu, tf.transpose(dropped_args[1], perm=(0, 2, 1)))
        res = subres0 + subres1 + subres2
        biases = tf.tile(biases, [tf.shape(res)[-1]])
        nn_ops.bias_add(res, biases)
        return res