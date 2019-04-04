import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops import array_ops


"""
some layers are taken and modified 
from https://github.com/HKUST-KnowComp/R-Net/blob/master/func.py
"""


def _last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    output_size = tf.shape(output)[2]
    index = tf.range(0, batch_size) * max_length + tf.maximum((length - 1), 0)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant


# -- A helper function to reverse a tensor along seq_dim
def _reverse(input_, seq_lengths, seq_dim, batch_dim):
    if seq_lengths is not None:
        return array_ops.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_dim=seq_dim, batch_dim=batch_dim)
    else:
        return array_ops.reverse(input_, axis=[seq_dim])


def mask_score(score, seq_len, mask_value="-1e30"):
    score_mask = tf.sequence_mask(
        seq_len, maxlen=tf.shape(score)[1])
    score_mask_values = float(mask_value) * tf.ones_like(score)
    return array_ops.where(score_mask, score, score_mask_values)


def dropout(args, keep_prob, is_train, mode="recurrent"):
    # if keep_prob < 1.0:
    noise_shape = None
    scale = 1.0
    shape = tf.shape(args)
    if mode == "embedding":
        noise_shape = [shape[0], 1]
        scale = keep_prob
    elif mode == "recurrent" and len(args.get_shape().as_list()) == 3:
        noise_shape = [shape[0], 1, shape[-1]]
    else:
        noise_shape = None
    args = tf.cond(is_train, lambda: tf.nn.dropout(
        args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args


def Dense(x, n_hidden, use_bias=True, reuse=False, seed=3435, scope='dense'):
    with tf.variable_scope(scope, reuse=reuse):
        shape = tf.shape(x)
        last_dim = x.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(x.get_shape().as_list()) - 1)] + [n_hidden]
        flat_x = tf.reshape(x, [-1, last_dim])
        W = tf.get_variable("weights", [last_dim, n_hidden],
                            initializer=tf.glorot_uniform_initializer(seed=seed))
        res = tf.matmul(flat_x, W)
        if use_bias:
            b = tf.get_variable(
                "b", [n_hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res


def LSTM(x, seq_len, n_hidden,
         input_dropout_keep_prob, output_dropout_keep_prob, state_dropout_keep_prob,
         n_layers=1, reuse=False, use_last=True, residual=False, seed=3435, scope='lstm'):
    # Define lstm cell with tensorflow
    with tf.name_scope(scope), tf.variable_scope(scope, reuse=reuse):
        layers = []
        for i in range(n_layers):
            cell = tf.contrib.rnn.LSTMCell(
                n_hidden,
                forget_bias=1.0,
                initializer=tf.initializers.variance_scaling(distribution='uniform', seed=seed),
                state_is_tuple=True,
                reuse=reuse
            )
            cell_drop = tf.contrib.rnn.DropoutWrapper(
                cell,
                input_keep_prob=input_dropout_keep_prob,
                output_keep_prob=output_dropout_keep_prob,
                state_keep_prob=state_dropout_keep_prob,
                variational_recurrent=True,
                dtype=tf.float32,
                input_size=x.get_shape()[2].value if i == 0 else [1, 2*n_hidden],
                seed=seed
            )

            if residual and i > 0:
                cell_drop = tf.contrib.rnn.ResidualWrapper(cell_drop)

            layers.append(cell_drop)

        lstm_drop_multi = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=True)
        rnn_outputs, _ = tf.nn.dynamic_rnn(lstm_drop_multi,
                                           x,
                                           dtype=tf.float32,
                                           sequence_length=seq_len)
    if use_last:
        return tf.identity(_last_relevant(rnn_outputs, seq_len), name='lstm_out'), rnn_outputs
    else:
        return tf.identity(rnn_outputs, name='lstm_out')


def LayerNormLSTM(x, seq_len, n_hidden, dropout_keep_prob,
                  n_layers=1, reuse=False, use_last=True, residual=False, seed=3435, scope='layer_norm_lstm'):
    # Define lstm cell with tensorflow
    with tf.name_scope(scope), tf.variable_scope(scope, reuse=reuse):
        layers = []
        for i in range(n_layers):
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                n_hidden,
                forget_bias=1.0,
                initializer=tf.initializers.variance_scaling(distribution='uniform', seed=seed),
                state_is_tuple=True,
                dropout_keep_prob=dropout_keep_prob,
                dropout_prob_seed=seed,
                reuse=reuse
            )

            if residual and i > 0:
                cell = tf.contrib.rnn.ResidualWrapper(cell)

            layers.append(cell)

        lstm_drop_multi = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=True)
        rnn_outputs, _ = tf.nn.dynamic_rnn(lstm_drop_multi,
                                           x,
                                           dtype=tf.float32,
                                           sequence_length=seq_len)
    if use_last:
        return tf.identity(_last_relevant(rnn_outputs, seq_len), name='lstm_out'), rnn_outputs
    else:
        return tf.identity(rnn_outputs, name='lstm_out')


def GRU(x, seq_len, n_hidden,
        input_dropout_keep_prob, output_dropout_keep_prob, state_dropout_keep_prob,
        n_layers=1, reuse=False, use_last=True, residual=False, seed=3435, scope='gru'):
    # Define lstm cell with tensorflow
    with tf.name_scope(scope), tf.variable_scope(scope, reuse=reuse):
        layers = []
        for i in range(n_layers):
            cell = tf.contrib.rnn.GRUCell(
                n_hidden,
                # bias_initializer=1.0, # 1.0 is default anyway (tf1.8)
                kernel_initializer=tf.initializers.variance_scaling(distribution='uniform', seed=seed),
                reuse=reuse
            )
            cell_drop = tf.contrib.rnn.DropoutWrapper(
                cell,
                input_keep_prob=input_dropout_keep_prob,
                output_keep_prob=output_dropout_keep_prob,
                state_keep_prob=state_dropout_keep_prob,
                variational_recurrent=True,
                dtype=tf.float32,
                input_size=x.get_shape()[2].value if i == 0 else [1, 2*n_hidden],
                seed=seed
            )

            if residual and i > 0:
                cell_drop = tf.contrib.rnn.ResidualWrapper(cell_drop)

            layers.append(cell_drop)

        gru_drop_multi = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=True)

        rnn_outputs, _ = tf.nn.dynamic_rnn(gru_drop_multi,
                                           x,
                                           dtype=tf.float32,
                                           sequence_length=seq_len)
    if use_last:
        output = tf.identity(_last_relevant(rnn_outputs, seq_len), name='gru_out')
        return output, rnn_outputs
    else:
        return tf.identity(rnn_outputs, name='gru_out')


def BiGRU(x, seq_len, n_hidden,
          input_dropout_keep_prob, output_dropout_keep_prob, state_dropout_keep_prob,
          n_layers=1, reuse=False, use_last=True, residual=False, seed=3435, scope='gru'):
    # Define lstm cell with tensorflow
    with tf.variable_scope(scope+'_fw'):
        layers = []
        for i in range(n_layers):
            cell = tf.contrib.rnn.GRUCell(
                n_hidden,
                # bias_initializer=1.0, # 1.0 is default anyway (tf1.8)
                kernel_initializer=tf.initializers.variance_scaling(distribution='uniform', seed=seed),
                reuse=reuse
            )
            # cell_drop = tf.contrib.rnn.DropoutWrapper(
            #     cell,
            #     input_keep_prob=input_dropout_keep_prob,
            #     output_keep_prob=output_dropout_keep_prob,
            #     state_keep_prob=state_dropout_keep_prob,
            #     variational_recurrent=True,
            #     dtype=tf.float32,
            #     input_size=x.get_shape()[2].value if i == 0 else [1, 2*n_hidden],
            #     seed=seed
            # )
            #
            # if residual and i > 0:
            #     cell_drop = tf.contrib.rnn.ResidualWrapper(cell_drop)

            layers.append(cell)

        gru_drop_multi_fw = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=True)

    with tf.variable_scope(scope+'_bw'):
        layers = []
        for i in range(n_layers):
            cell = tf.contrib.rnn.GRUCell(
                n_hidden,
                # bias_initializer=1.0, # 1.0 is default anyway (tf1.8)
                kernel_initializer=tf.initializers.variance_scaling(distribution='uniform', seed=seed),
                reuse=reuse
            )
            # cell_drop = tf.contrib.rnn.DropoutWrapper(
            #     cell,
            #     input_keep_prob=input_dropout_keep_prob,
            #     output_keep_prob=output_dropout_keep_prob,
            #     state_keep_prob=state_dropout_keep_prob,
            #     variational_recurrent=True,
            #     dtype=tf.float32,
            #     input_size=x.get_shape()[2].value if i == 0 else [1, 2*n_hidden],
            #     seed=seed
            # )
            #
            # if residual and i > 0:
            #     cell_drop = tf.contrib.rnn.ResidualWrapper(cell_drop)

            layers.append(cell)

        gru_drop_multi_bw = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=True)

    with tf.name_scope(scope), tf.variable_scope(scope, reuse=reuse):
        rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(gru_drop_multi_fw,
                                                         gru_drop_multi_bw,
                                                         x,
                                                         dtype=tf.float32,
                                                         sequence_length=seq_len)
    if use_last:
        lastfw = _last_relevant(rnn_outputs[0], seq_len)
        lastbw = _last_relevant(rnn_outputs[1], seq_len)
        last = tf.concat([lastfw, lastbw], 1, name='bigru_out')
        last = tf.reshape(last, [-1, 2*n_hidden])
        return last, rnn_outputs
    else:
        return tf.identity(rnn_outputs, name='bigru_out')


def BiLSTM(x, seq_len, n_hidden,
           input_dropout_keep_prob, output_dropout_keep_prob, state_dropout_keep_prob,
           n_layers=1, reuse=False, use_last=True, residual=False, seed=3435):
    # Define lstm cell with tensorflow
    with tf.name_scope("fw"), tf.variable_scope("fw", reuse=reuse):
        layers = []
        for i in range(n_layers):
            cell = tf.contrib.rnn.LSTMCell(
                n_hidden,
                forget_bias=1.0,
                initializer=tf.initializers.variance_scaling(distribution='uniform', seed=seed),
                state_is_tuple=True,
                reuse=reuse
            )
            cell_drop = tf.contrib.rnn.DropoutWrapper(
                cell,
                input_keep_prob=input_dropout_keep_prob,
                output_keep_prob=output_dropout_keep_prob,
                state_keep_prob=state_dropout_keep_prob,
                variational_recurrent=True,
                dtype=tf.float32,
                input_size=x.get_shape()[2].value if i == 0 else n_hidden,
                seed=seed
            )

            if residual and i > 0:
                cell_drop = tf.contrib.rnn.ResidualWrapper(cell_drop)

            layers.append(cell_drop)

        lstm_drop_multi_fw = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=True)

    with tf.name_scope("bw"), tf.variable_scope("bw", reuse=reuse):
        layers = []
        for i in range(n_layers):
            cell = tf.contrib.rnn.LSTMCell(
                n_hidden,
                forget_bias=1.0,
                initializer=tf.initializers.variance_scaling(distribution='uniform', seed=seed),
                state_is_tuple=True,
                reuse=reuse
            )
            cell_drop = tf.contrib.rnn.DropoutWrapper(
                cell,
                input_keep_prob=input_dropout_keep_prob,
                output_keep_prob=output_dropout_keep_prob,
                state_keep_prob=state_dropout_keep_prob,
                variational_recurrent=True,
                dtype=tf.float32,
                input_size=x.get_shape()[2].value if i == 0 else n_hidden,
                seed=seed
            )

            if residual and i > 0:
                cell_drop = tf.contrib.rnn.ResidualWrapper(cell_drop)

            layers.append(cell_drop)

        lstm_drop_multi_bw = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=True)

    with tf.name_scope("lstm"), tf.variable_scope("lstm", reuse=reuse):
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_drop_multi_fw,
                                                     lstm_drop_multi_bw,
                                                     x,
                                                     dtype=tf.float32,
                                                     sequence_length=seq_len)

    if use_last:
        lastfw = _last_relevant(outputs[0], seq_len)
        lastbw = _last_relevant(outputs[1], seq_len)
        last = tf.concat([lastfw, lastbw], 1, name='bilstm_out')
        last = tf.reshape(last, [-1, 2*n_hidden])
        return last, outputs
    else:
        outputs = tf.identity(outputs, name='bilstm_out')
        return outputs


def LayerNormBiLSTM(x, seq_len, n_hidden, dropout_keep_prob,
                    n_layers=1, reuse=False, use_last=True, residual=False, seed=3435, scope='layer_norm_lstm'):
    # Define lstm cell with tensorflow
    with tf.name_scope("fw"), tf.variable_scope("fw", reuse=reuse):
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            n_hidden,
            forget_bias=1.0,
            # initializer=tf.initializers.variance_scaling(distribution='uniform', seed=seed),
            # state_is_tuple=True,
            dropout_keep_prob=dropout_keep_prob,
            dropout_prob_seed=seed,
            reuse=reuse
        )

        if residual:
            cell = tf.contrib.rnn.ResidualWrapper(cell)

        lstm_drop_multi_fw = tf.contrib.rnn.MultiRNNCell([cell for _ in range(n_layers)], state_is_tuple=True)

    with tf.name_scope("bw"), tf.variable_scope("bw", reuse=reuse):
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            n_hidden,
            forget_bias=1.0,
            # initializer=tf.initializers.variance_scaling(distribution='uniform', seed=seed),
            # state_is_tuple=True,
            dropout_keep_prob=dropout_keep_prob,
            dropout_prob_seed=seed,
            reuse=reuse
        )

        if residual:
            cell = tf.contrib.rnn.ResidualWrapper(cell)

        lstm_drop_multi_bw = tf.contrib.rnn.MultiRNNCell([cell for _ in range(n_layers)], state_is_tuple=True)

    with tf.name_scope(scope), tf.variable_scope(scope, reuse=reuse):
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_drop_multi_fw,
                                                     lstm_drop_multi_bw,
                                                     x,
                                                     dtype=tf.float32,
                                                     sequence_length=seq_len)

    if use_last:
        lastfw = _last_relevant(outputs[0], seq_len)
        lastbw = _last_relevant(outputs[1], seq_len)
        last = tf.concat([lastfw, lastbw], 1, name='bilstm_out')
        return last, outputs
    else:
        outputs = tf.identity(outputs, name='bilstm_out')
        return outputs


def attention_pooling(memory, hidden, memory_len, dropout_keep_prob, is_train, scope="att_pool", seed=3435):
    '''
    initial hidden vector for pointer net using attention pooling over
    question representation; as used in r-net
    r-net uses additive attention - here we use multiplicative attention!!
    '''
    with tf.variable_scope(scope):
        memory_drop = dropout(memory, dropout_keep_prob, is_train)
        s0 = tf.nn.tanh(Dense(memory_drop, hidden, scope='s0', seed=seed))
        s = Dense(s0, 1, use_bias=False, scope='s', seed=seed)
        s1 = mask_score(tf.squeeze(s, [2]), memory_len, float("-inf"))
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * memory, axis=1)
    return res


def pointer_net_qa(input, init, input_len, hidden_units, dropout_keep_prob, is_train):
    def pointer(inputs, state, hidden, input_len, scope="pointer", seed=3435, reuse=False):
        '''
        compute logit for each time-step of inputs (context) in relation to
        state (question) using attention mechanism (see pointer net)
        r-net uses additive attention - here we use multiplicative attention!!
        '''
        with tf.variable_scope(scope):
            u = tf.concat([tf.tile(tf.expand_dims(state, axis=1), [
                1, tf.shape(inputs)[1], 1]), inputs], axis=2)
            s0 = tf.nn.tanh(Dense(u, hidden, use_bias=False, scope='s0', reuse=reuse, seed=seed))
            s = Dense(s0, 1, use_bias=False, scope='s', reuse=reuse, seed=seed)
            s1 = mask_score(tf.squeeze(s, [2]), input_len, float("-10e10"))
            a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
            res = tf.reduce_sum(a * inputs, axis=1)
        return res, s1

    init_drop = dropout(init, dropout_keep_prob, is_train)
    input_drop = dropout(input, dropout_keep_prob, is_train)
    inp, logits1 = pointer(input_drop, init_drop, hidden_units, input_len)
    inp_drop = dropout(inp, dropout_keep_prob, is_train)
    gru = GRUCell(hidden_units)
    _, state = gru(inp_drop, init)
    state_drop = dropout(state, dropout_keep_prob, is_train)
    _, logits2 = pointer(input, state_drop, hidden_units, input_len, reuse=True)

    return [logits1, logits2]


def scaled_dot_product_attention(inputs, memory, memory_len, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
    with tf.variable_scope(scope):
        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]

        with tf.variable_scope("attention"):
            inputs_ = tf.nn.relu(
                Dense(d_inputs, hidden, use_bias=False, scope="inputs"))
            memory_ = tf.nn.relu(
                Dense(d_memory, hidden, use_bias=False, scope="memory"))
            outputs = tf.matmul(inputs_, tf.transpose(
                memory_, [0, 2, 1])) / (hidden ** 0.5)

            score_mask = array_ops.sequence_mask(
                memory_len, maxlen=array_ops.shape(outputs)[2])
            score_mask = tf.tile(tf.expand_dims(score_mask, axis=1), [1, JX, 1])
            score_mask_values = float('-inf') * array_ops.ones_like(outputs)
            masked_outputs = array_ops.where(score_mask, outputs, score_mask_values)

            logits = tf.nn.softmax(masked_outputs)
            outputs = tf.matmul(logits, memory)
            res = tf.concat([inputs, outputs], axis=2)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(Dense(d_res, dim, use_bias=False))
            return res * gate