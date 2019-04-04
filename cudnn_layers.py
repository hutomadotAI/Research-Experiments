import tensorflow as tf

from layers import dropout, _last_relevant


class GRU:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0,
                 is_train=None, seed=3435, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else num_units
            gru = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units, seed=seed)
            init = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            mask = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                           keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append(gru)
            self.inits.append(init)
            self.dropout_mask.append(mask)

    def __call__(self, inputs, seq_len, concat_layers=True, use_last=False, scope='gru'):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru = self.grus[layer]
            init = self.inits[layer]
            mask = self.dropout_mask[layer]
            with tf.variable_scope("{}_{}".format(scope, layer)):
                out, _ = gru(outputs[-1] * mask, initial_state=(init, ))
            outputs.append(out)
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])

        if use_last:
            last = _last_relevant(res, seq_len)
            return last, res
        else:
            return res


class BiGRU:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0,
                 is_train=None, seed=3435, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units, seed=seed)
            gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units, seed=seed)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, concat_layers=True, use_last=False, scope='gru'):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]
            with tf.variable_scope("fw_{}_{}".format(scope, layer)):
                out_fw, _ = gru_fw(
                    outputs[-1] * mask_fw, initial_state=(init_fw, ))
            with tf.variable_scope("bw_{}_{}".format(scope, layer)):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_axis=0, batch_axis=1)
                out_bw, _ = gru_bw(inputs_bw, initial_state=(init_bw, ))
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_axis=0, batch_axis=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])

        if use_last:
            last = _last_relevant(res, seq_len)
            return last, res
        else:
            return res


class LSTM:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0,
                 is_train=None, seed=3435, scope=None):
        self.num_layers = num_layers
        self.lstms = []
        self.inits = []
        self.inits2 = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else num_units
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(1, num_units, seed=seed)
            init = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            init2 = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            mask = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                           keep_prob=keep_prob, is_train=is_train, mode=None)
            self.lstms.append(lstm)
            self.inits.append(init)
            self.inits2.append(init2)
            self.dropout_mask.append(mask)

    def __call__(self, inputs, seq_len, concat_layers=True, use_last=False, scope='lstm'):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            lstm = self.lstms[layer]
            init = self.inits[layer]
            init2 = self.inits2[layer]
            mask = self.dropout_mask[layer]
            with tf.variable_scope("{}_{}".format(scope, layer)):
                out, _ = lstm(outputs[-1] * mask, initial_state=(init, init2))
            outputs.append(out)
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])

        if use_last:
            last = _last_relevant(res, seq_len)
            return last, res
        else:
            return res


class BiLSTM:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0,
                 is_train=None, seed=3435, scope=None):
        self.num_layers = num_layers
        self.lstms = []
        self.inits = []
        self.inits2 = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            lstm_fw = tf.contrib.cudnn_rnn.CudnnLSTM(1, num_units, seed=seed)
            lstm_bw = tf.contrib.cudnn_rnn.CudnnLSTM(1, num_units, seed=seed)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            init_fw2 = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            init_bw2 = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.lstms.append((lstm_fw, lstm_bw,))
            self.inits.append((init_fw, init_bw,))
            self.inits2.append((init_fw2, init_bw2))
            self.dropout_mask.append((mask_fw, mask_bw,))

    def __call__(self, inputs, seq_len, concat_layers=True, use_last=False, scope='lstm'):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            lstm_fw, lstm_bw = self.lstms[layer]
            init_fw, init_bw = self.inits[layer]
            init_fw2, init_bw2 = self.inits2[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]

            with tf.variable_scope("fw_{}_{}".format(scope, layer)):
                out_fw, _ = lstm_fw(
                    outputs[-1] * mask_fw, initial_state=(init_fw, init_fw2))
            with tf.variable_scope("bw_{}_{}".format(scope, layer)):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_axis=0, batch_axis=1)
                out_bw, _ = lstm_bw(inputs_bw, initial_state=(init_bw, init_bw2))
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_axis=0, batch_axis=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])

        if use_last:
            last = _last_relevant(res, seq_len)
            return last, res
        else:
            return res
