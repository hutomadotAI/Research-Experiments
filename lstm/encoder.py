import tensorflow as tf

from layers import BiLSTM


class Encoder(object):
    def __init__(self, hidden_units, output_keep_prob, input_keep_prob,
                 state_keep_prob, n_layers=1, seed=3435):
        self.hidden_units = hidden_units
        self.seed = seed
        self.output_keep_prob = output_keep_prob
        self.input_keep_prob = input_keep_prob
        self.state_keep_prob = state_keep_prob
        self.n_layers = n_layers

    def setup_placeholders(self):
        pass

    def encode(self, inputs, masks, is_train):
        context, question = inputs
        context_mask, question_mask = masks

        with tf.variable_scope("encode_context"):
            # outshape: [batch_size, 2 * rnn_hidden_units]
            lstm_pool_context, lstm_out_context = BiLSTM(
                context,
                context_mask,
                self.hidden_units,
                tf.cond(is_train, lambda: self.output_keep_prob, lambda: 1.0),
                tf.cond(is_train, lambda: self.input_keep_prob, lambda: 1.0),
                tf.cond(is_train, lambda: self.state_keep_prob, lambda: 1.0),
                n_layers=self.n_layers,
                residual=True,
                use_last=True,
                seed=self.seed,
                reuse=False)
            lstm_out_context = tf.concat([lstm_out_context[0], lstm_out_context[1]],
                                         2, name='lstm_out_context')

        with tf.variable_scope('encode_question'):
            lstm_pool_question, lstm_out_question = BiLSTM(
                question,
                question_mask,
                self.hidden_units,
                tf.cond(is_train, lambda: self.output_keep_prob, lambda: 1.0),
                tf.cond(is_train, lambda: self.input_keep_prob, lambda: 1.0),
                tf.cond(is_train, lambda: self.state_keep_prob, lambda: 1.0),
                n_layers=self.n_layers,
                residual=True,
                use_last=True,
                seed=self.seed,
                reuse=False)
            lstm_out_question = tf.concat([lstm_out_question[0], lstm_out_question[1]],
                                          2, name='lstm_out_question')

        return [lstm_out_context, lstm_pool_context], [lstm_out_question, lstm_pool_question]