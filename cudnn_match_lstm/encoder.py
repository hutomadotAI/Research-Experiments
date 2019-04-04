import tensorflow as tf

from cudnn_layers import BiLSTM


class Encoder(object):
    def __init__(self, batch_size, hidden_units, keep_prob, n_layers=3, seed=3435):
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.seed = seed
        self.keep_prob = keep_prob

    def setup_placeholders(self):
        pass

    def encode(self, inputs, masks, is_train):
        context, question = inputs
        context_mask, question_mask = masks

        with tf.variable_scope("encode_context"):
            # outshape: [batch_size, 2 * rnn_hidden_units]
            lstm = BiLSTM(num_units=self.hidden_units, num_layers=self.n_layers, batch_size=self.batch_size,
                          input_size=context.get_shape()[-1].value, keep_prob=self.keep_prob,
                          is_train=is_train, seed=self.seed)
            lstm_pool_context, lstm_out_context = lstm(context, seq_len=context_mask,
                                                       concat_layers=False, use_last=True)

        with tf.variable_scope('encode_question'):
            lstm = BiLSTM(num_units=self.hidden_units, num_layers=self.n_layers, batch_size=self.batch_size,
                          input_size=context.get_shape()[-1].value, keep_prob=self.keep_prob,
                          is_train=is_train, seed=self.seed)
            lstm_pool_question, lstm_out_question = lstm(question, seq_len=question_mask,
                                                         concat_layers=False, use_last=True)

        return [lstm_out_context, lstm_pool_context], [lstm_out_question, lstm_pool_question]
