import tensorflow as tf

from layers import _reverse
from layers import BiLSTM, LSTM, attention_pooling, pointer_net_qa


class Decoder(object):
    def __init__(self, hidden_units, output_keep_prob, input_keep_prob,
                 state_keep_prob, keep_prob, seed=3435):
        self.seed=seed
        self.hidden_units = hidden_units
        self.output_keep_prob = output_keep_prob
        self.input_keep_prob = input_keep_prob
        self.state_keep_prob = state_keep_prob
        self.keep_prob = keep_prob

    def setup_placeholders(self):
        pass

    def run_lstm(self, context_out, question_pool, context_len, is_train):
        # tile pooled question rep and concat with context
        q_rep = tf.expand_dims(question_pool, 1) # (batch_size, 1, D)
        encoded_passage_shape = tf.shape(context_out)[1]
        q_rep = tf.tile(q_rep, [1, encoded_passage_shape, 1])

        q_c_rep = tf.concat([context_out, q_rep], axis=-1)

        with tf.variable_scope('lstm_') as scope:
            lstm_out = BiLSTM(q_c_rep,
                              context_len,
                              self.hidden_units,
                              tf.cond(is_train, lambda: self.output_keep_prob, lambda: 1.0),
                              tf.cond(is_train, lambda: self.input_keep_prob, lambda: 1.0),
                              tf.cond(is_train, lambda: self.state_keep_prob, lambda: 1.0),
                              use_last=False,
                              seed=self.seed,
                              reuse=False)
            lstm_out = tf.concat([lstm_out[0], lstm_out[1]], 2, name='lstm_out')
        return lstm_out

    def run_lstm_alt(self, context_out, question_pool, context_len, is_train):
        # tile pooled question rep and concat with context
        q_rep = tf.expand_dims(question_pool, 1) # (batch_size, 1, D)
        context_shape = tf.shape(context_out)[1]
        q_rep = tf.tile(q_rep, [1, context_shape, 1])

        q_c_rep = tf.concat([context_out, q_rep], axis=-1)

        with tf.variable_scope('lstm_') as scope:
            lstm_out_fw = LSTM(q_c_rep,
                               context_len,
                               self.hidden_units,
                               tf.cond(is_train, lambda: self.output_keep_prob, lambda: 1.0),
                               tf.cond(is_train, lambda: self.input_keep_prob, lambda: 1.0),
                               tf.cond(is_train, lambda: self.state_keep_prob, lambda: 1.0),
                               use_last=False,
                               seed=self.seed,
                               reuse=False)
            q_c_rep_rev = _reverse(q_c_rep, context_len, 1, 0)
            lstm_out_rev = LSTM(q_c_rep_rev,
                                context_len,
                                self.hidden_units,
                                tf.cond(is_train, lambda: self.output_keep_prob, lambda: 1.0),
                                tf.cond(is_train, lambda: self.input_keep_prob, lambda: 1.0),
                                tf.cond(is_train, lambda: self.state_keep_prob, lambda: 1.0),
                                use_last=False,
                                seed=self.seed,
                                reuse=True)
            lstm_out_bw = _reverse(lstm_out_rev, context_len, 1, 0)
            lstm_out = tf.concat([lstm_out_fw, lstm_out_bw], 2, name='lstm_out')
        return lstm_out

    def answer_ptr(self, output_lstm, question_out, context_len, question_len, is_train):
        init = attention_pooling(question_out, self.hidden_units, question_len,
                                 self.keep_prob, is_train)
        logits = pointer_net_qa(output_lstm, init, context_len, 2*self.hidden_units,
                                self.keep_prob, is_train)
        return logits

    def decode(self, context, question, context_len, question_len, is_train, labels=None):
        output_lstm = self.run_lstm_alt(context[0], question[1], context_len, is_train)
        logits = self.answer_ptr(output_lstm, question[0], context_len, question_len, is_train)
        probas1 = tf.nn.softmax(logits[0], -1)
        probas2 = tf.nn.softmax(logits[1], -1)
        return logits, [probas1, probas2]
