import tensorflow as tf

from layers import BiGRU, attention_pooling, pointer_net_qa
from layers import scaled_dot_product_attention


class Decoder(object):

    def __init__(self, hidden_units, output_keep_prob, input_keep_prob,
                 state_keep_prob, keep_prob, n_layers=3, seed=3435):
        self.seed = seed
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.output_keep_prob = output_keep_prob
        self.input_keep_prob = input_keep_prob
        self.state_keep_prob = state_keep_prob
        self.keep_prob = keep_prob

    def setup_placeholders(self):
        pass

    def run_attention(self, context_out, question_out, context_len, question_len, is_train):
        qc_att = scaled_dot_product_attention(context_out,
                                              question_out,
                                              memory_len=question_len,
                                              hidden=self.hidden_units,
                                              keep_prob=self.keep_prob,
                                              is_train=is_train,
                                              scope='attention_layer')
        att_pool, att_out = BiGRU(
            qc_att,
            context_len,
            self.hidden_units,
            output_dropout_keep_prob=tf.cond(is_train, lambda: self.output_keep_prob, lambda: 1.0),
            input_dropout_keep_prob=tf.cond(is_train, lambda: self.input_keep_prob, lambda: 1.0),
            state_dropout_keep_prob=tf.cond(is_train, lambda: self.state_keep_prob, lambda: 1.0),
            n_layers=1,
            residual=False,
            use_last=True,
            seed=self.seed,
            reuse=False,
            scope='attention_layer')
        att_out = tf.concat([att_out[0], att_out[1]], 2, name='att_out')
        return att_out

    def run_match_lstm(self, context_out, question_out, context_len, question_len, is_train):
        qc_att = scaled_dot_product_attention(context_out, question_out, memory_len=context_len,
                                              hidden=self.hidden_units, keep_prob=self.keep_prob,
                                              is_train=is_train, scope='match_lstm_layer')
        lstm_pool, lstm_out = BiGRU(
            qc_att,
            context_len,
            self.hidden_units,
            output_dropout_keep_prob=tf.cond(is_train, lambda: self.output_keep_prob, lambda: 1.0),
            input_dropout_keep_prob=tf.cond(is_train, lambda: self.input_keep_prob, lambda: 1.0),
            state_dropout_keep_prob=tf.cond(is_train, lambda: self.state_keep_prob, lambda: 1.0),
            use_last=True,
            seed=self.seed,
            reuse=False,
            scope='match_lstm_layer')
        lstm_out = tf.concat([lstm_out[0], lstm_out[1]], 2, name='lstm_out')
        return lstm_out

    def answer_ptr(self, output_lstm, question_out, context_len, question_len, is_train):
        init = attention_pooling(question_out, self.hidden_units, question_len,
                                 dropout_keep_prob=self.keep_prob, is_train=is_train)
        logits = pointer_net_qa(output_lstm, init, context_len, 2*self.hidden_units,
                                dropout_keep_prob=self.keep_prob, is_train=is_train)
        return logits

    def decode(self, context, question, context_len, question_len, is_train, labels=None):
        att_out = self.run_attention(context[0], question[0], context_len, question_len, is_train)
        output_lstm = self.run_match_lstm(att_out, att_out, context_len, question_len, is_train)
        logits = self.answer_ptr(output_lstm, question[0], context_len, question_len, is_train)
        probas1 = tf.nn.softmax(logits[0], -1)
        probas2 = tf.nn.softmax(logits[1], -1)
        return logits, [probas1, probas2]
