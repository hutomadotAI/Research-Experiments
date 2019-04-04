import tensorflow as tf

from layers import BiLSTM, attention_pooling, pointer_net_qa
from layers import scaled_dot_product_attention


class Decoder(object):

    def __init__(self, hidden_units, output_keep_prob, input_keep_prob,
                 state_keep_prob, keep_prob, seed=3435):
        self.seed = seed
        self.hidden_units = hidden_units
        self.output_keep_prob = output_keep_prob
        self.input_keep_prob = input_keep_prob
        self.state_keep_prob = state_keep_prob
        self.keep_prob = keep_prob

    def setup_placeholders(self):
        pass

    def run_match_lstm(self, context_out, question_out, context_len, question_len, is_train):
        qc_att = scaled_dot_product_attention(context_out, question_out, memory_len=question_len,
                                              hidden=self.hidden_units, keep_prob=self.keep_prob,
                                              is_train=is_train)
        lstm_out = BiLSTM(
            qc_att,
            context_len,
            self.hidden_units,
            output_dropout_keep_prob=tf.cond(is_train, lambda: self.output_keep_prob, lambda: 1.0),
            input_dropout_keep_prob=tf.cond(is_train, lambda: self.input_keep_prob, lambda: 1.0),
            state_dropout_keep_prob=tf.cond(is_train, lambda: self.state_keep_prob, lambda: 1.0),
            use_last=False,
            seed=self.seed,
            reuse=False
        )
        lstm_out = tf.concat([lstm_out[0], lstm_out[1]], 2, name='lstm_out')
        return lstm_out

    def answer_ptr(self, output_lstm, question_out, context_len, question_len, is_train):
        init = attention_pooling(question_out, self.hidden_units, question_len,
                                 dropout_keep_prob=self.keep_prob, is_train=is_train)
        logits = pointer_net_qa(output_lstm, init, context_len, 2*self.hidden_units,
                                dropout_keep_prob=self.keep_prob, is_train=is_train)
        return logits

    def decode(self, context, question, context_len, question_len, is_train, labels=None):
        output_lstm = self.run_match_lstm(context[0], question[0], context_len, question_len, is_train)
        logits = self.answer_ptr(output_lstm, question[0], context_len, question_len, is_train)
        probas1 = tf.nn.softmax(logits[0], -1)
        probas2 = tf.nn.softmax(logits[1], -1)
        return logits, [probas1, probas2]