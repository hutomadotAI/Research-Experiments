import tensorflow as tf

from layers import attention_pooling, pointer_net_qa, scaled_dot_product_attention
from cudnn_layers import BiGRU


class Decoder(object):

    def __init__(self, batch_size, hidden_units, keep_prob, n_layers=1, seed=3435):
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.seed = seed
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
                                              scope='att_layer')

        gru = BiGRU(num_units=self.hidden_units, num_layers=1, batch_size=self.batch_size,
                    input_size=qc_att.get_shape()[-1].value, keep_prob=self.keep_prob,
                    is_train=is_train, seed=self.seed, scope='att_layer')
        att_out = gru(qc_att, seq_len=context_len, concat_layers=True, use_last=False)
        return att_out

    def run_match_lstm(self, context_out, question_out, context_len, question_len, is_train):
        qc_att = scaled_dot_product_attention(context_out, question_out, memory_len=context_len,
                                              hidden=self.hidden_units, keep_prob=self.keep_prob,
                                              is_train=is_train, scope='match_lstm')

        gru = BiGRU(num_units=self.hidden_units, num_layers=1, batch_size=self.batch_size,
                    input_size=qc_att.get_shape()[-1].value, keep_prob=self.keep_prob,
                    is_train=is_train, seed=self.seed)
        lstm_out = gru(
            qc_att, seq_len=context_len, concat_layers=True, use_last=False, scope='match_lstm')

        return lstm_out

    def answer_ptr(self, output_lstm, question_out, context_len, question_len, is_train):
        init = attention_pooling(question_out, output_lstm.get_shape()[-1].value, question_len,
                                 dropout_keep_prob=self.keep_prob, is_train=is_train)
        logits = pointer_net_qa(output_lstm, init, context_len, init.get_shape()[-1].value,
                                dropout_keep_prob=self.keep_prob, is_train=is_train)
        return logits

    def decode(self, context, question, context_len, question_len, is_train, labels=None):
        att_out = self.run_attention(context[0], question[0], context_len, question_len, is_train)
        output_lstm = self.run_match_lstm(att_out, att_out, context_len, question_len, is_train)
        logits = self.answer_ptr(output_lstm, question[0], context_len, question_len, is_train)
        probas1 = tf.nn.softmax(logits[0], -1)
        probas2 = tf.nn.softmax(logits[1], -1)
        return logits, [probas1, probas2]
