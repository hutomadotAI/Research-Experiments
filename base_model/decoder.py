import tensorflow as tf

from layers import mask_score, Dense


class BaseDecoder(object):
    def __init__(self, seed=3435):
        self.seed = seed

    def setup_placeholders(self):
        pass

    def decode(self, context, question, context_len, question_len, is_train):
        # shape: context_pool: [-1, seq_len, 2 * rnn_hidden_units], question_out: [-1, 2 * rnn_hidden_units]
        with tf.name_scope("decode"):
            input_size = question[1].get_shape()[-1].value
            q_rep_start = Dense(question[1], input_size, seed=self.seed, scope='q_rep_start')
            q_rep_end = Dense(question[1], input_size, seed=self.seed, scope='q_rep_end')

            q_rep_start_exp = tf.expand_dims(q_rep_start, 1)
            q_rep_end_exp = tf.expand_dims(q_rep_end, 1)

            logits_start = tf.reduce_sum(context[0] * q_rep_start_exp, 2)
            logits_end = tf.reduce_sum(context[0] * q_rep_end_exp, 2)

            func = lambda score: mask_score(score, context_len, float("-1e30"))
            logits_start_masked = func(logits_start)
            logits_end_masked = func(logits_end)

            prob_start = tf.nn.softmax(logits_start_masked, 1)
            prob_end = tf.nn.softmax(logits_end_masked, 1)

        return [logits_start_masked, logits_end_masked], [prob_start, prob_end]