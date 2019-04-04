import tensorflow as tf

from layers_2 import optimized_trilinear_for_attention, conv, residual_block, mask_logits


class Decoder(object):

    def __init__(self, num_filters, num_heads, keep_prob, seed=3435):
        self.seed = seed
        self.num_filters = num_filters
        self.num_heads = num_heads
        self.keep_prob = keep_prob

    def setup_placeholders(self):
        pass

    def decode(self, context, question, c_mask, q_mask, c_maxlen, q_maxlen, is_train):

        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            S = optimized_trilinear_for_attention(
                [context, question], c_maxlen, q_maxlen,
                input_keep_prob=self.keep_prob, is_train=is_train)
            mask_q = tf.expand_dims(q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask_q))
            mask_c = tf.expand_dims(c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask_c), axis=1), (0, 2, 1))
            self.c2q = tf.matmul(S_, question)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), context)
            attention_outputs = [context, self.c2q, context * self.c2q, context * self.q2c]

        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.concat(attention_outputs, axis=-1)
            self.enc = [conv(inputs, self.num_filters, name="input_projection")]
            for i in range(3):
                if i % 2 == 0:  # dropout every 2 blocks
                    self.enc[i] = tf.nn.dropout(self.enc[i], self.keep_prob)
                self.enc.append(
                    residual_block(
                        self.enc[i],
                        num_blocks=7,
                        num_conv_layers=2,
                        kernel_size=5,
                        mask=c_mask,
                        num_filters=self.num_filters,
                        num_heads=self.num_heads,
                        scope="Model_Encoder",
                        bias=False,
                        reuse=True if i > 0 else None,
                        keep_prob=self.keep_prob,
                        is_train=is_train)
                )

        with tf.variable_scope("Output_Layer"):
            start_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[2]], axis=-1), 1,
                                           bias=False, name="start_pointer"), -1)

            end_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[3]], axis=-1), 1,
                                         bias=False, name="end_pointer"), -1)

            logits = [mask_logits(start_logits, c_mask),
                      mask_logits(end_logits, c_mask)]

            logits1, logits2 = [l for l in logits]

            probas1 = tf.nn.softmax(logits1, -1)
            probas2 = tf.nn.softmax(logits2, -1)

        return [logits1, logits2], [probas1, probas2]