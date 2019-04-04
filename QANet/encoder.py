import tensorflow as tf

from layers import dropout
from layers_2 import conv, residual_block


class Encoder(object):
    def __init__(self, num_filters, num_heads, keep_prob,
                 seed=3435):
        self.seed = seed
        self.num_filters = num_filters
        self.num_heads = num_heads
        self.keep_prob = keep_prob

    def encode(self, input, mask, is_train, scope='encoding_layer'):
        # output size: [bs, num_filters, emb_size]
        with tf.variable_scope("Embedding_Encoder_Layer"):
            out = residual_block(input,
                                 num_blocks=1,
                                 num_conv_layers=4,
                                 kernel_size=7,
                                 mask=mask,
                                 num_filters=self.num_filters,
                                 num_heads=self.num_heads,
                                 scope=scope,
                                 bias=False,
                                 keep_prob=self.keep_prob,
                                 is_train=is_train)

        return out


class CharEncoder(object):
    def __init__(self, num_filters, char_vocab_size, char_emb_size,
                 emb_keep_prob, char_mat=None, seed=3435):
        self.num_filters = num_filters
        self.char_emb_size = char_emb_size
        if char_mat:
            self.char_mat = tf.get_variable(
                "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))
        else:
            self.char_mat = tf.get_variable(shape=[char_vocab_size, self.char_emb_size],
                                            initializer=tf.random_normal_initializer(stddev=0.1,
                                                                                     seed=seed),
                                            trainable=True, name="char_mat", dtype=tf.float32)
        self.emb_keep_prob = emb_keep_prob
        self.seed = seed

    def encode(self, input_ids, max_words, max_chars, is_train, reuse=False):

        with tf.variable_scope("char_embedding", reuse=reuse):
            char_emb = tf.reshape(tf.nn.embedding_lookup(
                self.char_mat, input_ids), [-1, max_chars, self.char_emb_size])

            char_emb = dropout(char_emb, self.emb_keep_prob, is_train)

            ch_emb = conv(char_emb, self.num_filters,
                          bias=True, activation=tf.nn.relu,
                          kernel_size=5, name="char_conv", reuse=None)

            ch_emb = tf.reduce_max(ch_emb, axis=1)
            ch_emb = tf.reshape(ch_emb, [-1, max_words, ch_emb.shape[-1]])

        return ch_emb
