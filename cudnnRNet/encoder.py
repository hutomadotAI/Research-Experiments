
import tensorflow as tf

from cudnn_layers import BiGRU
from layers import dropout


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

        with tf.variable_scope("encode"):
            # outshape: [batch_size, 2 * rnn_hidden_units]
            gru = BiGRU(num_units=self.hidden_units, num_layers=self.n_layers, batch_size=self.batch_size,
                        input_size=context.get_shape()[-1].value, keep_prob=self.keep_prob,
                        is_train=is_train, seed=self.seed)
            lstm_pool_context, lstm_out_context = gru(context, seq_len=context_mask,
                                                      concat_layers=True, use_last=True)
            lstm_pool_question, lstm_out_question = gru(question, seq_len=question_mask,
                                                        concat_layers=True, use_last=True)

        return [lstm_out_context, lstm_pool_context], [lstm_out_question, lstm_pool_question]


class CharEncoderLSTM(object):
    def __init__(self, hidden_units, char_vocab_size, char_emb_size, batch_size,
                 keep_prob, emb_keep_prob, n_layers=1, seed=3435):
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.seed = seed
        self.keep_prob = keep_prob
        self.char_vocab_size = char_vocab_size
        self.char_emb_size = char_emb_size
        self.emb_keep_prob = emb_keep_prob

    def setup_placeholder(self):
        pass

    def encode(self, inputs, word_masks, is_train):
        context, question = inputs
        context_word_mask, question_word_mask = word_masks

        with tf.variable_scope("char_emb"):
            context_rep = self.encode_chars(context, context_word_mask, is_train)
            question_rep = self.encode_chars(question, question_word_mask, is_train, reuse=True)

        return context_rep, question_rep

    def encode_chars(self, char_ids, word_len, is_train, reuse=False):
        # compute word embedding from chars
        with tf.variable_scope("char_2_word_lstm", reuse=reuse):
            seq_len = tf.shape(char_ids)[1]
            max_char_len = tf.shape(char_ids)[2]
            char_ids_flat = tf.reshape(char_ids, shape=[-1, max_char_len])

            char_W = tf.get_variable(shape=[self.char_vocab_size, self.char_emb_size],
                                     trainable=True, name="char_W", dtype=tf.float32)

            # shape = (batch, sentence, word, dim of char embeddings)
            embedded_chars = tf.nn.embedding_lookup(char_W, char_ids_flat)

            embedded_chars = dropout(
                embedded_chars, keep_prob=self.emb_keep_prob, is_train=is_train)

            word_len_reshaped = tf.reshape(word_len, shape=[-1])

            # shape = (batch x sentence, max_word_length, char_hidden_units)
            gru = BiGRU(num_units=self.hidden_units,
                        num_layers=1,
                        batch_size=self.batch_size*seq_len,
                        input_size=embedded_chars.get_shape()[-1].value,
                        keep_prob=self.keep_prob,
                        is_train=is_train,
                        seed=self.seed)
            cwords_pool, cwords = gru(embedded_chars,
                                      seq_len=word_len_reshaped,
                                      use_last=True,
                                      concat_layers=False)

            # shape = (batch, sentence, char_hidden_size)
            char_rep = tf.reshape(cwords_pool,
                                  shape=[self.batch_size, seq_len, 2 * self.hidden_units],
                                  name='char_rep')

        return char_rep
