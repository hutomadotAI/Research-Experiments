
import tensorflow as tf

from layers import BiGRU
from embedding_dropout import get_embedding_dropout_mask


class Encoder(object):
    def __init__(self, hidden_units, output_keep_prob, input_keep_prob, state_keep_prob,
                 n_layers=3, seed=3435):
        self.hidden_units = hidden_units
        self.seed = seed
        self.n_layers = n_layers
        self.output_keep_prob = output_keep_prob
        self.input_keep_prob = input_keep_prob
        self.state_keep_prob = state_keep_prob

    def setup_placeholders(self):
        pass

    def encode(self, inputs, masks, is_train):
        context, question = inputs
        context_mask, question_mask = masks

        with tf.variable_scope("encode"):
            # outshape: [batch_size, 2 * rnn_hidden_units]
            lstm_pool_context, lstm_out_context = BiGRU(
                context,
                context_mask,
                self.hidden_units,
                output_dropout_keep_prob=tf.cond(is_train, lambda: self.output_keep_prob, lambda: 1.0),
                input_dropout_keep_prob=tf.cond(is_train, lambda: self.input_keep_prob, lambda: 1.0),
                state_dropout_keep_prob=tf.cond(is_train, lambda: self.state_keep_prob, lambda: 1.0),
                n_layers=self.n_layers,
                residual=True,
                use_last=True,
                seed=self.seed,
                reuse=False)
            lstm_out_context = tf.concat([lstm_out_context[0], lstm_out_context[1]],
                                         2, name='lstm_out_context')

            lstm_pool_question, lstm_out_question = BiGRU(
                question,
                question_mask,
                self.hidden_units,
                output_dropout_keep_prob=tf.cond(is_train, lambda: self.output_keep_prob, lambda: 1.0),
                input_dropout_keep_prob=tf.cond(is_train, lambda: self.input_keep_prob, lambda: 1.0),
                state_dropout_keep_prob=tf.cond(is_train, lambda: self.state_keep_prob, lambda: 1.0),
                n_layers=self.n_layers,
                residual=True,
                use_last=True,
                seed=self.seed,
                reuse=True)
            lstm_out_question = tf.concat([lstm_out_question[0], lstm_out_question[1]],
                                          2, name='lstm_out_question')

        return [lstm_out_context, lstm_pool_context], [lstm_out_question, lstm_pool_question]


class CharEncoderLSTM(object):
    def __init__(self, hidden_units, char_vocab_size, char_emb_size, keep_prob,
                 emb_keep_prob, output_keep_prob, input_keep_prob, state_keep_prob, seed=3435):
        self.hidden_units = hidden_units
        self.char_vocab_size = char_vocab_size
        self.char_emb_size = char_emb_size
        self.emb_keep_prob = emb_keep_prob
        self.keep_prob = keep_prob
        self.output_keep_prob = output_keep_prob
        self.input_keep_prob = input_keep_prob
        self.state_keep_prob = state_keep_prob
        self.seed = seed

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
            mask = get_embedding_dropout_mask(
                self.emb_keep_prob,
                [self.char_vocab_size, self.char_emb_size],
                char_ids_flat,
                seed=self.seed,
                is_train=is_train
            )
            char_W_drop = char_W * mask

            # shape = (batch, sentence, word, dim of char embeddings)
            embedded_chars = tf.nn.embedding_lookup(char_W_drop, char_ids_flat)

            # shape = (batch x sentence, word, dim of char embeddings)
            word_len_reshaped = tf.reshape(word_len, shape=[-1])

            # shape = (batch x sentence, max_word_length, char_hidden_units)
            cwords_pool, cwords = BiGRU(
                embedded_chars,
                word_len_reshaped,
                self.hidden_units,
                output_dropout_keep_prob=tf.cond(is_train, lambda: self.output_keep_prob, lambda: 1.0),
                input_dropout_keep_prob=tf.cond(is_train, lambda: self.input_keep_prob, lambda: 1.0),
                state_dropout_keep_prob=tf.cond(is_train, lambda: self.state_keep_prob, lambda: 1.0),
                n_layers=1,
                use_last=True,
                seed=self.seed)

            # shape = (batch, sentence, char_hidden_size)
            char_rep = tf.reshape(cwords_pool, shape=[-1, seq_len, 2 * self.hidden_units], name='char_rep')

        return char_rep
