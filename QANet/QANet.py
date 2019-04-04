import logging
import tensorflow as tf

from layers import dropout
from layers_2 import highway
from QANet.encoder import Encoder, CharEncoder
from QANet.decoder import Decoder


class QANet(object):
    def __init__(self, embeddings, output_types, output_shapes, mode='iter', **kwargs):
        self.model_name = 'QANet'
        self.kwargs = kwargs
        self.mode = mode

        self.bs = kwargs['batch_size']
        self.l1_reg_lambda = kwargs['l1_reg']
        self.l2_reg_lambda = kwargs['l2_reg']
        self.emb_keep_prob = kwargs['input_emb_keep_prob']
        self.decay = kwargs['decay']
        self.word_len = kwargs['max_word_len']
        self.embeddings = embeddings
        if mode == 'iter':
            self.handle = tf.placeholder(tf.string, shape=[])
            self.iterator = tf.data.Iterator.from_string_handle(
                self.handle, output_types, output_shapes)
        self.encoder = Encoder(num_filters=kwargs['num_filters'],
                               num_heads=kwargs['num_heads'],
                               keep_prob=kwargs['dropout_keep_prob'],
                               seed=kwargs['random_seed'])
        self.char_encoder = CharEncoder(num_filters=kwargs['num_filters'],
                                        char_vocab_size=kwargs['char_vocab_size'],
                                        char_emb_size=kwargs['char_emb_size'],
                                        seed=kwargs['random_seed'],
                                        emb_keep_prob=kwargs['input_emb_keep_prob'])
        self.decoder = Decoder(num_filters=kwargs['num_filters'],
                               num_heads=kwargs['num_heads'],
                               keep_prob=kwargs['dropout_keep_prob'],
                               seed=kwargs['random_seed'])
        self.question_embedding = None
        self.context_embedding = None
        self.context_ids = None
        self.context_char_ids = None
        self.context_ids_slice = None
        self.context_char_ids_slice = None
        self.c_len = None
        self.c_maxlen = None
        self.cch_len = None
        self.cch_max = None
        self.c_mask = None

        self.question_ids = None
        self.question_char_ids = None
        self.question_ids_slice = None
        self.question_char_ids_slice = None
        self.q_len = None
        self.q_maxlen = None
        self.qch_len = None
        self.qch_max = None
        self.q_mask = None

        self.labels = None
        self.labels_start = None
        self.labels_end = None
        self.qa_ids = None
        self.logger = logging.getLogger('phrase_level_qa.qa_system.QANet')

    def setup_model(self):
        with tf.variable_scope('QANet'):
            self.prepare_data()
            self.setup_placeholders()
            self.setup_system()
            self.setup_predictions()
            self.setup_loss()

    def setup_placeholders(self):
        if self.mode == 'iter':
            self.is_train = tf.placeholder(tf.bool, name="is_train")
        else:
            self.is_train = tf.constant(False, dtype=tf.bool, name="is_train")

    def prepare_data(self):
        if self.mode == 'iter':
            self.context_ids, self.question_ids, self.labels, self.qa_ids,\
                self.context_char_ids, self.question_char_ids =\
                self.iterator.get_next()
            bs = self.bs
        else:
            self.context_ids = tf.placeholder(tf.int32, [1, self.kwargs['max_para_len']],
                                              name='context_ids')
            self.context_char_ids = tf.placeholder(
                tf.int32, [1, self.kwargs['max_para_len'], self.kwargs['max_word_len']],
                name='context_char_ids')
            self.question_ids = tf.placeholder(tf.int32, [1, self.kwargs['max_quest_len']],
                                               name='question_ids')
            self.question_char_ids = tf.placeholder(
                tf.int32, [1, self.kwargs['max_quest_len'], self.kwargs['max_word_len']],
                name='question_char_ids')

            self.labels = tf.placeholder(tf.int32, [1, 2, self.kwargs['max_para_len']],
                                         name='labels')
            self.qa_ids = tf.placeholder(tf.int32, [1], name='qa_ids')
            bs = 1

        self.c_mask = tf.cast(self.context_ids, tf.bool)
        self.q_mask = tf.cast(self.question_ids, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
        self.c_maxlen = tf.reduce_max(self.c_len)
        self.q_maxlen = tf.reduce_max(self.q_len)

        self.cch_len = tf.reduce_sum(
            tf.cast(tf.cast(self.context_char_ids, tf.bool), tf.int32), axis=2)
        self.qch_len = tf.reduce_sum(
            tf.cast(tf.cast(self.question_char_ids, tf.bool), tf.int32), axis=2)
        self.cch_max = tf.reduce_max(self.cch_len)
        self.qch_max = tf.reduce_max(self.qch_len)

        self.context_ids_slice = tf.slice(self.context_ids, [0, 0], [bs, self.c_maxlen])
        self.question_ids_slice = tf.slice(self.question_ids, [0, 0], [bs, self.q_maxlen])
        self.c_mask = tf.slice(self.c_mask, [0, 0], [bs, self.c_maxlen])
        self.q_mask = tf.slice(self.q_mask, [0, 0], [bs, self.q_maxlen])
        self.labels_start = tf.slice(self.labels[:, 0, :], [0, 0], [bs, self.c_maxlen])
        self.labels_end = tf.slice(self.labels[:, 1, :], [0, 0], [bs, self.c_maxlen])

        self.context_char_ids_slice = tf.slice(self.context_char_ids, [0, 0, 0],
                                               [bs, self.c_maxlen, self.cch_max])
        self.question_char_ids_slice = tf.slice(self.question_char_ids, [0, 0, 0],
                                                [bs, self.q_maxlen, self.qch_max])

    def setup_system(self):
        self.c_word_emb = self.setup_word_embeddings(self.context_ids_slice)
        self.c_char_emb = self.setup_char_embeddings(self.context_char_ids_slice,
                                                     self.c_maxlen,
                                                     self.cch_max)
        self.c_emb_combined = tf.concat([self.c_word_emb, self.c_char_emb], axis=-1)
        self.c_embedding_full = highway(
            self.c_emb_combined, size=self.kwargs['num_filters'],
            scope="highway", keep_prob=self.kwargs['dropout_keep_prob'],
            reuse=None, is_train=self.is_train)

        self.q_word_emb = self.setup_word_embeddings(self.question_ids_slice)
        self.q_char_emb = self.setup_char_embeddings(self.question_char_ids_slice,
                                                     self.q_maxlen,
                                                     self.qch_max,
                                                     reuse=True)
        self.q_emb_combined = tf.concat([self.q_word_emb, self.q_char_emb], axis=-1)
        self.q_embedding_full = highway(
            self.q_emb_combined, size=self.kwargs['num_filters'],
            scope="highway", keep_prob=self.kwargs['dropout_keep_prob'],
            reuse=True, is_train=self.is_train)

        self.context = self.encoder.encode(
            self.c_embedding_full,
            self.c_mask,
            self.is_train,
            scope='context_encoding_layer')

        self.question = self.encoder.encode(
            self.q_embedding_full,
            self.q_mask,
            self.is_train,
            scope='question_encoding_layer')

        logits, probs = self.decoder.decode(
            self.context, self.question, self.c_mask, self.q_mask,
            self.c_maxlen, self.q_maxlen, self.is_train)

        self.logits = logits
        self.probs = probs

    def setup_predictions(self):
        outer = tf.matmul(tf.expand_dims(tf.nn.softmax(self.logits[0]), axis=2),
                          tf.expand_dims(tf.nn.softmax(self.logits[1]), axis=1))
        outer = tf.matrix_band_part(outer, 0, 30)
        self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
        self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

    def get_reg_loss(self):
        train_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regularizer = tf.contrib.layers.l1_l2_regularizer(
            scale_l1=self.l1_reg_lambda,
            scale_l2=self.l2_reg_lambda
        )
        reg_loss = tf.contrib.layers.apply_regularization(regularizer, train_vars)
        return reg_loss

    def setup_loss(self):
        with tf.variable_scope("loss"):
            reg_loss = tf.cond(self.is_train, lambda: self.get_reg_loss(), lambda: 0.)

            self.loss_start = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits[0], labels=self.labels_start)
            self.loss_end = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits[1], labels=self.labels_end)

            self.loss = tf.reduce_mean(self.loss_start + self.loss_end, name='loss') + reg_loss

        self.decay_vars()

    def decay_vars(self):
        if self.decay is not None:
            self.var_ema = tf.train.ExponentialMovingAverage(self.decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)

                self.assign_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v:
                        self.assign_vars.append(tf.assign(var, v))

    def setup_char_embeddings(self, char_ids, max_words, max_chars, reuse=False):
        with tf.variable_scope("char_embedding"):
            char_emb = self.char_encoder.encode(
                char_ids, max_words, max_chars, self.is_train, reuse=reuse)
        return char_emb

    def setup_word_embeddings(self, word_ids):
        # Embedding layer
        with tf.name_scope("embedding"):
            self.W_emb = tf.Variable(self.embeddings,
                                     trainable=self.kwargs['train_wemb'],
                                     name="W_emb",
                                     dtype=tf.float32)

            embedding = tf.nn.embedding_lookup(self.W_emb, word_ids)

            drop_embedding = dropout(embedding,
                                     self.kwargs['input_emb_keep_prob'],
                                     self.is_train)

        return drop_embedding

