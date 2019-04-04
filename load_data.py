import os
import pickle
import numpy as np
import logging
import tensorflow as tf


class SquadDataset(object):
    def __init__(self, data_dir, use, batch_size, max_para_len=400):
        self.max_para_len = max_para_len
        self.question_file = os.path.join(data_dir, use+'.ids.question')
        self.context_file = os.path.join(data_dir, use+'.ids.context')
        self.answer_file = os.path.join(data_dir, use+'.span')
        self.qa_ids = os.path.join(data_dir, use+'.qa_ids')
        self.context_char_file = os.path.join(data_dir, use+'.char_ids.context')
        self.question_char_file = os.path.join(data_dir, use+'.char_ids.question')
        self.context_char = self.load_pickle(self.context_char_file)
        self.question_char = self.load_pickle(self.question_char_file)
        self.batch_size = batch_size
        self.logger = logging.getLogger('phrase_level_qa.squad')

        self.length = None

    def get_num_batches(self):
        return int(np.ceil(len(self) / float(self.batch_size)))

    def get_output_types(self):
        return (tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64)

    def get_output_shapes(self):
        return ([None, None], [None, None], [None, None, None],
                [None], [None, None, None], [None, None, None])

    def setup_data_pipeline(self, shuffle=True):
        train_data = tf.data.Dataset.from_generator(
            lambda: self,
            output_types=(tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64),
            output_shapes=([None], [None], [None, None], None, [None, None], [None, None]))
        train_data = train_data.repeat()
        if shuffle:
            train_data = train_data.shuffle(buffer_size=10000)
        train_data = train_data.batch(self.batch_size)

        # create an reinitializable iterator given the dataset structure
        iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                   train_data.output_shapes)

        init = iterator.make_initializer(train_data)

        return init, iterator

    def iter_file(self, path, sep=','):
        with open(path, 'r') as f:
            for line in f:
                sample = line.strip().split(sep)
                sample = list(map(int, sample))
                yield sample, len(sample)

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        return d

    def __iter__(self):
        context_file_iter = self.iter_file(self.context_file, sep=',')
        question_file_iter = self.iter_file(self.question_file, sep=',')
        answer_file_iter = self.iter_file(self.answer_file, sep=',')
        qa_ids_file_iter = self.iter_file(self.qa_ids, sep=',')

        for c, q, a, id, cc, qc in zip(context_file_iter, question_file_iter,
                                       answer_file_iter, qa_ids_file_iter,
                                       self.context_char, self.question_char):
            if c[1] > self.max_para_len:
                c = (list(c[0][:self.max_para_len]), c[1])
            a_start = np.eye(self.max_para_len)[a[0][0]]
            a_end = np.eye(self.max_para_len)[a[0][1]]
            yield (c[0], q[0], np.array([a_start, a_end]), id[0][0], cc, qc)

    def __len__(self):
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length
