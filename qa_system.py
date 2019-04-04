import os
import sys
import numpy as np
import time
import logging
import signal
import json
from tabulate import tabulate

import tensorflow as tf

import config as conf
from utils import humanize_time, evaluate, convert_tokens


class QASystem(object):

    def __init__(self, model, outdir, timestamp, num_batches, inits,
                 iterators, logger_name='phrase_level_qa',  **kwargs):
        self.kwargs = kwargs
        self.timestamp = timestamp
        self.out_dir = outdir
        self.optimiser = kwargs['optimiser']
        self.summaries = kwargs['summaries']
        self.batch_size = kwargs['batch_size']
        self.num_epochs = kwargs['num_epochs']
        self.evaluate_every = kwargs['evaluate_every']
        self.data_dir = kwargs['data_dir']
        self.inits = inits
        self.iterators = iterators
        self.num_batches = num_batches

        self.model = model
        self.logger = logging.getLogger(logger_name + '.qa_system')

        self.train_summary_writer = None
        self.dev_summary_writer = None
        self.best_em_score = None
        self.best_em_score_start = None
        self.best_em_score_end = None
        self.best_f1 = None
        self.best_loss = None
        self.train_metrics = None

        with tf.variable_scope("QA_System"):
            self.model.setup_model()
            self.setup_train_op()
            self.saver = tf.train.Saver(tf.global_variables())
            if self.summaries:
                self.setup_saver()

    def setup_train_op(self):
        # Define Training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable=False)

        if self.optimiser == 'Adam':
            optimizer = tf.train.AdamOptimizer(
                self.learning_rate, beta1=0.8, beta2=0.999, epsilon=1e-7)
        elif self.optimiser == 'AdaDelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate,
                                                   rho=0.95,
                                                   epsilon=1e-6)
        elif self.optimiser == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            self.logger.error("optimiser is not implemented")
            NotImplementedError("optimiser is not implemented")
        self.logger.info("defined classifier object")

        vars = tf.trainable_variables()
        self.grads_and_vars = optimizer.compute_gradients(self.model.loss, var_list=vars)
        grads, vars = zip(*self.grads_and_vars)
        capped_grads, _ = tf.clip_by_global_norm(grads, 5.)
        self.tr_op_set = optimizer.apply_gradients(zip(capped_grads, vars),
                                                   global_step=self.global_step, name='tr_op_set')
        tf.add_to_collection("optimizer", self.tr_op_set)
        self.logger.info("defined training_ops")

        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    def setup_saver(self):
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in self.grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar(
                    "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.grad_summaries_merged = tf.summary.merge(grad_summaries)
        self.logger.info("defined gradient summaries")

        # Train Summaries
        self.train_summary_dir = os.path.join(self.out_dir, "summaries", "train")
        self.logger.info("defined train summaries")

        # Dev summaries
        self.dev_summary_dir = os.path.join(self.out_dir, "summaries", "dev")
        self.logger.info("defined dev summaries")

    def save_configs(self, session, timestamp):
        # save config file
        confOut = os.path.join(self.out_dir, "config_" + timestamp + ".txt")
        if not os.path.isfile(confOut):
            conf.write_config_to_file(confOut)
            self.logger.info("config file saved to {}".format(confOut))

        graph_def = tf.get_default_graph().as_graph_def()
        graphpb_txt = str(graph_def)
        graphOut = os.path.join(self.checkpoint_dir, "graphpb.txt")
        with open(graphOut, 'w') as f:
            f.write(graphpb_txt)

        if self.summaries:
            self.train_summary_writer = tf.summary.FileWriter(self.train_summary_dir, session.graph)
            self.dev_summary_writer = tf.summary.FileWriter(self.dev_summary_dir, session.graph)

    def init_model(self, sess):
        sess.run(self.init)
        self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
        if os.path.exists(self.checkpoint_dir):
            ckpt = self.kwargs['checkpoint']
            checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_dir)
            if ckpt != '':
                checkpoint_file = checkpoint_file.rsplit('-', 1)[0] + "-" + ckpt
            self.logger.info("init model from checkpoint file {}".format(checkpoint_file))
            self.epoch_start = int(checkpoint_file.rsplit('-', 1)[-1]) + 1
            self.saver.restore(sess, checkpoint_file)
            self.logger.info("starting epoch: {}".format(self.epoch_start))
        else:
            self.epoch_start = 0
            os.makedirs(self.checkpoint_dir)
            self.logger.info("create model with random parameters")

    def to_saved_model(self, session, version):
        export_dir = os.path.join(self.out_dir, 'saved_model/' + str(version))

        inputs = {'char_ids': self.model.question_char_ids,
                  'ids': self.model.question_ids}
        outputs = {'text_encoding': self.model.question}
        tf.saved_model.simple_save(
            session,
            export_dir,
            inputs,
            outputs
        )

    def train_batch(self, hdl, sess, tr_op_set, global_step):
        feed_dict = {self.model.is_train: True, self.model.handle: hdl}

        if self.summaries:
            _, step, loss, logits, ans, grad_sums = sess.run(
                [tr_op_set, global_step, self.model.loss, self.model.logits,
                 self.model.labels, self.grad_summaries_merged],
                feed_dict)
        else:
            _, step, loss, logits, ans, l_start, l_end = sess.run(
                [tr_op_set, global_step, self.model.loss, self.model.logits,
                 self.model.labels, self.model.labels_start, self.model.labels_end],
                feed_dict)
            grad_sums = None

        if np.isnan(loss) or np.isinf(loss):
            self.logger.error("answers: {}".format(ans.shape))
            self.logger.error("as: {} ae: {}".format(np.argmax(ans[:, 0, :], axis=1),
                                                     np.argmax(ans[:, 1, :], axis=1)))
            self.logger.error("logits: {}".format(logits[0].shape))

        return loss, step, grad_sums

    def evaluate_batch(self, epoch, num_batches, eval_file, sess, data_type, str_handle):
        answer_dict = {}
        remapped_dict = {}
        losses = []
        tStart = time.time()
        for nb in range(1, num_batches + 1):
            if self.summaries:
                qa_id, loss, yp1, yp2, grad_sums = sess.run(
                    [self.model.qa_ids, self.model.loss, self.model.yp1, self.model.yp2,
                     self.grad_summaries_merged],
                    feed_dict={self.model.handle: str_handle, self.model.is_train: False})
            else:
                qa_id, loss, yp1, yp2 = sess.run(
                    [self.model.qa_ids, self.model.loss, self.model.yp1, self.model.yp2],
                    feed_dict={self.model.handle: str_handle, self.model.is_train: False})
                grad_sums = None
            answer_dict_, remapped_dict_ = convert_tokens(
                eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
            answer_dict.update(answer_dict_)
            remapped_dict.update(remapped_dict_)
            losses.append(loss)
            running_time = time.time() - tStart
            rt = humanize_time(running_time / float(nb + 1) * num_batches - running_time)
            print('Epoch {0:d} {1:s}: {2:d}/{3:d} batches; ETA: {4:s}'.format(
                epoch+1+self.epoch_start, data_type, nb, num_batches, rt), end='\r')

        if data_type == 'test':
            with open(os.path.join(self.out_dir, 'answer.json'), "w") as fh:
                json.dump(remapped_dict, fh)
        loss = np.mean(losses)
        metrics = evaluate(eval_file, answer_dict)
        metrics["loss"] = loss
        loss_sum = tf.Summary(value=[tf.Summary.Value(
            tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
        f1_sum = tf.Summary(value=[tf.Summary.Value(
            tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
        em_sum = tf.Summary(value=[tf.Summary.Value(
            tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
        return metrics, [loss_sum, f1_sum, em_sum, grad_sums]

    def train(self, session, train_eval, dev_eval):
        signal.signal(signal.SIGINT, self.signal_handler)

        self.save_configs(session, self.timestamp)
        train_handle = session.run(self.iterators[0].string_handle())
        dev_handle = session.run(self.iterators[1].string_handle())

        self.best_loss = 10e10
        self.best_em_score = 0.
        self.best_f1 = 0.
        patience = 0
        lr = self.kwargs['learning_rate']
        session.run(tf.assign(self.learning_rate, tf.constant(lr, dtype=tf.float32)))

        session.run(self.inits[0])
        session.run(self.inits[1])
        for e in range(self.num_epochs):
            tStart = time.time()
            for nb in range(self.num_batches[0]):
                loss, step, grad_sum = self.train_batch(
                    train_handle, session, self.tr_op_set, self.global_step)
                if self.summaries:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(
                        tag="model/loss", simple_value=loss), ])
                    self.train_summary_writer.add_summary(loss_sum)
                    self.train_summary_writer.add_summary(grad_sum)

                running_time = time.time() - tStart
                rt = humanize_time(
                    running_time / float(nb + 1) * self.num_batches[0] - running_time)
                print('Epoch {0:d} TRAIN: {1:d}/{2:d} batches; ETA: {3:s} - train_loss: {4:.4f}'.format(
                    e+1+self.epoch_start, nb, self.num_batches[0], rt, np.round(loss, 4)), end='\r')

            self.train_metrics, summs = self.evaluate_batch(
                e, self.num_batches[1], train_eval, session, "train", train_handle)

            running_time = time.time() - tStart
            self.logger.info(
                "TRAIN epoch {0:d}: loss: {1:.4f}, em_score: {2:.4f}, em_score_start: {3:.4f}, "
                "em_score_end: {4:.4f}, f1_score: {5:.4f}, running_time: {6:s}".format(
                    e+1+self.epoch_start, self.train_metrics['loss'], self.train_metrics['exact_match'],
                    self.train_metrics['em_start'], self.train_metrics['em_end'],
                    self.train_metrics['f1'], humanize_time(running_time)))

            # =========== evaluating =========== #
            if e % self.evaluate_every == 0:
                tStart = time.time()
                dev_metrics, summs = self.evaluate_batch(
                    e, self.num_batches[1], dev_eval, session, "dev", dev_handle)

                running_time = time.time() - tStart
                self.logger.info("DEV epoch {0:d}: em_score: {1:.4f}, em_score_start: {2:.4f}, "
                    "em_score_end: {3:.4f}, f1_score: {4:.4f}, loss: {5:.4f} running_time: {6:s}".format(
                        e+1+self.epoch_start, dev_metrics["exact_match"], dev_metrics["em_start"],
                        dev_metrics["em_end"], dev_metrics["f1"], dev_metrics["loss"],
                        humanize_time(running_time)))

                if (dev_metrics["exact_match"] >= self.best_em_score) or\
                        (dev_metrics["f1"] > self.best_f1):
                    self.best_loss = dev_metrics["loss"]
                    self.best_em_score = dev_metrics["exact_match"]
                    self.best_em_score_start = dev_metrics["em_start"]
                    self.best_em_score_end = dev_metrics["em_end"]
                    self.best_f1 = dev_metrics["f1"]
                    self.saver.save(
                        session, self.checkpoint_prefix, global_step=e+self.epoch_start)
                    tf.train.write_graph(session.graph.as_graph_def(), self.checkpoint_prefix,
                                         "graph" + str(e+self.epoch_start) + ".pb", as_text=False)
                    self.logger.info(
                        "Saved model {0:d} with em_score={1:.4f} checkpoint to {2:s}".format(
                            e+self.epoch_start, self.best_em_score, self.checkpoint_prefix))
                    patience = 0
                else:
                    patience += 1

                    if patience >= self.kwargs['patience']:
                        lr /= 2.
                        session.run(tf.assign(
                            self.learning_rate, tf.constant(lr, dtype=tf.float32)))
                        self.logger.info("learning rate reduced to {}".format(lr))
                        patience = 0
                        if lr < 1e-6:
                            self.logger.info("learning rate is below 1e-6; stopping")
                            break

        self.save_results(
            [self.best_em_score, self.best_em_score_start, self.best_em_score_end, self.best_f1],
            [self.train_metrics["loss"], self.train_metrics["exact_match"],
             self.train_metrics["em_start"], self.train_metrics["em_end"],
             self.train_metrics["f1"]])

    def test(self, session, test_eval):
        self.logger.info("testing started")
        test_handle = session.run(self.iterators[0].string_handle())

        tStart = time.time()
        session.run(self.inits[0])
        test_metrics, _ = self.evaluate_batch(
            0, self.num_batches[0], test_eval, session, "test", test_handle)

        running_time = time.time() - tStart
        self.logger.info(
            "TEST: em_score: {0:.4f}, em_score_start: {1:.4f}, "
            "em_score_end: {2:.4f}, f1_score: {3:.4f}, loss: {4:.4f} running_time: {5:s}".format(
                test_metrics["exact_match"], test_metrics["em_start"],
                test_metrics["em_end"], test_metrics["f1"], test_metrics["loss"],
                humanize_time(running_time)))

    def save_results(self, dev_res, train_res):
        with open(os.path.join(self.out_dir, 'test_results.txt'), 'w') as f:
            f.write(tabulate(
                [(self.model.model_name, dev_res[0], dev_res[1], dev_res[2], dev_res[3],
                  train_res[0], train_res[1], train_res[2], train_res[3], train_res[4])],
                headers=('model', 'em_score', 'em_score_start', 'em_score_end', 'f1',
                         'train_loss', 'train_em', 'train_em_start', 'train_em_end', 'train_f1')))

    def signal_handler(self, signal, frame):
        self.logger.info("exiting program")
        self.save_results(
            [self.best_em_score, self.best_em_score_start, self.best_em_score_end, self.best_f1],
            [self.train_metrics["loss"], self.train_metrics["exact_match"],
             self.train_metrics["em_start"], self.train_metrics["em_end"],
             self.train_metrics["f1"]])
        sys.exit()
