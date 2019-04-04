import os
import json
import datetime
import numpy as np
import tensorflow as tf

from init_args import init_args
from match_lstm.match_lstm import MatchLSTM
from cudnn_match_lstm.match_lstm import CudnnMatchLSTM
from base_model.base_model import BaseModel
from RNet.RNet import RNet
from cudnnRNet.RNet import cudnnRNet
from lstm.lstm import LSTM
from QANet.QANet import QANet
from qa_system import QASystem
from load_data import SquadDataset
from logger import setup_logger


def main():
    FLAGS = init_args()
    kwargs = FLAGS.flag_values_dict()
    if not FLAGS.use_out_dir:
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        out_dir = os.path.join(FLAGS.out_dir, timestamp)
        os.makedirs(os.path.join(out_dir))
    else:
        timestamp = FLAGS.use_out_dir
        out_dir = os.path.join(FLAGS.out_dir, timestamp)

    # logger
    logger = setup_logger(name='phrase_level_qa', out_dir=out_dir)
    logger.info("saving to {}".format(out_dir))

    # print parameters
    logger.info("Parameters:")
    for attr, value in sorted(FLAGS.flag_values_dict().items()):
        logger.info("{}={}".format(attr.upper(), value))

    # load evaluation files
    with open(os.path.join(kwargs["data_dir"], "train_eval.json"), "r") as fh:
        train_eval_file = json.load(fh)
    with open(os.path.join(kwargs["data_dir"], "dev_eval.json"), "r") as fh:
        dev_eval_file = json.load(fh)

    train_set = SquadDataset(data_dir=kwargs['data_dir'], use='train', batch_size=FLAGS.batch_size)
    train_init, train_iter = train_set.setup_data_pipeline(shuffle=True)
    train_batches = train_set.get_num_batches()

    dev_set = SquadDataset(data_dir=kwargs['data_dir'], use='dev', batch_size=FLAGS.batch_size)
    dev_init, dev_iter = dev_set.setup_data_pipeline(shuffle=False)
    dev_batches = 150  # dev_set.get_num_batches()

    logger.info("INFO: train: {} dev: {}".format(len(train_set), len(dev_set)))

    initW = np.fromfile(os.path.join(
        FLAGS.data_dir, 'initW_' + FLAGS.use + '.dat')).reshape((-1, FLAGS.word_emb_size))

    if FLAGS.model_name == 'base':
        logger.info("INFO: using Base Model")
        model = BaseModel(
            embeddings=initW,
            output_types=train_set.get_output_types(),
            output_shapes=train_set.get_output_shapes(),
            **kwargs
        )
    elif FLAGS.model_name == 'lstm':
        logger.info("INFO: using LSTM Decoder")
        model = LSTM(
            embeddings=initW,
            output_types=train_set.get_output_types(),
            output_shapes=train_set.get_output_shapes(),
            **kwargs
        )
    elif FLAGS.model_name == 'match':
        logger.info("INFO: using Match-LSTM")
        model = MatchLSTM(
            embeddings=initW,
            output_types=train_set.get_output_types(),
            output_shapes=train_set.get_output_shapes(),
            **kwargs
        )
    elif FLAGS.model_name == 'cudnn_match':
        logger.info("INFO: using Match-LSTM")
        model = CudnnMatchLSTM(
            embeddings=initW,
            output_types=train_set.get_output_types(),
            output_shapes=train_set.get_output_shapes(),
            **kwargs
        )
    elif FLAGS.model_name == 'rnet':
        logger.info("INFO: using RNet")
        model = RNet(
            embeddings=initW,
            output_types=train_set.get_output_types(),
            output_shapes=train_set.get_output_shapes(),
            **kwargs
        )
    elif FLAGS.model_name == 'cudnn-rnet':
        logger.info("INFO: using cudnn-RNet")
        model = cudnnRNet(
            embeddings=initW,
            output_types=train_set.get_output_types(),
            output_shapes=train_set.get_output_shapes(),
            **kwargs
        )
    elif FLAGS.model_name == 'qanet':
        logger.info("INFO: using QANet")
        model = QANet(
            embeddings=initW,
            output_types=train_set.get_output_types(),
            output_shapes=train_set.get_output_shapes(),
            **kwargs
        )
    else:
        raise NotImplementedError("model name not implemented")

    qa_sys = QASystem(model=model,
                      outdir=out_dir,
                      timestamp=timestamp,
                      inits=[train_init, dev_init],
                      iterators=[train_iter, dev_iter],
                      num_batches=[train_batches, dev_batches],
                      **kwargs)
    logger.info("INFO: qa system defined")

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction)
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    logger.info("INFO: started session")

    with sess.as_default():
        qa_sys.init_model(sess)
        qa_sys.train(sess, train_eval_file, dev_eval_file)
    logger.info("INFO: training finished")


if __name__ == '__main__':
    main()
