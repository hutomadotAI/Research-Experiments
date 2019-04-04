import os
import numpy as np
import json
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


def test():
    FLAGS = init_args()
    kwargs = FLAGS.flag_values_dict()
    timestamp = FLAGS.use_out_dir
    out_dir = os.path.join(FLAGS.out_dir, timestamp)

    # logger
    logger_name = 'phrase_level_qa.test'
    logger = setup_logger(name=logger_name, out_dir=out_dir)
    logger.info("saving to {}".format(out_dir))

    # print parameters
    logger.info("Parameters:")
    for attr, value in sorted(FLAGS.flag_values_dict().items()):
        logger.info("{}={}".format(attr.upper(), value))

    # load evaluation files
    with open(os.path.join(kwargs["data_dir"], "test_eval.json"), "r") as fh:
        test_eval_file = json.load(fh)

    test_set = SquadDataset(data_dir=kwargs['data_dir'], use='test',
                            batch_size=FLAGS.batch_size, max_para_len=1000)
    test_init, test_iter = test_set.setup_data_pipeline(shuffle=False)
    test_batches = test_set.get_num_batches()
    logger.info("INFO: test: {}".format(len(test_set)))

    initW = np.fromfile(os.path.join(
        FLAGS.data_dir, 'initW_' + FLAGS.use + '.dat')).reshape((-1, FLAGS.word_emb_size))

    if FLAGS.model_name == 'base':
        logger.info("INFO: using Base Model")
        model = BaseModel(
            embeddings=initW,
            output_types=test_set.get_output_types(),
            output_shapes=test_set.get_output_shapes(),
            **kwargs
        )
    elif FLAGS.model_name == 'lstm':
        logger.info("INFO: using LSTM Decoder")
        model = LSTM(
            embeddings=initW,
            output_types=test_set.get_output_types(),
            output_shapes=test_set.get_output_shapes(),
            **kwargs
        )
    elif FLAGS.model_name == 'match':
        logger.info("INFO: using Match-LSTM")
        model = MatchLSTM(
            embeddings=initW,
            output_types=test_set.get_output_types(),
            output_shapes=test_set.get_output_shapes(),
            **kwargs
        )
    elif FLAGS.model_name == 'cudnn_match':
        logger.info("INFO: using Match-LSTM")
        model = CudnnMatchLSTM(
            embeddings=initW,
            output_types=test_set.get_output_types(),
            output_shapes=test_set.get_output_shapes(),
            **kwargs
        )
    elif FLAGS.model_name == 'rnet':
        logger.info("INFO: using RNet")
        model = RNet(
            embeddings=initW,
            output_types=test_set.get_output_types(),
            output_shapes=test_set.get_output_shapes(),
            **kwargs
        )
    elif FLAGS.model_name == 'cudnn-rnet':
        logger.info("INFO: using cudnn-RNet")
        model = cudnnRNet(
            embeddings=initW,
            output_types=test_set.get_output_types(),
            output_shapes=test_set.get_output_shapes(),
            **kwargs
        )
    elif FLAGS.model_name == 'qanet':
        logger.info("INFO: using QANet")
        model = QANet(
            embeddings=initW,
            output_types=test_set.get_output_types(),
            output_shapes=test_set.get_output_shapes(),
            **kwargs
        )
    else:
        raise NotImplementedError("model name not implemented")

    qa_sys = QASystem(model=model,
                      outdir=out_dir,
                      timestamp=timestamp,
                      inits=[test_init],
                      iterators=[test_iter],
                      num_batches=[test_batches],
                      logger_name=logger_name,
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
        qa_sys.test(sess, test_eval_file)
    logger.info("INFO: test finished")


if __name__ == '__main__':
    test()

