import sys
import tensorflow as tf

import config as conf
from config import get_from_config as gfc


def init_args():
    FLAGS = tf.app.flags.FLAGS

    try:
        config_path = sys.argv[-1]
    except:
        config_path = 'config.yaml'
    conf.loadconfig(config_path)

    tf.app.flags.DEFINE_string("model_name", gfc("model_name"), "defines which model is used")
    tf.app.flags.DEFINE_float("l2_reg", gfc("l2_reg"), "L2 regularization lambda")
    tf.app.flags.DEFINE_float("l1_reg", gfc("l1_reg"), "L1 regularization lambda")
    tf.app.flags.DEFINE_integer("random_seed", gfc("random_seed"),
                                "set random seed for all non-deterministic places in the code")
    tf.app.flags.DEFINE_integer("max_para_len", gfc("max_para_len"),
                                "maximum paragraph length")
    tf.app.flags.DEFINE_integer("max_quest_len", gfc("max_quest_len"),
                                "maximum question length")
    tf.app.flags.DEFINE_integer("encoder_n_layers", gfc("encoder_n_layers"),
                                "specify number of layers for encoder-lstm")
    tf.app.flags.DEFINE_integer("patience", gfc("patience"),
                                "number of steps to wait until learning-rate reduduction")
    tf.app.flags.DEFINE_float("decay", gfc("decay"),
                              "decay rate of trainable variables")

    # paths
    tf.app.flags.DEFINE_string("data_dir", gfc("data_dir"), "specify directory where data is stored")
    tf.app.flags.DEFINE_string("out_dir", gfc("out_dir"), "defines path to save pre-processed data")
    tf.app.flags.DEFINE_string("use_out_dir", gfc("use_out_dir"),
        "specify timestamp out_dir name if training should be continued from previous run; otherwise set to None")
    tf.app.flags.DEFINE_string("checkpoint", gfc("checkpoint"),
        "specifies which checkpoint to initialize a model from; default is '' which will load most recent checkpoint")

    # word embeddings
    tf.app.flags.DEFINE_string("use", gfc("use"), "which word embedding to use")
    tf.app.flags.DEFINE_integer("word_vocab_size", gfc("word_vocab_size"), "size of vocabulary")
    tf.app.flags.DEFINE_integer("word_emb_size", gfc("word_emb_size"),
                                "Dimensionality of word embedding (default: 300)")
    tf.app.flags.DEFINE_boolean("train_wemb", gfc("train_wemb"),
                                "whether word embeddings should be trainable or not")

    # char embeddings
    tf.app.flags.DEFINE_boolean("char_emb", gfc("char_emb"),
                                "boolean specifying if char embeddings are used")
    tf.app.flags.DEFINE_integer("char_vocab_size", gfc("char_vocab_size"),
                                "size of character vocabulary")
    tf.app.flags.DEFINE_integer("char_emb_size", gfc("char_emb_size"),
                                "Dimensionality of character embedding (default: 8)")
    tf.app.flags.DEFINE_integer("max_word_len", gfc("max_word_len"), "maximum length of a word")
    tf.app.flags.DEFINE_boolean("highway_layer", gfc("highway_layer"),
                                "attach highway layer to embeddings")
    tf.app.flags.DEFINE_integer("char_hidden_units", gfc("char_hidden_units"),
                                "size of character embedding out")

    # for lstm
    tf.app.flags.DEFINE_integer("rnn_hidden_units", gfc("rnn_hidden_units"), "hidden units for rnn/lstm cells")
    tf.app.flags.DEFINE_float("output_dropout_keep_prob", gfc("output_dropout_keep_prob"),
                              "Output Dropout keep probability (default: 0.5)")
    tf.app.flags.DEFINE_float("input_dropout_keep_prob", gfc("input_dropout_keep_prob"),
                              "Input Dropout keep probability (default: 0.5)")
    tf.app.flags.DEFINE_float("state_dropout_keep_prob", gfc("state_dropout_keep_prob"),
                              "State Dropout keep probability (default: 0.5)")

    # for conv + attention
    tf.app.flags.DEFINE_integer("num_filters", gfc("num_filters"), "number of output filters")
    tf.app.flags.DEFINE_integer("num_heads", gfc("num_heads"), "number of attention heads")

    # for fully connected
    tf.app.flags.DEFINE_float("dropout_keep_prob", gfc("dropout_keep_prob"),
                              "Dropout keep probability (default: 0.5)")
    tf.app.flags.DEFINE_integer("hidden_units", gfc("hidden_units"),
                                "Number of hidden units in softmax regression layer (default:50)")

    # Training parameters
    tf.app.flags.DEFINE_integer("batch_size", gfc("batch_size"), "Batch Size (default: 256)")
    tf.app.flags.DEFINE_integer("num_epochs", gfc("num_epochs"), "Number of training epochs (default: 100)")
    tf.app.flags.DEFINE_integer("evaluate_every", gfc("evaluate_every"),
                                "Evaluate model on dev set after this many steps (default: 100)")
    tf.app.flags.DEFINE_float("learning_rate", gfc("learning_rate"), "learning rate of optimizer")
    tf.app.flags.DEFINE_string("optimiser", gfc("optimiser"), "choice of optimiser (Adam, AdaDelta, SGD)")
    tf.app.flags.DEFINE_float("input_emb_keep_prob", gfc("input_emb_keep_prob"),
                              "keep probability of input embedding dropout; set to 1.0 if not needed")

    # Misc Parameters
    tf.app.flags.DEFINE_boolean("allow_soft_placement", gfc("allow_soft_placement"),
                                "Allow device soft device placement")
    tf.app.flags.DEFINE_boolean("log_device_placement", gfc("log_device_placement"),
                                "Log placement of ops on devices")
    tf.app.flags.DEFINE_float("gpu_fraction", gfc("gpu_fraction"),
                              "defines what percentage of gpu should be used for computation")
    tf.app.flags.DEFINE_boolean("summaries", gfc("summaries"),
                                "boolean which defines whether summaries should be stored")

    tf.app.flags.DEFINE_string("comments", gfc("comments"), "any comments to add to test_results file")

    # dummy flag so tf doesn't break
    tf.app.flags.DEFINE_string('config', 'config.yaml', 'specify config flag; must be first flag to be used!!')

    return FLAGS