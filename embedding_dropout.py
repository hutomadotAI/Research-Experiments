import tensorflow as tf


def get_embedding_dropout_mask(keep_prob, shape, input, seed=3435, is_train=None):
    if is_train == False:
        mask_dense = tf.ones(shape)
    else:
        unique_words, unique_word_idxs = tf.unique(tf.reshape(input, [-1]))
        idx2 = tf.tile(tf.reshape(tf.range(shape[1]), [1, -1]), [tf.shape(unique_words)[0], 1])
        idx2 = tf.reshape(idx2, [-1, 1])
        idx2 = tf.cast(idx2, tf.int64)
        if is_train is not None:
            keep_prob = tf.cond(is_train, lambda: keep_prob, lambda: 1.0)
        keep_prob = tf.convert_to_tensor(keep_prob)
        probs = keep_prob + tf.random_uniform(tf.shape(unique_words), maxval=1.0, seed=seed, dtype=tf.float32)
        probs = tf.floor(probs)
        probs2 = tf.tile(tf.reshape(probs, [tf.shape(unique_words)[0], 1]), [1, shape[1]])
        probs2 = tf.reshape(probs2, [-1])
        idx1 = tf.tile(tf.reshape(unique_words, [-1, 1]), [1, shape[1]])
        idx1 = tf.reshape(idx1, [-1, 1])
        idx1 = tf.cast(idx1, tf.int64)
        idx = tf.concat([idx1, idx2], 1)
        mask = tf.SparseTensor(idx, probs2, shape)
        mask_reo = tf.sparse_reorder(mask)
        mask_dense = tf.sparse_tensor_to_dense(mask_reo)
    return mask_dense #, probs2, unique_words, idx


if __name__ == "__main__":
    input = tf.constant([[4,6,2,1,4,9,2],[10,2,2,6,4,9,9]])
    shape = [12, 8]
    keep_prob = 1.0

    m = get_embedding_dropout_mask(keep_prob, shape, input)

    sess = tf.Session()
    inp, out = sess.run([input, m])

    print("input: {}, shape: {}, keep_prob: {}\nunique_words: {}\nprobs: {}\nmask: {}".format(
        inp, shape, keep_prob, out[2], out[1], out[0]))