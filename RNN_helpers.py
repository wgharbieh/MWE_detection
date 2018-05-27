
import tensorflow as tf

def weight_variable(shape, seed_val):
    initial = tf.truncated_normal(shape, stddev= 0.1, seed=seed_val)
    return tf.Variable(initial)
    #Initialize with small weights


def bias_variable(shape):
    initial = tf.constant(0.1, shape= shape)
    return tf.Variable(initial)
    # Initialize with small positive weights

def tag_feat(tag,tag_set):
    feats = []
    for ts in tag_set:
        if ts == tag:
            feats.append(1)
        else:
            feats.append(0)
    return feats

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int64)
    return length