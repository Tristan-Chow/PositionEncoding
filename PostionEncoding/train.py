import tensorflow as tf
from Transformer import TransFormer

tf.flags.DEFINE_integer('max_length', 100, "maxlength of sentences")
tf.flags.DEFINE_integer('embedding_size', 512,'model embedding size')
tf.flags.DEFINE_integer('batch_size', 32, 'batch data size')
tf.flags.DEFINE_integer('vocab_size', 1800, 'vocabulary size')

FLAGS = tf.app.flags.FLAGS

with tf.Graph().as_default():
    with tf.Session() as sess:
        TransFormer(FLAGS)
        sess.run(tf.global_variables_initializer())

