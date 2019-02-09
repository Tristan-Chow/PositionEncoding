import tensorflow as tf
from Postion_embedding import PositionEmbedding


class TransFormer:
    def __init__(self, config):
        self.max_length = config.max_length
        self.embedding_size = config.embedding_size
        self.batch_size = config.batch_size
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_length], name='input_x')
        self.vocab_size = config.vocab_size

        pebd = PositionEmbedding(self.max_length, self.embedding_size, self.batch_size)

        # Word embedding
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                         name="embedding")
            embed = tf.nn.embedding_lookup(self.embedding, self.input_x)

            # Position encoding
            self.postion_embedding = pebd.generate_position_matrix()
            self.word_postion = pebd.construct_word_postion()
            p_embed = tf.nn.embedding_lookup(self.postion_embedding, self.word_postion)

        inputs = tf.add(embed, p_embed, name='postion and word embeding')
