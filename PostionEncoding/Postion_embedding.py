import numpy as np
import tensorflow as tf


class PositionEmbedding:
    def __init__(self, max_length, embedding_size, batch_size):
        self.max_length = max_length
        self.embedding_size = embedding_size
        self.batch_size = batch_size

    def generate_position_matrix(self):
        position_matrix = np.array([
            [pos / np.power(10000, 2 * (j // 2) / self.embedding_size) for j in range(self.embedding_size)]
            if pos != 0 else np.zeros(self.embedding_size) for pos in range(self.max_length)])
        position_matrix[:, 0::2] = np.sin(position_matrix[:, 0::2])
        position_matrix[:, 1::2] = np.cos(position_matrix[:, 1::2])
        position_embedding = tf.convert_to_tensor(position_matrix, tf.float32)
        return position_embedding

    def construct_word_postion(self):
        word_postion = np.array([[j for j in range(self.max_length)] for _ in range(self.batch_size)])
        word_postion_matrix = tf.convert_to_tensor(word_postion, tf.int32)
        return word_postion_matrix

