"""define Factorization-Machine based Neural Network Model"""
import math
import numpy as np
import tensorflow as tf
from src.deepCross import DeepCrossModel

__all__ = ["CrossModel"]


class CrossModel(DeepCrossModel):
    """define Factorization-Machine based Neural Network Model"""

    def _build_graph(self, hparams):
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)
        self.layer_keeps = tf.placeholder(tf.float32)
        with tf.variable_scope("exDeepFm") as scope:
            with tf.variable_scope("embedding", initializer=self.initializer) as escope:
                self.embedding = tf.get_variable(name='embedding_layer',
                                                 shape=[hparams.FEATURE_COUNT, hparams.dim],
                                                 dtype=tf.float32)
                self.embed_params.append(self.embedding)
                self.embed_out, embed_layer_size = self._build_embedding(hparams)
            logit = self._build_cross(hparams, self.embed_out, embed_layer_size)
            return logit
