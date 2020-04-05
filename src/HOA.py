"""define Deep Attentional Crossing Model"""
import math
import numpy as np
import tensorflow as tf
from src.DACN import DeepAttentionalCrossingModel

__all__ = ["HigherOrderAttentionModel"]


class HigherOrderAttentionModel(DeepAttentionalCrossingModel):
    def _build_graph(self, hparams):
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)
        self.layer_keeps = tf.placeholder(tf.float32)
        with tf.variable_scope("DACN") as scope:
            with tf.variable_scope("embedding", initializer=self.initializer) as escope:
                self.embedding = tf.get_variable(name='embedding_layer',
                                                 shape=[hparams.FEATURE_COUNT, hparams.embedding_dim],
                                                 dtype=tf.float32)
                self.embed_params.append(self.embedding)
                self.embed_out, embed_layer_size = self._build_embedding(hparams)
            logit = self._build_HOA(hparams, self.embed_out, reduce='sum_pooling', do_projection=True, has_residual=True)

            return logit
