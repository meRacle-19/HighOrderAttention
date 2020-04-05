"""define Factorization-Machine based Neural Network Model"""
import math
import numpy as np
import tensorflow as tf
from src.base_model import BaseModel

__all__ = ["DeepCrossModel"]


class DeepCrossModel(BaseModel):
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
            logit = self._build_dnn(hparams, self.embed_out, embed_layer_size)
            logit = tf.add(logit, self._build_cross(hparams, self.embed_out, embed_layer_size))
            return logit

    def _build_embedding(self, hparams):
        fm_sparse_index = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                          self.iterator.dnn_feat_values,
                                          self.iterator.dnn_feat_shape)
        fm_sparse_weight = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                           self.iterator.dnn_feat_weights,
                                           self.iterator.dnn_feat_shape)
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(self.embedding,
                                                            fm_sparse_index,
                                                            fm_sparse_weight,
                                                            combiner="sum")
        embedding = tf.reshape(w_fm_nn_input_orgin, [-1, hparams.dim * hparams.FIELD_COUNT])
        embedding_size = hparams.FIELD_COUNT * hparams.dim
        return embedding, embedding_size

    def _build_dnn(self, hparams, embed_out, embed_layer_size):
        w_fm_nn_input = embed_out
        last_layer_size = embed_layer_size
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(w_fm_nn_input)
        with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.layer_sizes):
                curr_w_nn_layer = tf.get_variable(name='w_nn_layer' + str(layer_idx),
                                                  shape=[last_layer_size, layer_size],
                                                  dtype=tf.float32)
                curr_b_nn_layer = tf.get_variable(name='b_nn_layer' + str(layer_idx),
                                                  shape=[layer_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer())
                tf.summary.histogram("nn_part/" + 'w_nn_layer' + str(layer_idx),
                                     curr_w_nn_layer)
                tf.summary.histogram("nn_part/" + 'b_nn_layer' + str(layer_idx),
                                     curr_b_nn_layer)
                curr_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx],
                                                       curr_w_nn_layer,
                                                       curr_b_nn_layer)
                activation = hparams.activation[idx]
                curr_hidden_nn_layer = self._active_layer(logit=curr_hidden_nn_layer,
                                                          activation=activation)
                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size
                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)

            w_nn_output = tf.get_variable(name='w_nn_output',
                                          shape=[last_layer_size, 1],
                                          dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                          shape=[1],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
            tf.summary.histogram("nn_part/" + 'w_nn_output' + str(layer_idx),
                                 w_nn_output)
            tf.summary.histogram("nn_part/" + 'b_nn_output' + str(layer_idx),
                                 b_nn_output)
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            nn_output = tf.nn.xw_plus_b(hidden_nn_layers[-1], w_nn_output, b_nn_output)
            return nn_output
    def _build_cross(self, hparams, embed_out, embed_layer_size):
        ## x0 and x size is [batch, d]
        ## w  and b size is [d, 1]
        x0 = tf.expand_dims(embed_out, 2)  # [batch, d, 1]
        x0 = tf.transpose(x0, [0, 2, 1])
        x = embed_out
        for layer_idx, layer_size in enumerate(hparams.cross_layer_sizes):
            x = tf.expand_dims(x, 2)  # [batch, d, 1]
            w_cross_layer = tf.get_variable(name='w_cross_layer' + str(layer_idx),
                                              shape=[embed_layer_size],
                                              dtype=tf.float32)
            b_cross_layer = tf.get_variable(name='b_cross_layer' + str(layer_idx),
                                              shape=[embed_layer_size],
                                              dtype=tf.float32,
                                              initializer=tf.zeros_initializer())
            dot = tf.matmul(x, x0)  # [batch, d, d] = batch x {[dx1]x[1xd]
            x = tf.tensordot(dot, w_cross_layer, 1) + b_cross_layer  ## 写法来源maxnet
        w_cross_layer = tf.get_variable(name='w_out_layer',
                                        shape=[embed_layer_size,1],
                                        dtype=tf.float32)
        b_cross_layer = tf.get_variable(name='b_out_layer',
                                        shape=[1],
                                        dtype=tf.float32,
                                        initializer=tf.zeros_initializer())
        return tf.nn.xw_plus_b(x, w_cross_layer, b_cross_layer)
