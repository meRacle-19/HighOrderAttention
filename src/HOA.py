"""define Factorization-Machine based Neural Network Model"""
import math
import numpy as np
import tensorflow as tf
from src.base_model import BaseModel

__all__ = ["HighOrderAttentionModel"]


class HighOrderAttentionModel(BaseModel):
    """define Factorization-Machine based Neural Network Model"""

    def _build_graph(self, hparams):
        self.keep_prob_train = 1 - np.array(hparams.layer_dropout)
        self.keep_prob_test = np.ones_like(hparams.layer_dropout)
        self.layer_keeps = tf.placeholder(tf.float32)
        with tf.variable_scope("exDeepFm") as scope:
            with tf.variable_scope("embedding", initializer=self.initializer) as escope:
                self.embedding = tf.get_variable(name='embedding_layer',
                                                 shape=[hparams.FEATURE_COUNT, hparams.embedding_dim],
                                                 dtype=tf.float32)
                self.embed_params.append(self.embedding)
                self.embed_out, embed_layer_size = self._build_embedding(hparams)
            logit = tf.zeros([1])
            # logit = self._build_linear(hparams)
            # logit = tf.add(logit, self._build_fm(hparams))
            logit = tf.add(logit, self._build_HOA(hparams, self.embed_out))
            # logit = tf.add(logit, self._build_extreme_FM_quick(hparams, embed_out))
            #logit = tf.add(logit, self._build_dnn(hparams, self.embed_out, embed_layer_size))
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
        embedding = tf.reshape(w_fm_nn_input_orgin, [-1, hparams.embedding_dim * hparams.FIELD_COUNT])
        embedding_size = hparams.FIELD_COUNT * hparams.embedding_dim
        return embedding, embedding_size

    def _build_linear(self, hparams):
        with tf.variable_scope("linear_part", initializer=self.initializer) as scope:
            w_linear = tf.get_variable(name='w',
                                       shape=[hparams.FEATURE_COUNT, 1],
                                       dtype=tf.float32)
            b_linear = tf.get_variable(name='b',
                                       shape=[1],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer())
            x = tf.SparseTensor(self.iterator.fm_feat_indices,
                                self.iterator.fm_feat_values,
                                self.iterator.fm_feat_shape)
            linear_output = tf.add(tf.sparse_tensor_dense_matmul(x, w_linear), b_linear)
            self.layer_params.append(w_linear)
            self.layer_params.append(b_linear)
            tf.summary.histogram("linear_part/w", w_linear)
            tf.summary.histogram("linear_part/b", b_linear)
            return linear_output

    def _build_fm(self, hparams):
        with tf.variable_scope("fm_part") as scope:
            x = tf.SparseTensor(self.iterator.fm_feat_indices,
                                self.iterator.fm_feat_values,
                                self.iterator.fm_feat_shape)
            xx = tf.SparseTensor(self.iterator.fm_feat_indices,
                                 tf.pow(self.iterator.fm_feat_values, 2),
                                 self.iterator.fm_feat_shape)
            fm_output = 0.5 * tf.reduce_sum(
                tf.pow(tf.sparse_tensor_dense_matmul(x, self.embedding), 2) - \
                tf.sparse_tensor_dense_matmul(xx,
                                              tf.pow(self.embedding, 2)), 1,
                keep_dims=True)
            return fm_output


    def multihead_attention(self, queries, keys, values, base_values, num_units=None, num_heads=1,
                            attention_idx=1, has_residual=True):
        def normalize(inputs, epsilon=1e-8):
            '''
            Applies layer normalization
            Args:
                inputs: A tensor with 2 or more dimensions
                epsilon: A floating number to prevent Zero Division
            Returns:
                A tensor with the same shape and data dtype
            '''
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta
            return outputs

        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # [-1, field_num, num_units]
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(values, num_units, activation=tf.nn.relu)
        B = tf.layers.dense(base_values, num_units, activation=tf.nn.relu)
        if has_residual:
            V_res = tf.layers.dense(values, num_units, activation=tf.nn.relu)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # [-1, field_num, num_units/num_heads]
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
        B_ = tf.concat(tf.split(B, num_heads, axis=2), axis=0)

        # Multiplication
        weights = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        # Scale
        weights = weights / (K_.get_shape().as_list()[-1] ** 0.5)
        # Activation
        weights = tf.nn.softmax(weights)
        # Dropouts
        #weights = tf.layers.dropout(weights, rate=1 - dropout_keep_prob)
        weights = self._dropout(weights, attention_idx)
        # Weighted sum with higher order
        field_num = B_.get_shape().as_list()[1]
        B_sec_ord = tf.reshape(tf.tile(B_, [1, field_num, 1]), [-1, field_num, field_num, int(num_units/num_heads)])
        B_sec_ord = tf.multiply(B_sec_ord, tf.expand_dims(V_, 2))   # [-1, field_num, field_num, dim]
        # B_sec_ord = normalize(B_sec_ord)
        outputs = tf.reshape(tf.matmul(tf.expand_dims(weights, 2), B_sec_ord), [-1, field_num, int(num_units/num_heads)])  # [-1, field_num, dim]
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        # Residual connection
        if has_residual:
            outputs += V_res
        outputs = tf.nn.relu(outputs)
        outputs = self._dropout(outputs, attention_idx)
        # Normalize
        # outputs = normalize(outputs)

        return outputs

    def _build_HOA(self, hparams, nn_input):
        field_num = hparams.FIELD_COUNT
        nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), hparams.embedding_dim])
        nn_outputs = [nn_input]

        with tf.variable_scope("hoa_part", initializer=self.initializer) as scope:
            for idx in range(hparams.orders):
                nn_outputs.append(self.multihead_attention(queries=nn_input,
                                                     keys=nn_outputs[-1],
                                                     values=nn_outputs[-1],
                                                     base_values=nn_input,
                                                     num_units=hparams.attention_embedding_dims[idx],
                                                     num_heads=hparams.attention_heads[idx],
                                                     attention_idx=idx,
                                                     has_residual=True))
            # reduce = 'dnn'
            reduce = 'sum_pooling'
            # reduce = 'sum'
            if reduce in ['dnn']:
                combine_outputs = []
                for idx, output in enumerate(nn_outputs[1:], start=0):
                    dim_input_nn = int(field_num) * int(hparams.attention_embedding_dims[idx])
                    input_nn = tf.reshape(output, shape=[-1, dim_input_nn])
                    w_nn_hidden = tf.get_variable(name='w_hidden'+str(idx),
                                                   shape=[dim_input_nn, hparams.layer_sizes[idx]],
                                                   dtype=tf.float32)
                    b_nn_hidden = tf.get_variable(name='b_hidden'+str(idx),
                                                   shape=[hparams.layer_sizes[idx]],
                                                   dtype=tf.float32,
                                                   initializer=tf.zeros_initializer())
                    self.layer_params.append(w_nn_hidden)
                    self.layer_params.append(b_nn_hidden)
                    HOA_hidden = tf.nn.xw_plus_b(input_nn, w_nn_hidden, b_nn_hidden)
                    HOA_hidden = self._active_layer(logit=HOA_hidden,
                                                   scope=scope,
                                                   activation="relu",
                                                   layer_idx=idx)
                    combine_outputs.append(HOA_hidden)

                combined = tf.concat(combine_outputs, axis=1)
                w_nn_output = tf.get_variable(name='w_nn_output',
                                              shape=[combined.get_shape().as_list()[-1], 1],
                                              dtype=tf.float32)
                b_nn_output = tf.get_variable(name='b_nn_output',
                                              shape=[1],
                                              dtype=tf.float32,
                                              initializer=tf.zeros_initializer())
                self.layer_params.append(w_nn_output)
                self.layer_params.append(b_nn_output)
                logit = tf.nn.xw_plus_b(combined, w_nn_output, b_nn_output)
                return logit
            elif reduce in ['sum_pooling']:
                combine_outputs = []
                for idx, output in enumerate(nn_outputs[1:], start=0):
                    combine_outputs.append(tf.reduce_sum(output, axis=1))
                combined = tf.concat(combine_outputs, axis=1)
                w_nn_output = tf.get_variable(name='w_nn_output',
                                              shape=[combined.get_shape().as_list()[-1], 1],
                                              dtype=tf.float32)
                b_nn_output = tf.get_variable(name='b_nn_output',
                                              shape=[1],
                                              dtype=tf.float32,
                                              initializer=tf.zeros_initializer())
                self.layer_params.append(w_nn_output)
                self.layer_params.append(b_nn_output)
                logit = tf.nn.xw_plus_b(combined, w_nn_output, b_nn_output)
                return logit
            elif reduce in ['sum']:
                combine_outputs = []
                for idx, output in enumerate(nn_outputs[1:], start=0):
                    combine_outputs.append(tf.reduce_sum(output, axis=0))
                combined = tf.concat(combine_outputs, axis=1)
                logit = tf.reduce_sum(combined, axis=0)
                return logit

    def _build_dnn(self, hparams, embed_out, embed_layer_size):
        """
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
        w_fm_nn_input = tf.reshape(w_fm_nn_input_orgin, [-1, hparams.embedding_dim * hparams.FIELD_COUNT])
        last_layer_size = hparams.FIELD_COUNT * hparams.embedding_dim
        """
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
                scope = "nn_part" + str(idx)
                activation = hparams.activation[idx]
                curr_hidden_nn_layer = self._active_layer(logit=curr_hidden_nn_layer,
                                                          scope=scope,
                                                          activation=activation,
                                                          layer_idx=idx)
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
