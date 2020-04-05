"""define Deep Attentional Crossing Model"""
import math
import numpy as np
import tensorflow as tf
from src.base_model import BaseModel

__all__ = ["DeepAttentionalCrossingModel"]


class DeepAttentionalCrossingModel(BaseModel):
    def _build_graph(self, hparams):
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)
        self.layer_keeps = tf.placeholder(tf.float32)
        self.weights = None
        with tf.variable_scope("DACN") as scope:
            with tf.variable_scope("embedding", initializer=self.initializer) as escope:
                self.embedding = tf.get_variable(name='embedding_layer',
                                                 shape=[hparams.FEATURE_COUNT, hparams.embedding_dim],
                                                 dtype=tf.float32)
                self.embed_params.append(self.embedding)
                self.embed_out, embed_layer_size = self._build_embedding(hparams)
            logit = []
            #logit = tf.add(logit, self._build_linear(hparams))
            logit.append(self._build_HOA(hparams, self.embed_out, reduce='sum_pooling', do_projection=True, has_residual=True))
            logit.append(self._dnn(self.embed_out, hparams.dnn_layer_sizes, hparams.dnn_layer_activations, name='dnn_part'))

            return self._dnn(tf.concat(logit,axis=1), [1], ['identity'], name="combination")

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

    def _dnn(self, input_embed, layer_sizes=[], activations=[], name=None):
        last_layer_size = input_embed.get_shape().as_list()[-1]
        hidden_nn_layers = [input_embed]
        with tf.variable_scope(name, initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(layer_sizes):
                w_name = 'w_layer_{}'.format(idx)
                b_name = 'b_layer_{}'.format(idx)
                curr_w_nn_layer = tf.get_variable(name=w_name,
                                                  shape=[last_layer_size, layer_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.truncated_normal_initializer(0, 1/(last_layer_size+layer_size)))
                curr_b_nn_layer = tf.get_variable(name=b_name,
                                                  shape=[layer_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer())
                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)
                tf.summary.histogram(w_name, curr_w_nn_layer)
                tf.summary.histogram(b_name, curr_b_nn_layer)
                curr_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[idx],
                                                       curr_w_nn_layer,
                                                       curr_b_nn_layer)

                activation = activations[idx]
                curr_hidden_nn_layer = self._active_layer(logit=curr_hidden_nn_layer,
                                                          activation=activation)
                hidden_nn_layers.append(curr_hidden_nn_layer)
                last_layer_size = layer_size
            return hidden_nn_layers[-1]

    def HigherOrderAttention(self, hparams, queries, keys, values, bases,
                                layer_idx=1, has_residual=True, do_projection=None):
        # Linear projections
        dim_in = keys.get_shape().as_list()[-1]
        dim_base = bases.get_shape().as_list()[-1]
        field_num = queries.get_shape().as_list()[-2]
        if do_projection:
            num_units = hparams.cross_layer_dims[layer_idx]

            w_query = tf.get_variable(name='projection_query_w_{}'.format(layer_idx),
                                      shape=[dim_base, num_units],
                                      dtype=tf.float32)
            b_query = tf.get_variable(name='projection_query_b_{}'.format(layer_idx),
                                      shape=[num_units],
                                      dtype=tf.float32,
                                      initializer=tf.zeros_initializer())
            w_base_value = tf.get_variable(name='projection_base_w_{}'.format(layer_idx),
                                           shape=[dim_base, num_units],
                                           dtype=tf.float32)
            b_base_value = tf.get_variable(name='projection_base_b_{}'.format(layer_idx),
                                           shape=[num_units],
                                           dtype=tf.float32,
                                           initializer=tf.zeros_initializer())
            w_key = tf.get_variable(name='projection_key_w_{}'.format(layer_idx),
                                    shape=[dim_in, num_units],
                                    dtype=tf.float32)
            b_key = tf.get_variable(name='projection_key_b_{}'.format(layer_idx),
                                    shape=[num_units],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
            w_value = tf.get_variable(name='projection_value_w_{}'.format(layer_idx),
                                      shape=[dim_in, num_units],
                                      dtype=tf.float32)
            b_value = tf.get_variable(name='projection_value_b_{}'.format(layer_idx),
                                      shape=[num_units],
                                      dtype=tf.float32,
                                      initializer=tf.zeros_initializer())
            self.layer_params.append(w_key)
            self.layer_params.append(b_key)
            self.layer_params.append(w_value)
            self.layer_params.append(b_value)
            self.layer_params.append(w_query)
            self.layer_params.append(b_query)
            self.layer_params.append(w_base_value)
            self.layer_params.append(b_base_value)

            queries = tf.reshape(queries, [-1, dim_base])
            bases = tf.reshape(bases, [-1, dim_base])
            keys = tf.reshape(keys, [-1, dim_in])
            values = tf.reshape(values, [-1, dim_in])

            Q = tf.nn.xw_plus_b(queries, w_query, b_query)
            Q = self._active_layer(logit=Q,
                                   activation="relu")
            B = tf.nn.xw_plus_b(bases, w_base_value, b_base_value)
            B = self._active_layer(logit=B,
                                   activation="relu")
            K = tf.nn.xw_plus_b(keys, w_key, b_key)
            K = self._active_layer(logit=K,
                                   activation="relu")
            V = tf.nn.xw_plus_b(values, w_value, b_value)
            V = self._active_layer(logit=V,
                                   activation="relu")
            Q = tf.reshape(Q, [-1, field_num, num_units])
            B = tf.reshape(B, [-1, field_num, num_units])
            K = tf.reshape(K, [-1, field_num, num_units])
            V = tf.reshape(V, [-1, field_num, num_units])
        else:
            num_units = queries.get_shape().as_list()[-1]
            Q = queries
            B = bases
            K = keys
            V = values

        # Split and concat
        num_heads = hparams.cross_layer_heads[layer_idx]
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # [-1, field_num, num_units/num_heads]
        B_ = tf.concat(tf.split(B, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        # Multiplication
        weights = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        # Scale
        weights = weights / (K_.get_shape().as_list()[-1] ** 0.5)
        if (layer_idx == 0):
            self.weights = weights
        # Activation
        weights = tf.nn.softmax(weights)
        # Weighted sum with higher order
        V_expand = tf.reshape(tf.tile(V_, [1, field_num, 1]), [-1, field_num, field_num, int(num_units/num_heads)])
        BV_sec_ord = tf.multiply(V_expand, tf.expand_dims(B_, 2))   # [-1, field_num, field_num, dim]
        #BV_sec_ord = self._normalize(BV_sec_ord)
        outputs = tf.reshape(tf.matmul(tf.expand_dims(weights, 2), BV_sec_ord), [-1, field_num, int(num_units/num_heads)])  # [-1, field_num, dim]
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        # Residual connection
        if has_residual:
            V_res = tf.layers.dense(tf.reshape(values,[-1, dim_in]), num_units, activation=tf.nn.relu)
            outputs += tf.reshape(V_res,[-1, field_num, num_units])
        outputs = self._activate(outputs,'relu')
        outputs = self._dropout(outputs)
        # Normalize
        outputs = self._normalize(outputs)
        return outputs

    def _build_HOA(self, hparams, nn_input, has_residual=False, do_projection=False, reduce=None):
        field_num = hparams.FIELD_COUNT
        nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), hparams.embedding_dim])
        nn_outputs = [nn_input]

        with tf.variable_scope("HOA", initializer=self.initializer) as scope:
            for idx in range(hparams.orders):
                nn_outputs.append(self.HigherOrderAttention(hparams,
                                                            queries=nn_input,
                                                            keys=nn_outputs[-1],
                                                            values=nn_outputs[-1],
                                                            bases=nn_input,
                                                            layer_idx=idx,
                                                            do_projection=do_projection,
                                                            has_residual=has_residual))
            if reduce in ['mlp_pooling']:
                combine_outputs = []
                for idx, output in enumerate(nn_outputs[0:], start=0):
                    dim_input_nn = int(field_num) * int(output.get_shape().as_list()[-1])
                    input_nn = tf.reshape(output, shape=[-1, dim_input_nn])
                    w_nn_hidden = tf.get_variable(name='reduce_w_layer_{}'.format(idx),
                                                   shape=[dim_input_nn, hparams.reduce_layer_sizes[2*idx]],
                                                   dtype=tf.float32)
                    b_nn_hidden = tf.get_variable(name='reduce_b_layer_{}'.format(idx),
                                                   shape=[hparams.reduce_layer_sizes[2*idx]],
                                                   dtype=tf.float32,
                                                   initializer=tf.zeros_initializer())
                    self.layer_params.append(w_nn_hidden)
                    self.layer_params.append(b_nn_hidden)
                    w_nn_hidden_2 = tf.get_variable(name='reduce_w2_layer_{}'.format(idx),
                                                  shape=[hparams.reduce_layer_sizes[2*idx], hparams.reduce_layer_sizes[2*idx+1]],
                                                  dtype=tf.float32)
                    b_nn_hidden_2 = tf.get_variable(name='reduce_b2_layer_{}'.format(idx),
                                                  shape=[hparams.reduce_layer_sizes[2*idx+1]],
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer())
                    self.layer_params.append(w_nn_hidden_2)
                    self.layer_params.append(b_nn_hidden_2)
                    HOA_hidden = tf.nn.xw_plus_b(input_nn, w_nn_hidden, b_nn_hidden)
                    HOA_hidden = self._active_layer(logit=HOA_hidden,
                                                   activation=hparams.reduce_layer_activations[2*idx])
                    HOA_hidden = tf.nn.xw_plus_b(HOA_hidden, w_nn_hidden_2, b_nn_hidden_2)
                    HOA_hidden = self._active_layer(logit=HOA_hidden,
                                                    activation=hparams.reduce_layer_activations[2*idx+1])
                    combine_outputs.append(HOA_hidden)

                combined = tf.concat(combine_outputs, axis=1)
            elif reduce in ['max_pooling']:
               combine_outputs = []
               for output in nn_outputs[0:]:
                   combine_outputs.append(tf.reduce_max(output, reduction_indices=[2]))
               combined = tf.concat(combine_outputs, axis=1)
            elif reduce in ['sum_pooling']:
                combine_outputs = []
                for output in nn_outputs[0:]:
                    combine_outputs.append(tf.reduce_sum(output, axis=1))
                combined = tf.concat(combine_outputs, axis=1)
            elif reduce in ['cnn_pooling']:
                combine_outputs = []
                for poly,output in enumerate(nn_outputs,start=2):
                    combine_cnn = []
                    for width in list(range(1,4)):
                        embed_dim = output.get_shape().as_list()[-1]
                        output_1 = tf.reshape(output,[-1, field_num, embed_dim, 1])
                        conv_filter = tf.get_variable(name='conv_layer_{}_{}'.format(poly,width),
                                                        shape=[1, embed_dim, 1, 1],
                                                        dtype=tf.float32)
                        output_2 = tf.nn.conv2d(output_1, conv_filter, [1,1,1,1], padding='VALID')
                        output_3 = tf.nn.max_pool(output_2, [1, field_num, 1, 1], [1, 1, 1, 1], padding='VALID')
                        output_4 = tf.reshape(output_3, [-1, 1])
                        combine_cnn.append(output_4)
                    combine_outputs = tf.concat(combine_cnn, axis=1)
                combined = tf.concat(combine_outputs, axis=1)
            elif reduce in ['sum']:
                combine_outputs = []
                for idx, output in enumerate(nn_outputs[1:], start=0):
                    combine_outputs.append(tf.reduce_sum(output, axis=1))
                combined = tf.concat(combine_outputs, axis=1)
                combined = tf.reduce_sum(combined, axis=1)
            return self._dnn(combined, hparams.layer_sizes, hparams.layer_activations, name="HOA_output")
