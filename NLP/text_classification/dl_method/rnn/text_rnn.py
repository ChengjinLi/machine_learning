#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2017年10月05日

@author: MJ
"""
import tensorflow as tf


def print_variable_info(var):
    """
    print variable info
    """
    print (var.op.name, ' ', var.get_shape().as_list())


class TextRNN(object):
    """
    基于 TextRNN 的文本分类模型
    """
    def __init__(self, input_x, input_y, dropout_keep_prob, config, scope):
        """
        初始化
        """
        with tf.variable_scope('input_layer'):
            self.input_x = input_x
            self.input_y = input_y
            self.dropout_keep_prob = dropout_keep_prob
            self.learning_rate = config.learning_rate
            self.num_step = config.max_sentence_length
            self.num_classes = config.num_classes
            self.hidden_neural_size = config.hidden_neural_size
            self.vocabulary_size = config.vocabulary_size
            self.embedding_dim = config.embedding_dim
            self.hidden_layer_num = config.hidden_layer_num
            if 'adam' == config.optimizer:
                self.optimizer = tf.train.AdamOptimizer
            else:
                self.optimizer = tf.train.GradientDescentOptimizer
            self.learning_rate = config.learning_rate
            self.init_scale = config.init_scale
        # embedding层
        with tf.variable_scope("embedding_layer"):
            # embedding_weight是在训练时得到的嵌入矩阵，通过随机均匀分布进行初始化
            self.embedding_weight = tf.get_variable(
                'weight',
                shape=[self.vocabulary_size, self.embedding_dim],
                initializer=tf.random_uniform_initializer(-1 * self.init_scale, 1 * self.init_scale))
            print_variable_info(self.embedding_weight)
            # tf.nn.embedding_lookup是真正的embedding操作, 查找input_x中所有的ids，获取它们的word vector。batch中的每个sentence的每个word都要查找。
            # 所以得到的结果是一个三维的tensor，[None, sequence_length, embedding_size]
            self.inputs = tf.nn.embedding_lookup(self.embedding_weight, self.input_x)
            print_variable_info(self.inputs)
            # 在训练阶段，对inputs实行一些dropout
            self.inputs = tf.nn.dropout(self.inputs, self.dropout_keep_prob, name="dropout")
            # 输出inputs的shape, 方便调试
            print_variable_info(self.inputs)

        with tf.variable_scope("rnn_layer"):

            def rnn_cell(cell_type):
                if 'rnn' == cell_type:
                    cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_neural_size)
                elif 'lstm' == cell_type:
                    cell = tf.nn.rnn_cell.LSTMCell(self.hidden_neural_size, forget_bias=2.0, state_is_tuple=True)
                elif 'gru' == cell_type:
                    cell = tf.nn.rnn_cell.GRUCell(self.hidden_neural_size)
                else:
                    print ('Input cell_type %s is valid, please input rnn, lstm or gru !' % config.cell_type)
                    exit(0)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
                return cell

            if config.bidirectional:
                # 双向rnn
                if self.hidden_layer_num > 1:
                    self.fw_cells = tf.nn.rnn_cell.MultiRNNCell(
                        [rnn_cell(config.cell_type) for _ in range(self.hidden_layer_num)], state_is_tuple=True)
                    self.bw_cells = tf.nn.rnn_cell.MultiRNNCell(
                        [rnn_cell(config.cell_type) for _ in range(self.hidden_layer_num)], state_is_tuple=True)
                else:
                    self.fw_cells = rnn_cell(config.cell_type)
                    self.bw_cells = rnn_cell(config.cell_type)
                # 获取rnn输出outputs
                self.outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.fw_cells, self.bw_cells, self.inputs, dtype=tf.float32)
                # 获取最后的输出结果output
                self.outputs = tf.concat(self.outputs, axis=2)
                print_variable_info(self.outputs)

            else:
                # 单向rnn
                if self.hidden_layer_num > 1:
                    self.cells = tf.nn.rnn_cell.MultiRNNCell(
                        [rnn_cell(config.cell_type) for _ in range(self.hidden_layer_num)], state_is_tuple=True)
                else:
                    self.cells = rnn_cell(config.cell_type)
                # 获取rnn单元输出outputs
                self.outputs, _ = tf.nn.dynamic_rnn(self.cells, self.inputs, dtype=tf.float32)
                print_variable_info(self.outputs)

            if 'mean' == config.output_method:
                # 第一种方法,所有outputs求平均, 收敛容易点，顺序要求弱一点，词频特性会比较明显
                self.output = tf.reduce_mean(self.outputs, axis=1)
            elif 'last' == config.output_method:
                # 第二种方法,取最后一次的output, 会更强调序列的顺序，收敛也会难一点
                self.output = self.outputs[:, -1, :]
            else:
                print ('Input output_method %s is valid, please input mean or last !' % config.output_method)
                exit(0)
            print_variable_info(self.output)

        # 输出层
        with tf.variable_scope("output_layer"):
            if config.bidirectional:
                total_hidden_neural_size = 2 * self.hidden_neural_size
            else:
                total_hidden_neural_size = self.hidden_neural_size
            self.softmax_weight = tf.get_variable("weight", [total_hidden_neural_size, self.num_classes],
                                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            print_variable_info(self.softmax_weight)
            self.softmax_bias = tf.get_variable("bias", [self.num_classes], dtype=tf.float32)

            # 计算所有类别的分数, scores的shape为[batch, num_classes]
            self.scores = tf.add(tf.matmul(self.output, self.softmax_weight), self.softmax_bias, name="scores")
            print_variable_info(self.scores)

            # 计算交叉熵的平均损失
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores + 1e-10)
            self.loss = tf.reduce_mean(self.cross_entropy, name="loss")
            tf.summary.scalar("loss", self.loss)

            # 计算预测类别,分数最大对应的类别，因此argmax的时候是选取每行的max，dimention=1,[batch, 1]
            self.prediction = tf.argmax(self.scores, 1, name="prediction")

            # 计算精度
            # tf.equal(x, y)返回的是一个bool tensor，如果xy对应位置的值相等就是true，否则false。得到的tensor是[batch, 1]的。
            self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            # tf.cast(x, dtype)将bool tensor转化成float类型的tensor，方便计算
            # tf.reduce_mean()本身输入的就是一个float类型的vector（元素要么是0.0，要么是1.0），
            # 直接对这样的vector计算mean得到的就是accuracy，不需要指定reduction_indices
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="accuracy")
            tf.summary.scalar("accuracy", self.accuracy)

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        optimizer = self.optimizer(self.learning_rate)
        self.train_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.train_vars), config.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, self.train_vars), global_step=self.global_step)
        grad_summaries = []
        for grad, var in zip(grads, self.train_vars):
            if grad is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(var.name), grad)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(var.name), tf.nn.zero_fraction(grad))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))

