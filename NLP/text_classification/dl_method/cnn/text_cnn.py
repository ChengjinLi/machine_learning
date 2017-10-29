#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2017年10月03日

@author: MJ
"""
import tensorflow as tf


def print_variable_info(var):
    """
    print variable info
    """
    print (var.op.name, ' ', var.get_shape().as_list())


class TextCNN(object):
    """
    搭建一个用于文本数据的CNN模型
    """
    def __init__(self, input_x, input_y, dropout_keep_prob, config, scope):
        """
        初始化
        """
        with tf.variable_scope('input_layer'):
            self.input_x = input_x
            self.input_y = input_y
            self.filter_sizes = list(map(int, config.filter_sizes.split(",")))
            self.learning_rate = config.learning_rate
            self.sentence_length = config.max_sentence_length
            self.num_classes = config.num_classes
            self.num_filters = config.num_filters
            self.vocabulary_size = config.vocabulary_size
            self.embedding_dim = config.embedding_dim
            if 'relu' == config.activation:
                self.activation = tf.nn.relu
            else:
                self.activation = tf.nn.tanh
            if 'adam' == config.optimizer:
                self.optimizer = tf.train.AdamOptimizer
            else:
                self.optimizer = tf.train.GradientDescentOptimizer
            self.l2_reg_lambda = config.l2_reg_lambda
            # L2正规化损失记录（可选）
            l2_loss = tf.constant(0.0)
            self.init_scale = config.init_scale
        # 嵌入层
        # tf.name_scope("embedding"),把所有操作加到命名为embedding的顶层节点，用于可视化网络视图
        with tf.variable_scope("embedding"):
            # embedding_weight是在训练时得到的嵌入矩阵，通过随机均匀分布进行初始化
            self.embedding_weight = tf.get_variable(
                'weight',
                shape=[self.vocabulary_size,  self.embedding_dim],
                initializer=tf.random_uniform_initializer(-1 * self.init_scale, 1 * self.init_scale))
            print_variable_info(self.embedding_weight)
            # tf.nn.embedding_lookup是真正的embedding操作, 查找input_x中所有的ids，获取它们的word vector。batch中的每个sentence的每个word都要查找。
            # 所以得到的结果是一个三维的tensor，[None, sequence_length, embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding_weight, self.input_x)
            print_variable_info(self.embedded_chars)
            # 因为卷积操作conv2d的input要求4个维度的tensor, 所以需要给embedding结果增加一个维度来适应conv2d的input要求
            # 传入的-1表示在最后位置插入, 得到[None, sequence_length, embedding_size, 1]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            print_variable_info(self.embedded_chars_expanded)

        # 对不同大小的filter建立不同的卷积层+最大池层
        # pooled_outputs用于存储池化之后的结果,用于后面的全连接层
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # 卷积层
                filter_shape = [filter_size, self.embedding_dim, 1, self.num_filters]
                # weight是卷积的输入矩阵
                # 利用truncated_normal生成截断正态分布随机数, 尺寸是filter_shape, 均值mean, 标准差stddev,
                # 不过只保留[mean-2*stddev,mean+2*stddev]范围内的随机数
                weight = tf.get_variable(
                    'weight',
                    shape=filter_shape,
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
                print_variable_info(weight)
                # bias是卷积的输入偏置量
                bias = tf.get_variable(name='bias', shape=[self.num_filters], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.1))

                # 卷积操作, “VALID”表示使用narrow卷积，得到的结果大小为[batch, sequence_length - filter_size + 1, 1, num_filters]
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    weight,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                print_variable_info(conv)
                # h: 卷积的结果加上偏置项b，之后应用ReLU函数处理的结果
                # h的大小为[batch, sequence_length - filter_size + 1, 1, num_filters]
                h = self.activation(tf.nn.bias_add(conv, bias), name="activation")
                # 用max-pooling处理上层的输出结果,每一个卷积结果
                # pooled的大小为[batch, 1, 1, num_filters]，
                print_variable_info(h)
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sentence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                print_variable_info(pooled)
                pooled_outputs.append(pooled)

        with tf.variable_scope('full_connected_layer'):
            # 全连接输出层
            # 将上面的pooling层输出全连接到输出层
            num_filters_total = self.num_filters * len(self.filter_sizes)
            # 把相同filter_size的所有pooled结果concat起来，再将不同的filter_size之间的结果concat起来
            # tf.concat按某一维度进行合并, h_pool的大小为[batch, 1, 1, num_filters_total]
            self.h_pool = tf.concat(pooled_outputs, 3)
            print_variable_info(self.h_pool)
            # h_pool_flat也就是[batch, num_filters_total]维的tensor。
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            print_variable_info(self.h_pool_flat)
            # 在训练阶段，对max-pooling layer的输出实行一些dropout，以概率p激活，激活的部分传递给softmax层。
            self.h_drop = tf.nn.dropout(self.h_pool_flat, dropout_keep_prob)

        # 输出层
        with tf.variable_scope('output_layer'):
            self.softmax_weight = tf.get_variable('weight', shape=[num_filters_total, self.num_classes],
                                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            print_variable_info(self.softmax_weight)
            self.softmax_bias = tf.get_variable(name='bias', shape=[self.num_classes], dtype=tf.float32,
                                                initializer=tf.constant_initializer(0.1))

            l2_loss += tf.nn.l2_loss(self.softmax_weight)
            l2_loss += tf.nn.l2_loss(self.softmax_bias)

            # 计算所有类别的分数, scores的shape为[batch, num_classes]
            self.scores = tf.add(tf.matmul(self.h_drop, self.softmax_weight), self.softmax_bias, name="scores")
            print_variable_info(self.scores)

            # 计算交叉熵的平均损失
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores + 1e-10)
            # 为了防止过拟合，最后还要在loss func中加入l2正则项，即l2_loss。l2_reg_lambda来确定惩罚的力度
            self.loss = tf.reduce_mean(self.cross_entropy) + self.l2_reg_lambda * l2_loss
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
        self.grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

        grad_summaries = []
        for grad, var in self.grads_and_vars:
            if grad is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(var.name), grad)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(var.name), tf.nn.zero_fraction(grad))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))
