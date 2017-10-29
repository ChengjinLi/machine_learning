#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017年10月03日

@author: MJ
"""
from __future__ import absolute_import

import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if p not in sys.path:
    sys.path.append(p)
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from data.data_helpers import load_mr_polarity_data_and_labels, load_sohu_news_data_and_labels, batch_iter
from NLP.text_classification.dl_method.cnn.text_cnn import TextCNN
from NLP.text_classification.dl_method.rnn.text_rnn import TextRNN


def get_config_args():
    """
    读取参数
    """
    parser = argparse.ArgumentParser(description='train_argument')
    # 共同参数
    # 数据集，目前支持电影评论数据(movie_review_data)和搜狐新闻分类数据(sohu_news_data)
    parser.add_argument('--data-set', type=str, default='movie_review_data', help='Data set')
    # 模型，目前支持text_cnn，text_rnn
    parser.add_argument('--model-name', type=str, default='text_cnn', help='Model name')
    # 分类的类别个数
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classification')
    # 每个词表表示成词向量的长度
    parser.add_argument('--embedding-dim', type=int, default=128, help='Dimensionality of word embedding')
    # 保留一个神经元的概率，这个概率只在训练的时候用到
    parser.add_argument('--dropout-keep-prob', type=float, default=0.5, help='Dropout keep probability')
    # 每批读入样本的数量
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    # 优化器
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    # 文本最大长度
    parser.add_argument('--max-sentence-length', type=int, default=50, help='Max sentence length')
    # 初始的学习率
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Initial learning rate')
    # 参数随机初始化的最大值
    parser.add_argument('--init-scale', type=float, default=0.1, help='Init scale')
    # 每次训练读取的数据随机的次数
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of training epochs')
    # 训练过程中, 用于验证数据的数量
    parser.add_argument('--valid-num', type=int, default=1000, help='Number of validation')
    # 在每个固定迭代次数之后，输出结果
    parser.add_argument('--show-freq', type=int, default=10, help='Show train results after this many steps')
    # 在每个固定迭代次数之后，在验证数据上评估模型
    parser.add_argument('--valid-freq', type=int, default=100, help='Validate model on dev set after this many steps')
    # 在每个固定迭代次数之后，保存模型
    parser.add_argument('--save-freq', type=int, default=100, help='Save model after this many steps')
    # 模型保存路径
    parser.add_argument('--model-dir', type=str, default='./data/model', help='The path of saved model')
    # 设置为True时, 如果你指定的设备不存在，允许TF自动分配设备
    parser.add_argument('--allow-soft-placement', action='store_true', default=False, help='Allow device soft device placement')
    # 设备上放置操作日志的位置
    parser.add_argument('--log-device-placement', action='store_true', default=False, help='Log placement of ops on devices')

    # CNN模型需要的参数
    # 每个卷积核的高度(宽度为embedding_dim大小)
    parser.add_argument('--filter-sizes', type=str, default='3,4,5', help='Comma-separated filter sizes')
    # 每个尺寸下的卷积核的个数
    parser.add_argument('--num-filters', type=int, default=128, help='Number of filters per filter size')
    # 激活函数
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    # L2 正则项的参数lambda
    parser.add_argument('--l2-reg-lambda', type=float, default=0.0, help='L2 regularization lambda')

    # RNN模型需要的参数
    # 是否使用双向
    parser.add_argument('--bidirectional', action='store_true', default=False, help='Is bidirectional or not')
    # 选择cell的类型，可选rnn，lstm，gru
    parser.add_argument('--cell-type', type=str, default='gru', help='Cell type')
    # 隐层数
    parser.add_argument('--hidden-layer-num', type=int, default=3, help='Number of hidden layer')
    # 隐层单元数
    parser.add_argument('--hidden-neural-size', type=int, default=128, help='Number of hidden neural')
    # 梯度最大值
    parser.add_argument('--max-grad-norm', type=int, default=5, help='Max grad norm')
    # output计算方法，可选mean，last
    parser.add_argument('--output-method', type=str, default='mean', help='Method of calculate output')

    config = parser.parse_args()
    
    print("{}={}".format('data-set', config.data_set))
    print("{}={}".format('model-name', config.model_name))
    print("{}={}".format('num-classes', config.num_classes))
    print("{}={}".format('embedding-dim', config.embedding_dim))
    print("{}={}".format('dropout-keep-prob', config.dropout_keep_prob))
    print("{}={}".format('batch-size', config.batch_size))
    print("{}={}".format('optimizer', config.optimizer))
    print("{}={}".format('max-sentence-length', config.max_sentence_length))
    print("{}={}".format('learning-rate', config.learning_rate))
    print("{}={}".format('init-scale', config.init_scale))
    print("{}={}".format('num-epochs', config.num_epochs))
    print("{}={}".format('valid-num', config.valid_num))
    print("{}={}".format('show-freq', config.show_freq))
    print("{}={}".format('valid-freq', config.valid_freq))
    print("{}={}".format('save-freq', config.save_freq))
    print("{}={}".format('model-dir', config.model_dir))
    print("{}={}".format('allow-soft-placement', config.allow_soft_placement))
    print("{}={}".format('log-device-placement', config.log_device_placement))

    if 'text_cnn' == config.model_name:
        print("{}={}".format('filter-sizes', config.filter_sizes))
        print("{}={}".format('num-filters', config.num_filters))
        print("{}={}".format('activation', config.activation))
        print("{}={}".format('l2-reg-lambda', config.l2_reg_lambda))

    elif 'text_rnn' == config.model_name:
        print("{}={}".format('bidirectional', config.bidirectional))
        print("{}={}".format('cell-type', config.cell_type))
        print("{}={}".format('hidden-layer-num', config.hidden_layer_num))
        print("{}={}".format('hidden-neural-size', config.hidden_neural_size))
        print("{}={}".format('output-method', config.output_method))
    else:
        print ('Input model_name %s is valid, please input text_cnn or text_rnn !' % config.model_name)
        exit(0)
    return config


class Train(object):

    def __init__(self, config):
        self.config = config
        with tf.name_scope('train') as scope:
            with tf.variable_scope("model"):
                # 输入x占位符
                self.input_x_ph = tf.placeholder(tf.int32, [None, config.max_sentence_length], name="input_x")
                # 输入y占位符
                self.input_y_ph = tf.placeholder(tf.int32, [None, config.num_classes], name="input_y")
                # dropout占位符
                self.dropout_keep_prob_ph = tf.placeholder(tf.float32, name="dropout_keep_prob")
                # 根据 model_name 来确定使用哪个模型
                if 'text_cnn' == self.config.model_name:
                    self.model = TextCNN(self.input_x_ph, self.input_y_ph, self.dropout_keep_prob_ph,
                                         self.config, scope)
                elif 'text_rnn' == self.config.model_name:
                    self.model = TextRNN(self.input_x_ph, self.input_y_ph, self.dropout_keep_prob_ph,
                                         self.config, scope)
                else:
                    print ('Input model_name %s is valid, please input text_cnn or text_rnn !' % config.model_name)
                    exit(0)

    def train_step(self, session, input_x, input_y, summary_writer=None):
        """train step"""
        start_time = time.time()
        feed_dict = dict()
        feed_dict[self.input_x_ph] = input_x
        feed_dict[self.input_y_ph] = input_y
        feed_dict[self.dropout_keep_prob_ph] = self.config.dropout_keep_prob
        fetches = [self.model.train_op,
                   self.model.global_step,
                   self.model.loss,
                   self.model.accuracy,
                   self.model.summary]
        _, global_step, loss_val, accuracy_val, summary = session.run(fetches, feed_dict)

        if self.config.show_freq and (global_step <= 100 or global_step % self.config.show_freq == 0):
            step_time = time.time() - start_time
            examples_per_sec = self.config.batch_size / step_time
            print ("Train, step {}, loss {:g}, acc {:g}, step-time {:g}, examples/sec {:g}"
                   .format(global_step, loss_val, accuracy_val, step_time, examples_per_sec))
        if summary_writer:
            summary_writer.add_summary(summary, global_step)
        return global_step


class Valid(object):

    def __init__(self, config):
        self.config = config
        with tf.name_scope('valid') as scope:
            with tf.variable_scope('model', reuse=True):
                # 输入x占位符
                self.input_x_ph = tf.placeholder(tf.int32, [None, config.max_sentence_length], name="input_x")
                # 输入y占位符
                self.input_y_ph = tf.placeholder(tf.int32, [None, config.num_classes], name="input_y")
                # dropout占位符
                self.dropout_keep_prob_ph = tf.placeholder(tf.float32, name="dropout_keep_prob")
                # 根据 model_name 来确定使用哪个模型
                if 'text_cnn' == self.config.model_name:
                    self.model = TextCNN(self.input_x_ph, self.input_y_ph, self.dropout_keep_prob_ph,
                                         self.config, scope)
                elif 'text_rnn' == self.config.model_name:
                    self.model = TextRNN(self.input_x_ph, self.input_y_ph, self.dropout_keep_prob_ph,
                                         self.config, scope)
                else:
                    print ('Input model_name %s is valid, please input text_cnn or text_rnn !' % config.model_name)
                    exit(0)

    def valid_step(self, session, input_x, input_y):
        """test step"""
        start_time = time.time()
        valid_batches = batch_iter(zip(input_x, input_y), self.config.batch_size)
        valid_loss = []
        valid_correct_prediction = []
        for valid_batch in valid_batches:
            x_batch, y_batch = zip(*valid_batch)

            feed_dict = dict()
            feed_dict[self.input_x_ph] = x_batch
            feed_dict[self.input_y_ph] = y_batch
            feed_dict[self.dropout_keep_prob_ph] = 1.0  # 测试时需要将dropout_keep_prob置为1.0
            fetches = [self.model.global_step,
                       self.model.loss,
                       self.model.correct_prediction]
            global_step, loss_val, correct_prediction_val = session.run(fetches, feed_dict)
            valid_loss.append(loss_val)
            valid_correct_prediction.append(correct_prediction_val)
        step_time = time.time() - start_time
        examples_per_sec = input_y.shape[0] / step_time
        loss_val = np.mean(valid_loss)
        accuracy_val = np.mean(np.concatenate(valid_correct_prediction))
        print ("Valid, loss {:g}, acc {:g}, step-time {:g}, examples/sec {:g}"
               .format(loss_val, accuracy_val, step_time, examples_per_sec))


def main():
    # 1、加载参数
    print('Loading config...')
    config = get_config_args()
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(config.model_dir, config.model_name, timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('Writing to {}\n'.format(out_dir))
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
    # 2、加载数据
    print('Loading %s ...' % config.data_set)
    if 'movie_review_data' == config.data_set:
        x, y = load_mr_polarity_data_and_labels()
        # 获取最大的样本分词长度
        config.max_sentence_length = max([len(item.split(' ')) for item in x])
        print ('max_sentence_length: %d' % config.max_sentence_length)
    elif 'sohu_news_data' == config.data_set:
        x, y = load_sohu_news_data_and_labels()
    else:
        print ('Input data_set %s is valid, please input movie_review_data or sohu_news_data !' % config.data_set)
        exit(0)
    # 3、建立词典
    vocab_processor = learn.preprocessing.VocabularyProcessor(config.max_sentence_length)
    # 将训练数据根据词汇处理器转换成相应的格式
    x = np.array(list(vocab_processor.fit_transform(x)))

    # 4、将数据分割为训练数据和测试数据
    # 设置随机数种子
    np.random.seed(0)
    # 返回一个洗牌后程度为len(y)的数组
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    # 随机后的数据x
    x_shuffled = x[shuffle_indices]
    # 随机后的类别y
    y_shuffled = y[shuffle_indices]
    # 分割训练集和测试集, 用于交叉验证
    valid_sample_index = -config.valid_num
    x_train, x_valid = x_shuffled[:valid_sample_index], x_shuffled[valid_sample_index:]
    y_train, y_valid = y_shuffled[:valid_sample_index], y_shuffled[valid_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Valid split: {:d}/{:d}".format(len(y_train), len(y_valid)))
    config.vocabulary_size = len(vocab_processor.vocabulary_)

    # 5、训练
    print("Begin training")
    with tf.Graph().as_default():
        # session配置
        session_conf = tf.ConfigProto(
            allow_soft_placement=config.allow_soft_placement,
            log_device_placement=config.log_device_placement)
        # 自定义session然后通过session.as_default() 设置为默认session
        with tf.Session(config=session_conf).as_default() as sess:
            train = Train(config)
            valid = Valid(config)
            train_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "train"), sess.graph)
            # 保存字典
            vocab_processor.save(os.path.join(out_dir, "vocab"))
            # 全局变量初始化
            sess.run(tf.global_variables_initializer())
            # 保存全局变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)

            for num_epoch in range(config.num_epochs):
                training_batches = batch_iter(zip(x_train, y_train), config.batch_size)
                print ('epoch {}'.format(num_epoch + 1))
                for training_batch in training_batches:
                    x_batch, y_batch = zip(*training_batch)
                    step = train.train_step(sess, x_batch, y_batch, train_summary_writer)
                    if step % config.valid_freq == 0:
                        valid.valid_step(sess, x_valid, y_valid)
                    if step % config.save_freq == 0:
                        path = saver.save(sess, checkpoint_prefix, step)
                        print("Saved model checkpoint to {}\n".format(path))
            sess.close()

if __name__ == '__main__':
    main()


