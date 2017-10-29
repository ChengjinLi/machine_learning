#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2017年10月01日

@author: MJ
"""
from __future__ import absolute_import
import os
import sys
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)
import jieba
from constant import PROJECT_DIRECTORY, sohu_news_data_label_dict, news_sohusite_url_to_label_dict
from utils.utils import str_q2b, ensure_unicode


def prepare_news_sohusite_xml_txt_for_classification():
    """
    news_sohusite_xml.txt 用于classification的预处理
    """
    data_file = os.path.join(PROJECT_DIRECTORY, 'data/news_sohusite_xml.dat')
    write_directory = os.path.join(PROJECT_DIRECTORY, 'data/news_sohusite_for_classification')
    if not os.path.exists(write_directory):
        os.makedirs(write_directory)
    count = 0
    num = 0
    with open(data_file, 'r') as reader:
        url = ''
        label = ''
        docno = ''
        content = ''
        for line in reader.readlines():
            line = line.replace('\n', '').replace('\r', '')
            count += 1
            if count % 100000 == 0:
                print ('第%d行' % count)
            try:
                data = line.decode('GB18030').encode('utf-8')
            except:
                try:
                    data = line.decode('gbk').encode('utf-8')
                except:
                    data = line.decode('GB2312').encode('utf-8')
            if data.startswith('<url>') and data.endswith('</url>'):
                url = data.replace('<url>', '').replace('</url>', '').strip()
                for key, val in news_sohusite_url_to_label_dict.items():
                    if url.startswith(key):
                        label = val
                        break
                if '' == label:
                    print (url)
            if data.startswith('<docno>') and data.endswith('</docno>'):
                docno = data.replace('<docno>', '').replace('</docno>', '').strip()
            if data.startswith('<content>') and data.endswith('</content>'):
                content = str_q2b(data.replace('<content>', '').replace('</content>', '').strip())
            if '' != url and '' != label and '' != docno and '' != content:
                num += 1
                write_label_dir = os.path.join(write_directory, label)
                if not os.path.exists(write_label_dir):
                    os.makedirs(write_label_dir)
                write_file = open(os.path.join(write_label_dir, docno), 'w')
                write_file.write(content + '\n')
                write_file.close()
                if num % 100000 == 0:
                    print (docno)
                    print (content)
                url = ''
                label = ''
                docno = ''
                content = ''
    print (count)
    print (num)


def prepare_sohu_news_data():
    """
    sohu_news_data预处理
    """
    read_dir = os.path.join(PROJECT_DIRECTORY, 'data/sogou_original_data')
    write_dir = os.path.join(PROJECT_DIRECTORY, 'data/sogou_processed_data')
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    label_dir_list = os.listdir(read_dir)
    for label in label_dir_list:
        label_read_dir = os.path.join(read_dir, label)
        label_name = sohu_news_data_label_dict.get(label)
        label_write_dir = os.path.join(write_dir, label_name)
        print ('process label name: %s' % label_name)
        if not os.path.exists(label_write_dir):
            os.makedirs(label_write_dir)
        label_file_list = os.listdir(label_read_dir)
        file_num = 0
        for label_file in label_file_list:
            file_num += 1
            label_file_read_path = os.path.join(label_read_dir, label_file)
            label_file_write_path = os.path.join(label_write_dir, label_file)
            write_file = open(label_file_write_path, 'w')
            with open(label_file_read_path, 'r') as reader:
                for each_line in reader.readlines():
                    try:
                        data = each_line.decode('GB18030').encode('utf-8')
                    except:
                        try:
                            data = each_line.decode('gbk').encode('utf-8')
                        except:
                            try:
                                data = each_line.decode('GB2312').encode('utf-8')
                            except Exception as e:
                                print e
                                print label_file_read_path
                                continue
                    write_file.write(data)
            write_file.close()
        print ('success fine num: %s' % file_num)


sohu_news_stopwords_set = None


def get_sohu_news_stopwords_set():
    """
    从文件加载搜狐新闻分类停用词表
    """
    global sohu_news_stopwords_set
    if not sohu_news_stopwords_set:
        sohu_news_stopwords_set = set()
        with open(os.path.join(PROJECT_DIRECTORY, 'data/sohu_news_stopwords.txt'), 'r') as reader:
            for each_line in reader.readlines():
                word = ensure_unicode(each_line.replace('\n', '').strip().lower())
                sohu_news_stopwords_set.add(word)
    return sohu_news_stopwords_set


def segment_sohu_news_data():
    """
    sohu_news_data分词
    """
    stopwords_set = get_sohu_news_stopwords_set()
    read_dir_path = os.path.join(PROJECT_DIRECTORY, "data/sohu_news_data")
    write_dir_path = os.path.join(PROJECT_DIRECTORY, "data/sohu_news_segment_data")
    if not os.path.exists(write_dir_path):
        os.makedirs(write_dir_path)
    label_dir_list = os.listdir(read_dir_path)
    for label in label_dir_list:
        label_read_dir = os.path.join(read_dir_path, label)
        label_write_dir = os.path.join(write_dir_path, label)
        print ('process label: %s' % label)
        if not os.path.exists(label_write_dir):
            os.makedirs(label_write_dir)
        label_file_list = os.listdir(label_read_dir)
        file_num = 0
        for label_file in label_file_list:
            file_num += 1
            label_file_read_path = os.path.join(label_read_dir, label_file)
            label_file_write_path = os.path.join(label_write_dir, label_file)
            write_file = open(label_file_write_path, 'w')
            with open(label_file_read_path, 'r') as reader:
                text = ensure_unicode(reader.read().replace('\n', '').replace('\r', '').strip())
                segment_list = jieba.cut(text)
                word_list = []
                for word in segment_list:
                    word = word.strip()
                    if '' != word and word not in stopwords_set:
                        word_list.append(word)
                word_str = ' '.join(word_list)
                write_file.write(word_str)
            write_file.close()
        print ('success fine num: %s' % file_num)


if __name__ == "__main__":
    prepare_news_sohusite_xml_txt_for_classification()
    prepare_sohu_news_data()
    segment_sohu_news_data()
