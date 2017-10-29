#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017年10月01日

@author: MJ
"""
import os

# 存储项目所在的绝对路径
PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

rt_polaritydata_label_list = ['pos', 'neg']

sohu_news_label_list = ['IT', '体育', '健康', '军事', '招聘', '教育', '文化', '旅游', '财经']

sohu_news_data_label_dict = {
    'C000008': '财经',
    'C000010': 'IT',
    'C000013': '健康',
    'C000014': '体育',
    'C000016': '旅游',
    'C000020': '教育',
    'C000022': '招聘',
    'C000023': '文化',
    'C000024': '军事'
}

news_sohusite_url_to_label_dict = {
    'http://auto.sohu.com/': '汽车',
    'http://business.sohu.com/': '财经',
    'http://it.sohu.com/': 'it',
    'http://health.sohu.com/': '健康',
    'http://sports.sohu.com/': '体育',
    'http://travel.sohu.com/': '旅游',
    'http://learning.sohu.com/': '教育',
    'http://career.sohu.com/': '招聘',
    'http://cul.sohu.com/': '文化',
    'http://mil.news.sohu.com/': '军事',
    'http://news.sohu.com/shehuixinwen.shtml': '社会',
    'http://news.sohu.com/guoneixinwen.shtml': '国内',
    'http://news.sohu.com/guojixinwen.shtml': '国际',
    'http://house.sohu.com/': '房产',
    'http://yule.sohu.com/': '娱乐',
    'http://women.sohu.com/': '时尚',
    'http://media.sohu.com/': '传媒',
    'http://gongyi.sohu.com/': '公益',
}