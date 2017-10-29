#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2017年10月01日

@author: MJ
"""


def ensure_utf8(str1):
    if not str1:
        return ''
    if isinstance(str1, unicode):
        return str1.encode('utf-8')
    return str1


def ensure_unicode(str1):
    if not str1:
        return u''
    if isinstance(str1, unicode):
        return str1
    else:
        return str1.decode('utf-8')


def str_q2b(ustring):
    """
    全角转半角
    """
    rstring = ""
    for uchar in ustring.decode('utf-8'):
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += unichr(inside_code)
    return rstring


def str_b2q(ustring):
    """
    半角转全角
    """
    rstring = ""
    for uchar in ustring.decode('utf-8'):
        inside_code = ord(uchar)
        if inside_code == 32:  # 半角空格直接转化
            inside_code = 12288
        elif 32 <= inside_code <= 126:  # 半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += unichr(inside_code)
    return rstring