#!/bin/sh

PWD=`pwd`

source ${PWD}/scripts/_common.sh

echo ${ECHO_EXT} "${Red}初始化开始: ${Gre}${ENV_PATH}${RCol} >>>>>>>"

#
if [ -f /usr/local/python27/bin/virtualenv ]; then
    VIRTUAL_ENV=/usr/local/python27/bin/virtualenv
else
    VIRTUAL_ENV=`which virtualenv`
fi


if [ ! -d ${ENV_PATH} ]; then
    ${VIRTUAL_ENV} ${ENV_PATH}
fi

source ${ENV_PATH}/bin/activate

SITE_CUSTOMIZE="${ENV_PATH}/lib/python2.7/site-packages/sitecustomize.py"
if [ ! -f ${SITE_CUSTOMIZE}  ]; then
    cat >> ${SITE_CUSTOMIZE} << "EOF"
# -*- coding:utf-8 -*-
#
# 设置系统的默认编码, 这样utf-8和unicode之间就可以自由转换了；否则系统默认的编码为ascii
#
import sys
sys.setdefaultencoding('utf-8')
EOF
fi

pip install pip==1.5.6

deactivate

echo ${ECHO_EXT} "${Red}初始化完毕: ${Gre}${ENV_PATH}${RCol} <<<<<<"
