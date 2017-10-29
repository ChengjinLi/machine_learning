# 中文文档

    * Created on 2017年10月01日
    * author: MJ
    * project: scripts

pip 使用国内源
常用国内的pip源如下：
清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/
阿里云 http://mirrors.aliyun.com/pypi/simple/
中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/
豆瓣(douban) http://pypi.douban.com/simple/
中国科学技术大学 http://pypi.mirrors.ustc.edu.cn/simple/

临时使用直接 -i 加 url ：

pip install web.py -i http://pypi.douban.com/simple

如果有如下报错：


请使用命令信任域名：

 pip install web.py -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

配置成默认源的方法如下：
需要创建或修改配置文件（一般都是创建），
linux的文件在~/.pip/pip.conf，
windows在%HOMEPATH%\pip\pip.ini），
修改内容为：

[global]
index-url = http://pypi.douban.com/simple
[install]
trusted-host=pypi.douban.com
这样在使用pip来安装时，会默认调用该镜像。

临时使用其他源安装软件包的python脚本如下：

#!/usr/bin/python
import os
package = raw_input("Please input the package which you want to install!\n")
command = "pip install %s -i http://pypi.mirrors.ustc.edu.cn/simple --trusted-host pypi.mirrors.ustc.edu.cn" % package
os.system(command)