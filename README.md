# 文档

    * Created on 2017年10月01日
    * author: MJ
    * project: machine_learning
    * 本人会陆续将自己使用过的机器学习算法示例进行整理,欢迎各位读者挑错指正,如果对你有帮助,请给个star哦!
      如果你对此项目有任何疑问, 可以申请加入以下几个QQ群给予答疑解惑:
      TensorFlow深度学习交流：299814789
      Scikit-learn机器学习交流群：397163918
      个人博客：chengjin.li


# 项目依赖
    * 1、IDE PyCharm Community Edition 2017.1
    * 2、安装 python 2.7 (没有请自行网上查找)
    * 3、安装 pip
        sudo easy_install pip
    * 4、安装 virtualenv
        pip install virtualenv
    * 5、创建项目的虚拟环境(需要先切换到machine_learning下)
        sh scripts/env_prepare.sh
    * 6、安装依赖包
        sh scripts/env_update.sh
        备注: 如果是GPU环境,需要将requiremens.txt中的tensorflow替换成tensorflow-gpu
        未安装成功的依赖包，可以使用pip install xxx方式安装


# 项目结构说明
    * CV(计算机视觉)
        * image_classification(图像分类)
        * image_generation(图像生成)
    * NLP(自然语言处理)
        * IR(信息检索)
        * NER(命名实体识别)
        * text_classification(文本分类)
            * ml_method(基于传统机器学习的方法)
            * dp_method(基于深度学习的方法)
                * cnn(基于卷积神经网络的方法)
                    * text_cnn(用于文本分类的TextCNN模型)
                * rnn(基于循环神经网络的方法)
                    * text_rnn(用于文本分类的TextRNN模型)
        * text_matching(文本匹配)
        * text_similarity(文本相关性)
        * translation(翻译)
    * numpy_use_tutorial(numpy使用示例)
    * tensorflow_use_tutorial(tensorflow使用示例)
    * scripts(shell脚本文件)
    * utils(基础工具)
        * word2vec(词向量工具)
