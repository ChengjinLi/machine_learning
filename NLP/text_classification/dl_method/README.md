# 中文文档

    * Created on 2017年10月03日
    * author: MJ
    * project: dl_method

# 1. 数据集
    movie_review_data：
        电影评论数据(Movie Review data from Rotten Tomatoes)
        类别：2类，分别为积极评论和消极评论
        样本数：每类5331个样本
        词表大小：18758
        注意：数据集过小容易过拟合，可以使用10%的训练数据进行交叉验证
    sohu_news_data：
        搜狐公开的新闻数据集
        类别：9类，分别为财经，IT，健康，体育，旅游，教育，招聘，文化，军事
        样本数：每类1990个样本
        词表大小为：37832
# 2.预处理
    步骤：
        (1) 加载两类数据
        (2) 文本数据清洗
        (3) 把每个句子填充到最大的句子长度，填充字符是<PAD>，使得每个句子都包含固定个数单词，相同的长度有利于进行高效的批处理
        (4) 根据所有单词的词表，建立一个索引，用一个整数代表一个词，则每个句子由一个整数向量表示
   
# 3. 模型
    根据model_name选择不同参数，详见模型目录下的README.md
# 4. 实现
    get_config_args函数用于加载参数
    Train类用于训练
    Valid类用于验证
    
# 5. 运行
    1、先切换到项目所在路径，例如：
        cd /Users/mj/machine_learning
    2、激活虚拟环境，例如：
        source ../env_machine_learning/bin/activate
    3、切换到执行脚本所在目录，例如：
        cd NLP/text_classification/dl_method
    4、执行训练，例如：
        movie_review_data数据集使用text_cnn模型进行训练    
            sh movie_review_data_text_cnn_train.sh
            输出示例:
                epoch 10
                Train, step 1360, loss 0.00466952, acc 1, step-time 0.109341, examples/sec 585.325
                Train, step 1370, loss 0.00542108, acc 1, step-time 0.115732, examples/sec 553.002
                Train, step 1380, loss 0.00197282, acc 1, step-time 0.108384, examples/sec 590.492
                Train, step 1390, loss 0.00325939, acc 1, step-time 0.110109, examples/sec 581.243
                Train, step 1400, loss 0.0028875, acc 1, step-time 0.111411, examples/sec 574.449
                Valid, step 1400, loss 0.97295, acc 0.767, step-time 0.458529, examples/sec 2180.89
                Saved model checkpoint to /Users/mj/machine_learning/NLP/text_classification/dl_method/model/text_cnn/1507033730/checkpoints/model-1400
                
                Train, step 1410, loss 0.00727031, acc 1, step-time 0.107005, examples/sec 598.104
                Train, step 1420, loss 0.00215597, acc 1, step-time 0.11568, examples/sec 553.25
                Train, step 1430, loss 0.00152452, acc 1, step-time 0.114104, examples/sec 560.892
                Train, step 1440, loss 0.00117493, acc 1, step-time 0.110422, examples/sec 579.594
                Train, step 1450, loss 0.00348132, acc 1, step-time 0.114583, examples/sec 558.547
                Train, step 1460, loss 0.000839302, acc 1, step-time 0.112776, examples/sec 567.496
                Train, step 1470, loss 0.00146231, acc 1, step-time 0.111317, examples/sec 574.935
                Train, step 1480, loss 0.00463232, acc 1, step-time 0.113081, examples/sec 565.966
                Train, step 1490, loss 0.00447236, acc 1, step-time 0.115058, examples/sec 556.241
                Train, step 1500, loss 0.00485656, acc 1, step-time 0.107251, examples/sec 596.731
                Valid, step 1500, loss 1.00974, acc 0.77, step-time 0.455335, examples/sec 2196.19
                Saved model checkpoint to /Users/mj/machine_learning/NLP/text_classification/dl_method/model/text_cnn/1507033570/checkpoints/model-1500
                
        movie_review_data数据集使用text_rnn模型进行训练    
            sh movie_review_data_text_rnn_train.sh
            输出示例：
                epoch 10
                Train, step 1360, loss 0.0171199, acc 1, step-time 0.207476, examples/sec 308.469
                Train, step 1370, loss 0.0177492, acc 1, step-time 0.207557, examples/sec 308.349
                Train, step 1380, loss 0.119373, acc 0.953125, step-time 0.212059, examples/sec 301.803
                Train, step 1390, loss 0.00629758, acc 1, step-time 0.203493, examples/sec 314.507
                Train, step 1400, loss 0.0763804, acc 0.984375, step-time 0.204624, examples/sec 312.769
                Valid, step 1400, loss 1.00897, acc 0.772, step-time 1.73681, examples/sec 575.767
                Saved model checkpoint to /Users/mj/machine_learning/NLP/text_classification/dl_method/model/text_rnn/1507206530/checkpoints/model-1400
                
                Train, step 1410, loss 0.00643247, acc 1, step-time 0.198187, examples/sec 322.928
                Train, step 1420, loss 0.00511679, acc 1, step-time 0.199786, examples/sec 320.343
                Train, step 1430, loss 0.0189883, acc 0.984375, step-time 0.209993, examples/sec 304.772
                Train, step 1440, loss 0.00632823, acc 1, step-time 0.206568, examples/sec 309.825
                Train, step 1450, loss 0.040887, acc 0.984375, step-time 0.215938, examples/sec 296.381
                Train, step 1460, loss 0.0110089, acc 1, step-time 0.209199, examples/sec 305.929
                Train, step 1470, loss 0.024168, acc 0.984375, step-time 0.209438, examples/sec 305.58
                Train, step 1480, loss 0.00274176, acc 1, step-time 0.205021, examples/sec 312.163
                Train, step 1490, loss 0.0176849, acc 0.984375, step-time 0.206212, examples/sec 310.36
                Train, step 1500, loss 0.00952766, acc 1, step-time 0.211434, examples/sec 302.695
                Valid, step 1500, loss 1.06715, acc 0.783, step-time 1.71584, examples/sec 582.805
                Saved model checkpoint to /Users/mj/machine_learning/NLP/text_classification/dl_method/model/text_rnn/1507206530/checkpoints/model-1500
        
        sohu_news_data数据集使用text_cnn模型进行训练    
            sh sohu_news_data_text_cnn_train.sh
            输出示例：
                epoch 10
                Train, step 2390, loss 0.0888882, acc 0.984375, step-time 10.0836, examples/sec 6.34696
                Train, step 2400, loss 0.160806, acc 0.953125, step-time 10.1459, examples/sec 6.30799
                Valid, step 2400, loss 0.361294, acc 0.892, step-time 154.638, examples/sec 6.46672
                Saved model checkpoint to /Users/mj/machine_learning/NLP/text_classification/dl_method/model/text_cnn/1507206750/checkpoints/model-2400
                
                Train, step 2410, loss 0.0752184, acc 0.984375, step-time 9.93438, examples/sec 6.44227
                Train, step 2420, loss 0.0807312, acc 0.984375, step-time 9.87757, examples/sec 6.47933
                Train, step 2430, loss 0.166612, acc 0.953125, step-time 9.79804, examples/sec 6.53192
                Train, step 2440, loss 0.120254, acc 0.96875, step-time 9.97383, examples/sec 6.41679
                Train, step 2450, loss 0.121654, acc 0.96875, step-time 9.99747, examples/sec 6.40162
                Train, step 2460, loss 0.165171, acc 0.953125, step-time 9.96666, examples/sec 6.42141
                Train, step 2470, loss 0.174791, acc 0.9375, step-time 9.93513, examples/sec 6.44179
                Train, step 2480, loss 0.110574, acc 0.96875, step-time 10.097, examples/sec 6.33849
                Train, step 2490, loss 0.100745, acc 0.953125, step-time 9.82911, examples/sec 6.51127
                Train, step 2500, loss 0.134869, acc 0.9375, step-time 10.0469, examples/sec 6.37014
                Valid, step 2500, loss 0.361465, acc 0.893, step-time 154.925, examples/sec 6.45475
                Saved model checkpoint to /Users/mj/machine_learning/NLP/text_classification/dl_method/model/text_cnn/1507206750/checkpoints/model-2500
                
                Train, step 2510, loss 0.250979, acc 0.921875, step-time 9.94158, examples/sec 6.43761
                Train, step 2520, loss 0.18441, acc 0.96875, step-time 10.0222, examples/sec 6.38585
                Train, step 2530, loss 0.153656, acc 0.9375, step-time 10.1741, examples/sec 6.29051
                Train, step 2540, loss 0.119561, acc 0.96875, step-time 9.89759, examples/sec 6.46622
                Train, step 2550, loss 0.121505, acc 0.984375, step-time 10.1182, examples/sec 6.32524
                Train, step 2560, loss 0.163044, acc 0.953125, step-time 9.8497, examples/sec 6.49766
                Train, step 2570, loss 0.198085, acc 0.9375, step-time 9.90212, examples/sec 6.46326
                Train, step 2580, loss 0.107887, acc 0.96875, step-time 9.9948, examples/sec 6.40333
                Train, step 2590, loss 0.102814, acc 0.96875, step-time 9.88293, examples/sec 6.47581
                Train, step 2600, loss 0.219771, acc 0.953125, step-time 9.98666, examples/sec 6.40855
                Valid, step 2600, loss 0.35596, acc 0.888, step-time 153.713, examples/sec 6.50565
                Saved model checkpoint to /Users/mj/machine_learning/NLP/text_classification/dl_method/model/text_cnn/1507206750/checkpoints/model-2600
                
        sohu_news_data数据集使用text_rnn模型进行训练    
            sh sohu_news_data_text_rnn_train.sh
            输出如下：
                epoch 50
                Train, step 12990, loss 0.037701, acc 0.96875, step-time 4.21098, examples/sec 15.1983
                Train, step 13000, loss 0.00656504, acc 1, step-time 4.17654, examples/sec 15.3237
                Valid, loss 1.16032, acc 0.817, step-time 24.4525, examples/sec 40.8956
                Saved model checkpoint to /Users/mj/machine_learning/NLP/text_classification/dl_method/model/text_rnn/1508653714/checkpoints/model-13000
                
                Train, step 13010, loss 0.026306, acc 1, step-time 4.33075, examples/sec 14.778
                Train, step 13020, loss 0.0518438, acc 0.984375, step-time 4.16506, examples/sec 15.3659
                Train, step 13030, loss 0.0199023, acc 0.984375, step-time 4.07781, examples/sec 15.6947
                Train, step 13040, loss 0.0616554, acc 0.984375, step-time 4.15902, examples/sec 15.3882
                Train, step 13050, loss 0.0591692, acc 0.96875, step-time 4.25369, examples/sec 15.0458
                Train, step 13060, loss 0.0754844, acc 0.96875, step-time 4.80702, examples/sec 13.3139
                Train, step 13070, loss 0.0149331, acc 1, step-time 4.00431, examples/sec 15.9828
                Train, step 13080, loss 0.046701, acc 0.984375, step-time 4.14495, examples/sec 15.4405
                Train, step 13090, loss 0.0149407, acc 1, step-time 4.23512, examples/sec 15.1117
                Train, step 13100, loss 0.0540731, acc 0.96875, step-time 4.11245, examples/sec 15.5625
                Valid, loss 1.17758, acc 0.816, step-time 24.3796, examples/sec 41.0179
                Saved model checkpoint to /Users/mj/machine_learning/NLP/text_classification/dl_method/model/text_rnn/1508653714/checkpoints/model-13100
                
                Train, step 13110, loss 0.0285655, acc 0.984375, step-time 4.16834, examples/sec 15.3538
                Train, step 13120, loss 0.00277612, acc 1, step-time 4.18533, examples/sec 15.2915
                Train, step 13130, loss 0.0119377, acc 1, step-time 3.97655, examples/sec 16.0943
                Train, step 13140, loss 0.00700734, acc 1, step-time 4.02984, examples/sec 15.8815
                Train, step 13150, loss 0.0509342, acc 0.96875, step-time 3.996, examples/sec 16.016
                Train, step 13160, loss 0.0892873, acc 0.984375, step-time 4.31169, examples/sec 14.8434
                Train, step 13170, loss 0.0154307, acc 1, step-time 4.28788, examples/sec 14.9258
                Train, step 13180, loss 0.0183856, acc 1, step-time 4.21669, examples/sec 15.1778
                Train, step 13190, loss 0.072192, acc 0.984375, step-time 4.1764, examples/sec 15.3242
                Train, step 13200, loss 0.104614, acc 0.96875, step-time 4.29308, examples/sec 14.9077
                Valid, loss 1.18614, acc 0.822, step-time 24.3938, examples/sec 40.994
          
# 4、预测
    注: 先在predict.py中设置相关参数
    cd NLP/text_classification/dl_method
    python predict.py

# 5、实验结果
    在验证集上的精度：
    |-------------------|----------|----------|
    |        数据集      | text_cnn | text_rnn |
    |-------------------|----------|----------|
    | movie_review_data |  0.77    |  0.783   |
    |-------------------|----------|----------|
    |   sohu_news_data  |  0.888   |  0.822   |
    |-------------------|----------|----------|

    
# 6、实验结论
    训练的指标不是平滑的，原因是我们每个批处理的数据过少
    训练集正确率过高，测试集正确率过低，过拟合。
    避免过拟合：更多的数据；更强的正规化；更少的模型参数。
    短文本有序的数据集上，基于RNN一般会比基于CNN的方法要好。
    长文本或者无序的数据集上，基于CNN一般会比基于RNN的方法要好。
    一般是在数据集上实验不同的方法，选取最适合当前数据集的方法。