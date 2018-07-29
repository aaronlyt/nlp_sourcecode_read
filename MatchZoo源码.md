# MatchZoo 概述

*   包含模型： https://github.com/aaronlyt/MatchZoo
    >1. A Deep Relevance Matching Model for Ad-hoc Retrieval
    >2. MatchPyramid
    >3. ARC-I ARC-II
    >4. DSSM CDSSM
    >5. MV-LSTM
    >6. aNMM:  aNMM: Ranking Short Answer Texts with        Attention-Based Neural Matching Model
    >7. DUET:   Learning to Match Using Local and           Distributed Representations of Text for Web         Search
    >8. models under development: Match-SRNN, DeepRank,     K-NRM ....
    
    
*   调用方法
    ``` 
    python main.py --phase train --model_file ./models/arci_ranking.config
    
    python main.py --phase predict --model_file ./models/arci_ranking.config
    ```
--------------------------------------------------
# MatchZoo 源码
-------------------------------------------------

# 主调用流程 main.py

*   流程
    >1. 解析config文件
        >>1.    读入config配置文件内容
        >>2.    初始化embedding
        >>3.    获取指定的配置(train  eval等)
        >>4.    读取数据集(utils.rank_io read_data)
        >>5.    获取batch generator对象
            >>>1.   参考batch generator
            >>>2.   主要是初始化 Pointer/List/Pair            Genrator三个对象中的一个
        >>6.    加载keras模型(main.py load_model)
        >>7.    设置keras模型的loss function
            >>>1.   hinge loss
            >>>2.   交叉熵
            >>>3.   详细见loss function 模块
        >>8.    设置keras模型的evaluation metrics
        >>9.    compile模型，然后进行训练
        >>10.   预测
*   测试代码
```
    for tag, generator in eval_gen.items():
        genfun = generator.get_batch_generator()
        print('[%s]\t[Eval:%s] ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag), end='')
        res = dict([[k,0.] for k in eval_metrics.keys()])
        num_valid = 0
        for input_data, y_true in genfun:
            y_pred = model.predict(input_data, batch_size=len(y_true))
            # 判断ListGenerator
            if issubclass(type(generator), inputs.list_generator.ListBasicGenerator):
                list_counts = input_data['list_counts']
                for k, eval_func in eval_metrics.items():
                    # the ground truth scores for documents under a query
                    for lc_idx in range(len(list_counts)-1):
                        pre = list_counts[lc_idx]
                        suf = list_counts[lc_idx+1]
                        res[k] += eval_func(y_true = y_true[pre:suf], y_pred = y_pred[pre:suf])
                num_valid += len(list_counts) - 1
            else:
                # 直接batch 计算
                for k, eval_func in eval_metrics.items():
                    res[k] += eval_func(y_true = y_true, y_pred = y_pred)
                num_valid += 1
        generator.reset()
        print('Iter:%d\t%s' % (i_e, '\t'.join(['%s=%f'%(k,v/num_valid) for k, v in res.items()])), end='\n')
        sys.stdout.flush()
```
-----------------------------------
#  Data batch generator
* batch generatro类型
    >1. 定义在inputs模块下面
    >2. pointer_generator.py, 定义类
        >>1.    PointGenerator
        >>2.    Triletter_PointGenerator
        >>3.    DRMM_PointGenerator
        >>4.    应该主要用于classification models
    >3. pair_generator.py
        >>1.    PairBasicGenerator
        >>2.    PairGenerator
        >>3.    Triletter_PairGenerator
        >>4.    DRMM_PairGenerator
        >>5.    PairGenerator_Feats
        >>6.    应该用于pair ranking models
    >4. list_generator.py, 定义类
        >>1.    ListBasicGenerator
        >>2.    ListGenerator
        >>3.    DRMM_ListGenerator
        >>4.    ListGenerator_Feats
        >>5.    应该用于list ranking models

* 模型训练使用

model name | classify | ranking| model |
---|---|---|---|
arcii/arci |PointGenerator |    PairGenerator　| Convolutional Neural Network Architectures for Matching Natural Language Sentences
matchpyramid |　PointGenerator |  PairGenerator | Text Matching as Image Recognition
knrn    | PointGenerator   |  PairGenerator |End-to-End Neural Ad-hoc Ranking with Kernel Pooling
duet    | PointGenerator   |  PairGenerator |Learning to Match Using Local and Distributed Representations of Text for Web Search
mvlstm    | PointGenerator   |  PairGenerator|A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations
dssm    | Triletter_PointGenerator   |  Triletter_PairGenerator|Learning Deep Structured Semantic Models for Web Search using Clickthrough Data
cdssm    | Triletter_PointGenerator   |  Triletter_PairGenerator|Learning Semantic Representations Using Convolutional Neural Networks for Web Search
anmn    | DRMM_PointGenerator   |  DRMM_PairGenerator|aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model
drmm    | DRMM_PointGenerator   |  DRMM_PairGenerator|A Deep Relevance Matching Model for Ad-hoc Retrieval.
drmm_tks| PointGenerator   |  PairGenerator|

##   初始化过程
*   在main.py中被初始化调用,配置在对应的config文件中
*   **主要优点是batch *==genrator对象可配置化==***
*   batch generator对象加载过程
    >1.    传入conf['input_type'](之一)参数
    >2.    调用inputs模块(__init__.py文件中)
                的get函数
    >3.    使用deserialize_keras_object函数，
                获取类
    >4.    传入参数，初始化类对象
    >5.    目的便是产生inputs模块下面
            PointerGenerator/ListGenerator/
            PairGenerator三个中的一个类对象
            ，便于后续的batch 数据获取

## Pointer Generator(inputs/point_generator.py)

* 作用：主要用于分类模型数据准备
* 数据返回的格式
```
 def get_batch_generator(self):
    if self.is_train:
        while True:
            X1, X1_len, X2, X2_len, Y, ID_pairs = self.get_batch()
            if self.config['use_dpool']:
                yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'dpool_index': DynamicMaxPooling.dynamic_pooling_index(X1_len, X2_len, self.config['text1_maxlen'], self.config['text2_maxlen'])}, Y)
            else:
                yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len}, Y)
    else:
        while self.point + self.batch_size <= self.total_rel_num:
            X1, X1_len, X2, X2_len, Y, ID_pairs = self.get_batch(randomly = False)
            if self.config['use_dpool']:
                yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'dpool_index': DynamicMaxPooling.dynamic_pooling_index(X1_len, X2_len, self.config['text1_maxlen'], self.config['text2_maxlen']), 'ID':ID_pairs}, Y)
            else:
                yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'ID':ID_pairs}, Y)
```

* batch_generator源码
```
    def get_batch(self, randomly=True):
        ID_pairs = []
        X1 = np.zeros((self.batch_size, self.data1_maxlen), dtype=np.int32)
        X1_len = np.zeros((self.batch_size,), dtype=np.int32)
        X2 = np.zeros((self.batch_size, self.data2_maxlen), dtype=np.int32)
        X2_len = np.zeros((self.batch_size,), dtype=np.int32)
        if self.target_mode == 'regression':
            Y = np.zeros((self.batch_size,), dtype=np.int32)
        elif self.target_mode == 'classification':
            Y = np.zeros((self.batch_size, self.class_num), dtype=np.int32)

        X1[:] = self.fill_word
        X2[:] = self.fill_word
        for i in range(self.batch_size):
            # 随机选取关系文件中的样本对
            # self.rel是从关系文件中读取的内容
            if randomly:
                label, d1, d2 = random.choice(self.rel)
            else:
                # 按顺序读取关系文件中的样本对
                label, d1, d2 = self.rel[self.point]
                self.point += 1
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            d2_len = min(self.data2_maxlen, len(self.data2[d2]))
            # 获取训练样本的query doc内容
            X1[i, :d1_len], X1_len[i]   = self.data1[d1][:d1_len], d1_len
            X2[i, :d2_len], X2_len[i]   = self.data2[d2][:d2_len], d2_len
            if self.target_mode == 'regression':
                Y[i] = label
            elif self.target_mode == 'classification':
                Y[i, label] = 1.
            ID_pairs.append((d1, d2))
        return X1, X1_len, X2, X2_len, Y, ID_pairs
```
### class Triletter_PairGenerator(PairBasicGenerator)
*   用于模型dssm cdssm
    >1. dssm 模型没有embedding layer
    >2. cdssm模型有wordhash embedding layer
*   主要是将word转换成对应的letter n-grams
*   源码
```
    def read_word_triletter_map(self, wt_map_file):
        """
        the result dict:
            *   key the word term id
            *   value is the list of letter n-grams id of the term
        :param wt_map_file:
        :return:
        """
        word_triletter_map = {}
        for line in open(wt_map_file):
            r = line.strip().split()
            word_triletter_map[int(r[0])] = map(int, r[1:])
        return word_triletter_map

    def map_word_to_triletter(self, words):
        """
        return list of letter n grams of words
        :param words:
        :return:
        """
        triletters = []
        for wid in words:
            if wid in self.word_triletter_map:
                triletters.extend(self.word_triletter_map[wid])
        return triletters

    def transfer_feat2sparse(self, dense_feat):
        """
        the vector as the dssm model input
        the word is represented using a vector of letter n-grams.
        and also the document is represented using a 0-1 vector of letter n-grams
        csr_matrix:
            * where the column indices for row i are stored in indices[indptr[i]:indptr[i+1]]
            * their corresponding values are stored in data[indptr[i]:indptr[i+1]].
        :param dense_feat:
        :return:
        """
        data = []
        indices = []
        indptr = [0]
        for feat in dense_feat:
            for val in feat:
                indices.append(val)
                data.append(1)
            indptr.append(indptr[-1] + len(feat))
        return sp.csr_matrix((data, indices, indptr), shape=(len(dense_feat), self.vocab_size), dtype="float32")

    def transfer_feat2fixed(self, feats, max_len, fill_val):
        """
        the matrix value is the letter n-grams id
        the i row is the letter n-grams id sequence of ith document
        :param feats:
        :param max_len:
        :param fill_val:
        :return:
        """
        num_feat = len(feats)
        nfeat = np.zeros((num_feat, max_len), dtype=np.int32)
        nfeat[:] = fill_val
        for i in range(num_feat):
            rlen = min(max_len, len(feats[i]))
            nfeat[i,:rlen] = feats[i][:rlen]
        return nfeat

    def get_batch(self, randomly=True):
        """
        important to notice:
            *   the dssm model have no embedding layer, the input should be vector
            *   the cdssm model have wordhash embedding layer
        :param randomly:
        :return:
        """
        ...同
        X1, X2 = [], []
        for i in range(self.batch_size):
            ...同
            # document as list of letter n-grams
            X1.append(self.map_word_to_triletter(self.data1[d1]))
            X2.append(self.map_word_to_triletter(self.data2[d2]))
            ...同
            ID_pairs.append((d1, d2))
        if self.dtype == 'dssm':
            return self.transfer_feat2sparse(X1).toarray(), X1_len, self.transfer_feat2sparse(X2).toarray(), X2_len, Y, ID_pairs
        elif self.dtype == 'cdssm':
            return self.transfer_feat2fixed(X1, self.data1_maxlen, self.fill_word), X1_len,  \
                    self.transfer_feat2fixed(X2, self.data2_maxlen, self.fill_word), X2_len, Y, ID_pairs
```

### class DRMM_PointGenerator(object)
*   DRMM 模型 batch generator实现，同时实现Matching Histogram Mapping Layer
*   ==the embeddings used are pre-trained via word2vec [17] because the histograms     are not diferentiable and prohibit end-to-end **learning**==
*   源码
```
    def cal_hist(self, t1, t2, data1_maxlen, hist_size):
        """
        Matching Histogram Mapping　Layer of drmm model implementation
        query中term embedding和document中所有的term embedding都计算相似性分数
        :param t1: 
        :param t2: 
        :param data1_maxlen: 
        :param hist_size: 
        :return: 
        """
        mhist = np.zeros((data1_maxlen, hist_size), dtype=np.float32)
        d1len = len(self.data1[t1])
        if self.use_hist_feats:
            assert (t1, t2) in self.hist_feats
            caled_hist = np.reshape(self.hist_feats[(t1, t2)], (d1len, hist_size))
            if d1len < data1_maxlen:
                mhist[:d1len, :] = caled_hist[:, :]
            else:
                mhist[:, :] = caled_hist[:data1_maxlen, :]
        else:
            # query terms embedding matrix
            t1_rep = self.embed[self.data1[t1]]
            # document terms embedding matrix
            t2_rep = self.embed[self.data2[t2]]
            # query term embedding dot every document term embedding in documents
            mm = t1_rep.dot(np.transpose(t2_rep))
            for (i,j), v in np.ndenumerate(mm):
                if i >= data1_maxlen:
                    break
                # vid is the bin id
                vid = int((v + 1.) / 2. * ( hist_size - 1.))
                mhist[i][vid] += 1.
            mhist += 1.
            # LogCount-based Histogram (LCH)
            mhist = np.log10(mhist)
        return mhist


    def get_batch(self, randomly=True):
        ...同上
        for i in range(self.batch_size):
            if randomly:
                label, d1, d2 = random.choice(self.rel)
            else:
                label, d1, d2 = self.rel[self.point]
                self.point += 1
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            d2_len = min(self.data2_maxlen, len(self.data2[d2]))
            # input for Term Gating Network component
            X1[i, :d1_len], X1_len[i] = self.data1[d1][:d1_len], d1_len
            # Matching Histogram Mapping Layer
            X2[i], X2_len[i] = self.cal_hist(d1, d2, self.data1_maxlen, self.hist_size), d2_len
            ...同上
        return X1, X1_len, X2, X2_len, Y, ID_pairs
```
-------------------------

## Pair Generator(inputs/pair_generator.py)
*   产生数据用于pair ranking models
*    
####   class PairBasicGenerator
*   pairGenrator
*   主要定义pair_list 产生函数
    >1. pair_list : 三元组
        >>1.    document d1
        >>2.    high rating document high_d2 related to d1
        >>3.    low rating document low_d2 compared to high_d1 related to d1
*   函数make_pair_static
```
    def make_pair_static(self, rel):
        rel_set = {}
        pair_list = []
        for label, d1, d2 in rel:
            if d1 not in rel_set:
                rel_set[d1] = {}
            if label not in rel_set[d1]:
                rel_set[d1][label] = []
            rel_set[d1][label].append(d2)
        for d1 in rel_set:
            # 按照ranking分数的高低排序
            label_list = sorted(rel_set[d1].keys(), reverse=True)
            for hidx, high_label in enumerate(label_list[:-1]):
                # lower rating lable list as high_label
                for low_label in label_list[hidx+1:]:
                    # high label中的所有document
                    for high_d2 in rel_set[d1][high_label]:
                        # low label中的所有document
                        for low_d2 in rel_set[d1][low_label]:
                            """
                            d1(document), high_rating document related to d1(high_d2), 
                            low_rating document related to d1(low_d2) corresponding to high_d2.
                            是所有不同rating中的document list之间的组合,high rating document在前面, 
                            low rating document在后面
                            """
                            pair_list.append((d1, high_d2, low_d2))
        print('Pair Instance Count:', len(pair_list), end='\n')
        return pair_list
```
*   函数make_pair_iter
```
    def make_pair_iter(self, rel):
        ...同上
        while True:
            # 此处，随机选择一部分document id，并且不断重复
            rel_set_sample = random.sample(rel_set.keys(), self.config['query_per_iter'])

            for d1 in rel_set_sample:
                label_list = sorted(rel_set[d1].keys(), reverse=True)
                ....同上
            yield pair_list

```
### class PairGenerator(PairBasicGenerator)

*   关键函数 get_batch_static(self)
```
    def get_batch_static(self):
        # batch_size * 2，后续hinge loss使用，详细见下面目的
        X1 = np.zeros((self.batch_size*2, self.data1_maxlen), dtype=np.int32)
        X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        X2 = np.zeros((self.batch_size*2, self.data2_maxlen), dtype=np.int32)
        X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        Y = np.zeros((self.batch_size*2,), dtype=np.int32)

        Y[::2] = 1
        X1[:] = self.fill_word
        X2[:] = self.fill_word
        for i in range(self.batch_size):
            d1, d2p, d2n = random.choice(self.pair_list)
            # document d1 as list
            d1_cont = list(self.data1[d1])
            # high rating document d2p as list related to d1
            d2p_cont = list(self.data2[d2p])
            # low rating document d2n as list related to d1 compared to d2p
            d2n_cont = list(self.data2[d2n])
            d1_len = min(self.data1_maxlen, len(d1_cont))
            d2p_len = min(self.data2_maxlen, len(d2p_cont))
            d2n_len = min(self.data2_maxlen, len(d2n_cont))
            """
            这样做的目的是：
            rank_losses.py rank_hinge_loss函数，采用hinge loss ranking的方式，
            相邻的位置正好组成pos neg 对，并且pos_score > neg_score的
            """
            # document replicated twice,因为要对应high rating document和low rating document
            X1[i*2, :d1_len],  X1_len[i*2] = d1_cont[:d1_len],   d1_len
            X1[i * 2 + 1, :d1_len], X1_len[i * 2 + 1] = d1_cont[:d1_len], d1_len
            # i * 2为high rating document, i * 2 + 1 存储low rating document
            X2[i*2, :d2p_len], X2_len[i*2] = d2p_cont[:d2p_len], d2p_len
            X2[i*2+1, :d2n_len], X2_len[i*2+1] = d2n_cont[:d2n_len], d2n_len

        return X1, X1_len, X2, X2_len, Y
```
-------------------------------
##  List Generator(inputs/list_generator.py)
*   **==给出的实例程序中，只在测试(一般是ranking model中)*使用*==**
*
### class ListBasicGenerator(object)
```
class ListBasicGenerator(object):
    def __init__(self, config={}):
        self.__name = 'ListBasicGenerator'
        self.config = config
        self.batch_list = config['batch_list']
        if 'relation_file' in config:
            self.rel = read_relation(filename=config['relation_file'])
            self.list_list = self.make_list(self.rel)
            self.num_list = len(self.list_list)
        self.check_list = []
        self.point = 0

    def make_list(self, rel):
        """
        return: [("doc_id", [...]), ...]
        """
        list_list = {}
        for label, d1, d2 in rel:
            if d1 not in list_list:
                list_list[d1] = []
            list_list[d1].append( (label, d2) )
        for d1 in list_list:
            list_list[d1] = sorted(list_list[d1], reverse = True)
        print('List Instance Count:', len(list_list), end='\n')
        return list(list_list.items())

```
### class ListGenerator(ListBasicGenerator)
*   给出的实例程序中，只在测试(一般是ranking model中)使用
*   source code
```
    def get_batch(self):
        """
        每一次返回batch样本的数目是变长的,感觉其他的和Point Generator没什么区别
        :return: 
        """
        while self.point < self.num_list:
            currbatch = []
            if self.point + self.batch_list <= self.num_list:
                currbatch = self.list_list[self.point: self.point+self.batch_list]
                self.point += self.batch_list
            else:
                currbatch = self.list_list[self.point:]
                self.point = self.num_list
            # currbatch中pair对的总数
            bsize = sum([len(pt[1]) for pt in currbatch])
            ID_pairs = []
            list_count = [0]
            X1 = np.zeros((bsize, self.data1_maxlen), dtype=np.int32)
            X1_len = np.zeros((bsize,), dtype=np.int32)
            X2 = np.zeros((bsize, self.data2_maxlen), dtype=np.int32)
            X2_len = np.zeros((bsize,), dtype=np.int32)
            Y = np.zeros((bsize,), dtype= np.int32)
            X1[:] = self.fill_word
            X2[:] = self.fill_word
            j = 0
            for pt in currbatch:
                d1, d2_list = pt[0], pt[1]
                d1_cont = list(self.data1[d1])
                # 存儲相同query所有的相关document list的长度
                list_count.append(list_count[-1] + len(d2_list))
                d1_len = min(self.data1_maxlen, len(d1_cont))
                for l, d2 in d2_list:
                    d2_cont = list(self.data2[d2])
                    d2_len = min(self.data2_maxlen, len(d2_cont))
                    X1[j, :d1_len], X1_len[j] = d1_cont[:d1_len], d1_len
                    X2[j, :d2_len], X2_len[j] = d2_cont[:d2_len], d2_len
                    ID_pairs.append((d1, d2))
                    Y[j] = l
                    j += 1
            yield X1, X1_len, X2, X2_len, Y, ID_pairs, list_count
```
-----------------------------------
##  loss function 模块
*   源码（位于losses.rank_losses.py）
*   分类使用的是keras　loss funciton:　categorical_crossentropy
    >1. 最终调用tf.nn.softmax_cross_entropy_with_logits
    >2. keras loss function：
        >>1.    https://github.com/keras-team/keras/blob/master/keras/losses.py

```
    mz_specialized_losses = {'rank_hinge_loss', 'rank_crossentropy_loss'}

def rank_hinge_loss(kwargs=None):
    margin = 1.
    if isinstance(kwargs, dict) and 'margin' in kwargs:
        margin = kwargs['margin']
    def _margin_loss(y_true, y_pred):
        #output_shape = K.int_shape(y_pred)
        # 这里正样本 负样本没有交叉，负样本选择策略也比较简单，正样本后面一个样本
        # 要和PairGenerator的batch generator一起看
        y_pos = Lambda(lambda a: a[::2, :], output_shape= (1,))(y_pred)
        y_neg = Lambda(lambda a: a[1::2, :], output_shape= (1,))(y_pred)
        loss = K.maximum(0., margin + y_neg - y_pos)
        return K.mean(loss)
    return _margin_loss

def rank_crossentropy_loss(kwargs=None):
    """
    所有的example当中都没有使用这个loss函数
    """
    neg_num = 1
    if isinstance(kwargs, dict) and 'neg_num' in kwargs:
        neg_num = kwargs['neg_num']
    def _cross_entropy_loss(y_true, y_pred):
        """
        :param y_true:
        :param y_pred:
        :return:
        """
        y_pos_logits = Lambda(lambda a: a[::(neg_num+1), :], output_shape= (1,))(y_pred)
        y_pos_labels = Lambda(lambda a: a[::(neg_num+1), :], output_shape= (1,))(y_true)
        logits_list, labels_list = [y_pos_logits], [y_pos_labels]
        for i in range(neg_num):
            y_neg_logits = Lambda(lambda a: a[(i+1)::(neg_num+1), :], output_shape= (1,))(y_pred)
            y_neg_labels = Lambda(lambda a: a[(i+1)::(neg_num+1), :], output_shape= (1,))(y_true)
            logits_list.append(y_neg_logits)
            labels_list.append(y_neg_labels)
        logits = tf.concat(logits_list, axis=1)
        labels = tf.concat(labels_list, axis=1)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return _cross_entropy_loss
```
--------------------------------------
##  evaluation 模块
*   函数recall
```
# compute recall@k
# the input is all documents under a single query
def recall(k=10):
    def top_k(y_true, y_pred, rel_threshold=0.):
        if k <= 0:
            return 0.
        s = 0.
        y_true = _to_list(np.squeeze(y_true).tolist()) # y_true: the ground truth scores for documents under a query
        y_pred = _to_list(np.squeeze(y_pred).tolist()) # y_pred: the predicted scores for documents under a query
        pos_count = sum(i > rel_threshold for i in y_true) # total number of positive documents under this query
        c = list(zip(y_true, y_pred))
        random.shuffle(c)
        # 根据预测的值的大小，已经排序了，意味着top_k会取到前面k个结果作为result（即判断为正样本）
        # 所以这时候只需要根据前k个样本的真实值是否是正样本即可判断是否预测正确
        c = sorted(c, key=lambda x: x[1], reverse=True)
        ipos = 0
        recall = 0.
        for i, (g, p) in enumerate(c):
            if i >= k:
                break
            if g > rel_threshold:
                recall += 1
        recall /= pos_count
        return recall
    return top_k
```