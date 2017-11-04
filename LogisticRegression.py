# -*- coding:utf-8 -*-
from numpy import *
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import neighbors

def get_train_and_test_data(train_dir,test_dir):
    """
    :param dir: D:\Text_classification\Text_classification\test
    :return: test_datas ;test_datas
    """
    test_datas=load_files(test_dir)
    train_datas=load_files(train_dir)
    return train_datas,test_datas

def  createfeatures(train_datas,test_datas):
    """
    根据数据生成特征，可以采用不同的方法
    :param train_datas: 训练数据
    :param test_datas: 测试数据
    :return: 词向量空间
    """
    count_vec = CountVectorizer(binary = True,decode_error='replace',analyzer='word',token_pattern='\w+')
    doc_train_count = count_vec.fit_transform(train_datas.data)
    doc_test_count=count_vec.transform((test_datas.data))
    # print(shape(doc_train_count))
    # print(shape(doc_test_count))
    return doc_train_count,doc_test_count

def logisticClassifer():
    """
    简单的logisctic回归算法
    :return: result,recall,accurate，每个类的正确率，召回率，正确率
    """
    train_datas,test_datas=get_train_and_test_data(r'D:\Text_classification\Text_classification\test')
    doc_train_count, doc_test_count=createfeatures(train_datas,test_datas)
    clf = LogisticRegression()
    clf.fit(doc_train_count,train_datas.target)
    doc_test_predict=clf.predict(doc_test_count)
    result=metrics.precision_score(test_datas.target,doc_test_predict,average=None)
    recall=metrics.recall_score(test_datas.target,doc_test_predict,average=None)
    accurate=metrics.accuracy_score(test_datas.target,doc_test_predict)
    return result,recall,accurate

def KNN_Brute_Force():
    """
    Brute Force算法是最近邻算法的三个主要方法之一，意思是暴力解决，是最简单的最近邻方法
    :return: result,recall,accurate
    """
    train_datas, test_datas = get_train_and_test_data(r"D:\Text_classification\Text_classification\training",r'D:\Text_classification\Text_classification\test')
    # print(shape(train_datas.data))
    # print(shape(test_datas.data))
    # print(train_datas.data[1])
    # print(train_datas.data[0])
    print(shape(train_datas.target))
    # print(train_datas.target)
    doc_train_count, doc_test_count = createfeatures(train_datas, test_datas)
    # print(shape(doc_train_count))
    # print(shape(doc_test_count))
    clf=neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='brute')
    clf.fit(doc_train_count,train_datas.target)
    doc_test_predict=clf.predict(doc_test_count)
    result=metrics.precision_score(test_datas.target,doc_test_predict,average=None)
    recall=metrics.recall_score(test_datas.target,doc_test_predict,average=None)
    accurate=metrics.accuracy_score(test_datas.target,doc_test_predict)
    return result,recall,accurate

if __name__=="__main__":
    result, recall, accurate=KNN_Brute_Force()
    # print("brute_Force_KNN : ")
    # print("The result is : ",result)
    # print("The recall is : ",recall)
    # print("The accurate is : ",accurate)
    # result, recall, accurate =logisticClassifer()
    # print("logisticClassifer : ")
    # print("The result is : ", result)
    # print("The recall is : ", recall)
    # print("The accurate is : ", accurate)