"""
load_files()//模仿sklearn的load_files将一个目录下的所有文件读取出来，
return file_list=[],记录所有的文件,以及target
"""
import os
import nltk
from collections import Counter,OrderedDict

def load_files(file_dir):
    if (os.path.isdir(file_dir))==False:
        print("This is not a dir")
        return
    target=[]
    data=[]
    dir_list=os.listdir(file_dir)
    # for i in dir_list:  #如果是隐藏的文件路径，则删除
    #     if os.path.isfile(os.path.join(file_dir,i)):
    #         print(i)
    #         del i
    for i in dir_list:
        print(i)
    # 扫描建立词典
    worddict = {}  #所有的文件的共同的词向量
    articallist = [] #单篇文章的词向量，所以可能他们的key不一定完全一致
    for i in dir_list:
        file_list=os.listdir(os.path.join(file_dir,i))
        for j in file_list:
            # articallist.append(dict())
            target.append(i)
            with open(os.path.join(os.path.join(file_dir,i),j)) as file:
                doc=file.read()
                sentence=ie_preprocess(doc)
                articallist.append(list_to_doc(sentence,worddict))
    return  worddict,articallist,target


def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    return sentences


#去除一个单词的非字母字符
def OnlyChar(s,oth=''):
    s2 = s.lower()
    fomart = 'abcdefghijklmnopqrstuvwxyz'
    for c in s2:
        if not c in fomart:
            s = s.replace(c,'')
    return s

def list_to_doc(list_list,word_doc):
    """
    篇文章被分词为这样的结构[[],[],[],[“word”,"word"]->代表一个句子的词体划分]
    :param list_list: 经过nltk处理过的文章句子
    :param word_doc: {}  所有文章的词向量字典
    :return: 单篇文章的词向量字典
    """
    temp={}
    for i in list_list:
        for j in i:
            word=OnlyChar(j)
            if word in word_doc:
                word_doc[word]+=1
            else:
                word_doc[word]=1
            if word in temp:
                temp[word]+=1
            else:
                temp[word]=1
    return temp

def slice(word_doc,count):
    """
    排序切分词向量，保留原来的，返回另外生成的字典
    :param word_doc:
    :param count: 切分的数量，即保留前count项,,8000??
    :return: 返回新的排序的从大到小的词向量
    """
    new_doc=OrderedDict(sorted(word_doc.items(),key=lambda t:t[1],reverse=True))
    new_doc=Counter(new_doc).most_common(count)
    return new_doc

def load_stopwords(filename):
    """
    导入停止词，停止词是英文或中文中常用的词频很高的单词
    :param filename:
    :return: list 停止词列表
    """
    word_list=[]
    with open(filename) as file:
        for line in file:
            word_list.append(line.strip())
    return word_list

def delete_stopwords(stop_list,word_dict):
    for i in stop_list:
        if i in word_dict:
            del worddict[i]


if __name__=="__main__":
    worddict, articallist, target=load_files(r"D:\Text_classification\Text_classification\training")
    print(len(worddict))
    # print(worddict)
    # if 'is' in worddict:
    #     print("is have : ",worddict['is'])
    new_doc=slice(worddict,10000)
    print(new_doc)
    print(len(articallist))
    # for i in articallist:
    #     print(i)
    print(len(target))