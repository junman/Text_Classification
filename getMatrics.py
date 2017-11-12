"""
load_files()//模仿sklearn的load_files将一个目录下的所有文件读取出来，
return file_list=[],记录所有的文件,以及target
"""
import os
import nltk
from collections import Counter,OrderedDict
import math
from sklearn import neighbors


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
    if ''in worddict:
        del worddict['']
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
            del word_dict[i]

def CHI(artical_list,word_dict,nums,target_list):
    """
    计算CHI，对worddict计算,对于一个词对每个类别的CHI，取最大的CHI作为最终的结果
    排序选取前nums个特征
    :param artical_list:
    :param word_dict:
    :param nums:
    :param target_list:
    :return:
    """
    CHI_dict = {}
    count=0
    for i in word_dict:
        # print(count)
        count+=1
        chi=[]
        for j in set(target_list): #属于体育类的标签
            A,B,C,D=0.0,0.0,0.0,0.0
            for k in range(0,len(artical_list)):
                if target_list[k]==j:
                    if i in artical_list[k]:
                        A+=1
                    else:
                        C+=1
                else:
                    if i in artical_list[k]:
                        B+=1
                    else:
                        D+=1
            # N=A+B+C+D
            # if i=="cts":
            #     print("这个类别是：",k)
            #     print(A,B,C,D)
            chi.append((A*D-B*C)**2/((A+B)*(C+D)))
        # CHI_dict[i]=float(sum(chi))/len(chi)
        CHI_dict[i]=max(chi)
    if nums==-1:
        nums=len(word_dict) #如果为-1，就是全部
    w_doc = OrderedDict(sorted(CHI_dict.items(), key=lambda t: t[1], reverse=True))
    new_doc = Counter(w_doc).most_common(nums)
    return new_doc


def reconsitution(artical_list,new_doc,word_dict):
    """

    :param artical_list: [{ },{ },{ }]
    :param new_doc: {  }
    :return: new_artical_list
    """
    copy_dict=word_dict.copy()
    new_artical_list=[]
    for i in range(0,len(artical_list)):
        new_artical_list.append({})
        for j in new_doc:
            if j in artical_list[i]:#如果文章中有这个词，就设置为原来的个数，如果原来的文章中没有，就设置为0
                new_artical_list[i][j]=artical_list[i][j]
            else:
                new_artical_list[i][j]=0.0
    for i in copy_dict:
        if i not in new_doc:
            del word_dict[i]            #如果没有，就删除
    return new_artical_list

def TFIDF(artical_list,word_dict):
    """
    每一篇文章的每一个词对应一个权重,每一个词的n/m出现的次数可以一次算出
    :param artical_list:
    :param word_dict:
    :return:
    """
    count={}
    for i in word_dict:
        n=0.0
        for j in range(0,len(artical_list)):
            if artical_list[j][i]!=0.0:
                n+=1
        count[i]=n
    new_artical_list=[]
    for i in range(0,len(artical_list)):
        new_artical_list.append({})
        for j in artical_list[i]:
            TF=artical_list[i][j]/sum(artical_list[i].values())
            IDF=math.log((len(artical_list)+1)/(count[j]+1))+1
            new_artical_list[i][j]=TF*IDF
    return new_artical_list

# if __name__=="__main__":
#     worddict, articallist, target=load_files(r"D:\Text_classification\Text_classification\training")
#     print(len(worddict))
#     print(worddict)
#     if 'is' in worddict:
#         print("is have : ",worddict['is'])
#     new_doc=slice(worddict,10000)
#     print(new_doc)
#     print(len(articallist))
#     # for i in articallist:
#     #     print(i)
#     print(len(target))

# if __name__=="__main__":
#     worddict, articallist, target = load_files(r"D:\Text_classification\Text_classification\training")
#     stop_list=load_stopwords(r"D:\Text_classification\Text_classification\text_classfier\stopwords.txt")
#     print("词向量的长度为：",len(worddict))
#     # print("文本向量的长度为：",len((articallist)))
#     delete_stopwords(stop_list,worddict)
#     print("去停止词后的长度为：",len(worddict))
#     for i in articallist:
#         delete_stopwords(stop_list,i)


# if __name__=="__main__":
#     worddict, articallist, target = load_files(r"D:\Text_classification\Text_classification\training")
#     stop_list=load_stopwords(r"D:\Text_classification\Text_classification\text_classfier\stopwords.txt")
#     delete_stopwords(stop_list, worddict)
#     print(len(target))
#     chi=CHI(articallist,worddict,-1,target)
#     fil=open("chi2.txt",'w')
#     fil.write(str(chi))
#     # print(CHI)
#     # for j in set(target):
#     #     print(j)

if __name__=="__main__":
    # worddict, articallist, target = load_files(r"D:\Text_classification\Text_classification\training")
    # stop_list=load_stopwords(r"D:\Text_classification\Text_classification\text_classfier\stopwords.txt")
    # delete_stopwords(stop_list, worddict)
    file0=open("artical.txt","r")
    articallist=eval(file0.read())
    file1=open("worddict.txt","r")
    worddict=eval(file1.read())
    file2=open("target.txt","r")
    target=eval(file2.read())
    # file2.write(str(target))
    # chi=CHI(articallist,worddict,8000,target)              #提取特征
    # fil = open("chi2.txt", 'w')
    # fil.write(str(chi))
    # new_artical_list=reconsitution(articallist,new_doc=chi,word_dict=worddict)  #删除articallist列表中多余的词汇
    #                                                                             # 删除word_dict中多余的信息
    # new_artical_list=TFIDF(new_artical_list,worddict)
    # print(new_artical_list)
    # fil=open("chi2.txt",'r')
    # new_doc=dict(eval(fil.read()))
    # print(len(worddict))
    # new_artical_list=reconsitution(articallist,new_doc,worddict)
    # print(len(new_artical_list[0]))
    # print(len(worddict))
    # ifidf_list=TFIDF(new_artical_list,worddict)
    # TFIDF_file=open("tfidf.txt","w")
    # TFIDF_file.write(str(ifidf_list))
    # print(ifidf_list[0])
    #
    # ll=open("tfidf.txt","r")
    # pp=eval(ll.read())
    chi=CHI(articallist,worddict,1000,target)
    fil=open("chi3.txt","w")
    fil.write(str(chi))


