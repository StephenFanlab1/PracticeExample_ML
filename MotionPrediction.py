"""
人体运动状态预测
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer#数据预处理
from sklearn.model_selection import train_test_split#自动生成训练集和测试集
from sklearn.metrics import classification_report#预测结果评估

from sklearn.neighbors import KNeighborsClassifier#KNN
from sklearn.tree import DecisionTreeClassifier#决策树
from sklearn.naive_bayes import GaussianNB#朴素贝叶斯

def load_dataset(feature_paths,label_paths):
    """
    数据导入，读取特征文件列表和label文件列表中的内容，归并后返回
    :param feature_paths:
    :param label_paths:
    :return:
    """
    feature=np.ndarray(shape=(0,41))
    label=np.ndarray(shape=(0,1))
    for file in feature_paths:
        # 逗号分隔读取特征数据，将问好替换标记为缺失值，文件中不包含表头
        df=pd.read_table(file,delimiter=',',na_values='?',header=None)
        #使用平均值补全缺失值，然后将数据进行补全
        imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
        imp.fit(df)#训练处理器
        df=imp.transform(df)#生成预处理结果
        #将新读入的数据合并到特征集中
        feature=np.concatenate((feature,df))
    for file in label_paths:
        #读取标签文件，文件汇总不包含表头
        df=pd.read_table(file,header=None)
        #将新读入的数据合并到标签集中
        label=np.concatenate((label,df))
    #标签规整为一维向量
    label=np.ravel(label)
    return feature,label





from sklearn.model_selection import cross_val_score#交叉验证法
from sklearn.datasets import load_iris
def main():
    #设置数据路径
    feature_paths=['A/A.feature','B/B.feature','C/C.feature','D/D.feature','E/E.feature']
    label_paths=['A/A.label','B/B.label','C/C.label','D/D.label','E/E.label']
    #将前四个数据集作为数据集读入
    x_train,y_train=load_dataset(feature_paths[:4],label_paths[:4])
    #将剩余的数据集作为测试集读入
    x_test,y_test=load_dataset(feature_paths[4:],label_paths[4:])
    #使用全量数据作为数据集，将数据随机打乱
    x_train,x_,y_train,y_=train_test_split(x_train,y_train,test_size=0.0)
    #使用KNN预测
    print('Start training KNN')
    knn=KNeighborsClassifier().fit(x_train,y_train)
    print('Training done!')
    result_knn=knn.predict(x_test)
    print('Prediction done!')
    #使用决策树预测
    print('Start training DT')
    dt=DecisionTreeClassifier().fit(x_train,y_train)
    print('Training done!')
    result_dt = dt.predict(x_test)
    print('Prediction done!')
    #Bayes预测
    print('Start training Bayes')
    gnb=GaussianNB().fit(x_train,y_train)
    print('Training done!')
    result_gnb = gnb.predict(x_test)
    print('Prediction done!')
    #计算准确率 召回率 f1值 支持度
    print("\n\nThe classification report for KNN:")
    print(classification_report(y_test,result_knn))
    print("\n\nThe classification report for DT:")
    print(classification_report(y_test, result_dt))
    print("\n\nThe classification report for Bayes:")
    print(classification_report(y_test, result_gnb))











if __name__ == '__main__':
    main()

