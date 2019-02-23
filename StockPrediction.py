"""
上证指数涨跌预测
"""
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import cross_validation

def main():
    #数据加载、预处理
    data=pd.read_csv('stock/000777.csv',encoding='gbk',parse_dates=[0],index_col=0)
    data.sort_index(0,ascending=True,inplace=True)

    day_feature=150#选取150天的数据
    feature_num=5*day_feature#取5列数据作为特征 收盘价、最高价、最低价、开盘价、成交量
    x=np.zeros((data.shape[0]-day_feature,feature_num+1))#训练150天数据
    y=np.zeros((data.shape[0]-day_feature))
    for i in range(0,data.shape[0]-day_feature):
        x[i,0:feature_num]=np.array(data[i:i+day_feature]\
                                    [[u'收盘价',u'最高价',u'最低价',u'开盘价',u'成交量']]).reshape((1,feature_num))
        x[i,feature_num]=data.ix[i+day_feature][u'开盘价']#最后一列为当日开盘价
    for i in range(0,data.shape[0]-day_feature):
        if data.ix[i+day_feature][u'收盘价']>=data.ix[i+day_feature][u'开盘价']:
            y[i]=1
        else:y[i]=0#收盘价高于开盘价
    #创建SVM分类器，核函数分别为rbf和sigmoid，进行交叉验证
    clf_rbf=svm.SVC(kernel='rbf')
    clf_sig=svm.SVC(kernel='sigmoid')
    result_rbf=[]
    result_sig=[]
    for i in range(5):
        x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=0.2)#取0.8的数据为训练集
        clf_rbf.fit(x_train,y_train)
        clf_sig.fit(x_train,y_train)
        result_rbf.append(np.mean(y_test==clf_rbf.predict(x_test)))#预测结果与验证集数据比较
        result_sig.append(np.mean(y_test==clf_sig.predict(x_test)))
    print("svm classifier with kernel_rbf accuracy")
    print(result_rbf)
    print("svm classifier with kernel_sigmoid accuracy")
    print(result_sig)


if __name__ == '__main__':
    main()

