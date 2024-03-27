import pandas as pd
import numpy as np
import re
import json
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


dataset1_path="./movies_dataset.csv"
dataset2_path="./Alzheimer Disease and Healthy Aging Data In US.csv"


def judgeType(data):                 #判断属性，标称数据：1，数值属性：0,异常值：-1
    if type(data)==str:
        return 1
    elif type(data)==np.int64 or type(data)==np.float64 or type(data)==float:
        return 0
    else:
        return -1


def GetNominalFrequency(datas,dict_len=50):      #计算标称数据的频数，返回字典
    ans=dict()
    for data in datas:
        if type(data)==str and data!="":
            ans[data]=ans.get(data,0)+1
    sorted_items=sorted(ans.items(),key=lambda item:item[1],reverse=True)
    ans=dict(sorted_items)
    if len(sorted_items)>dict_len:            #只保留前50个值
        ans=dict(sorted_items[:dict_len])
    ans=json.dumps(ans,indent=4)
    return ans

def readDataset(filepath):      #读取数据集,返回dataframe和属性列表
    data_df=pd.read_csv(filepath)
    header=list(data_df.columns)         #属性列表
    return data_df,header

def countValidNum(data):           #计算数值属性缺失值的个数
    nan_mask=np.isnan(data)
    num_nan=np.sum(nan_mask)
    return len(data)-num_nan,num_nan

def countValidStr(data):           #计算数值属性缺失值的个数
    nan_num=0
    for d in data:
        if type(d)!=str:
            nan_num+=1
    return len(data)-nan_num,nan_num

def Str2Num(data):              #将内容为数值但实际上是字符的数组进行预处理
    for i,d in enumerate(data):
        if type(d)==str:
            data[i]=data[i].replace(",","")
    return data.astype(np.float64)


def Time2Num(data):             #将时间字符串转换为数值
    for i,d in enumerate(data):
        if type(d)==str and ("min" in d or "h" in d):
            head,tail=0,0
            hours,mins=0,0
            if 'h' in d:
                while d[head]!='h':
                    hours=hours*10+int(d[head])
                    head+=1
                    tail=head+2
            if 'min' in d:
                while re.match("[0-9]",d[tail]):
                    mins=mins*10+int(d[tail])
                    tail+=1
            data[i]=hours*60+mins
    return data.astype(np.float64)            

def drawBoxandHist(data):   #同时画盒图和直方图
    fig,axs=plt.subplots(1,2,figsize=(8,4))
    # 绘制直方图  
    axs[0].hist(data, bins=30, density=True, alpha=0.6, color='g')  
    # 设置标题和坐标轴标签  
    axs[0].set_title('Histogram')  
    # plt.boxplot(data, vert=True,medianprops={'linewidth': 2}) 
    # plt.grid(linestyle="--",alpha=0.3)
    # plt.title("Box Plot")
    df=pd.DataFrame(data)
    df.plot.box(title="Boxplot",ax=axs[1])
    plt.grid(linestyle="--",alpha=0.3)
    plt.show()
    return


def valueAnalysis(data_df,attr_name,dict_len=50,miss=True):        #分析数据集一列数据    
    attr_value=np.array(data_df.loc[:,attr_name])
    attr_type=judgeType(attr_value[0])          #判断属性类别
    if attr_type==0:            #数值属性
        print("{}: numberic attribute".format(attr_name))
        valid,missing=countValidNum(attr_value)
        print("Valid:{}\nMissing:{}".format(valid,missing))
        print("Five number summary: min: {}, Q1: {:.2f}, median: {:.2f}, Q3: {:.2f}, max: {}".format(np.nanmin(attr_value),np.nanpercentile(attr_value,25),np.nanmedian(attr_value),np.nanpercentile(attr_value,75),np.nanmax(attr_value))) #给出5数概括
        drawBoxandHist(attr_value)
        if missing>0 and miss:           #有缺失值
            old_df1,old_df2,old_df3=data_df.loc[:,attr_name],data_df.loc[:,attr_name],data_df.loc[:,attr_name]
            mode=old_df1.mode()
            old_df1=np.array(old_df1.fillna(mode[0]))
            old_df2=np.array(old_df2.fillna(method='ffill'))
            print("Fill with mode:")
            print("Five number summary: min: {}, Q1: {:.2f}, median: {:.2f}, Q3: {:.2f}, max: {}".format(np.nanmin(old_df1),np.nanpercentile(old_df1,25),np.nanmedian(old_df1),np.nanpercentile(old_df1,75),np.nanmax(old_df1)))
            drawBoxandHist(old_df1)
            print("Fill with the previous value:")
            print("Five number summary: min: {}, Q1: {:.2f}, median: {:.2f}, Q3: {:.2f}, max: {}".format(np.nanmin(old_df2),np.nanpercentile(old_df2,25),np.nanmedian(old_df2),np.nanpercentile(old_df2,75),np.nanmax(old_df2)))
            drawBoxandHist(old_df2)

            


    elif attr_type==1:          #标称属性
        print("{}: nominal attribute".format(attr_name))
        valid,missing=countValidStr(attr_value)
        print("Valid:{}\nMissing:{}".format(valid,missing))
        result=GetNominalFrequency(attr_value,dict_len)
        print(result)  
    else:                       #存在异常值的处理
        pass
