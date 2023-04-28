#from numpy import DataSource
import pandas as pd
import copy
import math
import sys
import json

def ncompute(openfilepath):
    with open(openfilepath, encoding='utf-8') as f:
        id=0
        reader = pd.read_csv(f)
        #print(reader)
        data={}
        for i in range(0, len(reader)):
            id=reader.iloc[i]['ID']
            mn=reader.iloc[i]['Metric Name']
            if(mn=='gpu__time_duration.sum'):
                data[id]={}
            data[id][mn]=reader.iloc[i]['Metric Value']
        size=len(data)
        kernels=size
        #kernels=size/10
        metrics=['lts__t_sectors.avg.pct_of_peak_sustained_elapsed']
        ans=[]
        tmp=[0]*len(metrics)
        totaldurationtime=0
        for i in range(size):
            if(i%kernels==0 and i>1):
                #print(totaldurationtime)
                tmp=[t/totaldurationtime for t in tmp]
                ans.append(copy.deepcopy(tmp))
                totaldurationtime=0
                for j in range(len(tmp)):
                    tmp[j]=0
            totaldurationtime+=data[i]['gpu__time_duration.sum']/1000
            for j in range(len(metrics)):
                tmp[j]+=data[i][metrics[j]]*data[i]['gpu__time_duration.sum']/1000
        tmp=[t/totaldurationtime for t in tmp]
        ans.append(copy.deepcopy(tmp))
        avgans=[0] * len(metrics)
        for i in range(len(ans)):
            for j in range(len(metrics)):
                avgans[j]+=ans[i][j]
        #avgans=[i/10 for i in avgans]
        print(avgans[0])
        print(size)
        return avgans[0],kernels
        
def saverecords(kind,metrics,value):
    path="config"
    records={kind:{metrics:value}}
    json_str=json.dumps(records)
    #print(json_str)
    with open(path,"a") as file:
        file.write(json_str+"\n")


def saveKernelConfig(model,value): # 不确定json文件是否会重复覆盖 TODO
    path="kernelConfig"
    records={model:value}
    json_str=json.dumps(records)
    #print(json_str)
    with open(path,"a") as file:
        file.write(json_str+"\n")


if __name__ == '__main__':
    '''
    model = ["alexnet", "resnet50","vgg16","googlenet","inception_v3","densenet","mobilenet_v3_l","squeezenet"]
    batchs = [1, 2, 4, 8, 16, 32]
    for i in model:
        l2cache, kernels = ncompute("data/"+i+".csv")
        #l2cache,kernels=ncompute(sys.argv[1])
        saverecords(i, "l2cache", l2cache)
        saverecords(i, "kernel", kernels)
        #saveKernelConfig(i, kernels)
    '''
    #l2cache,kernels=ncompute("data/resnet50.csv")
    l2cache,kernels=ncompute(sys.argv[1])
    saverecords(sys.argv[2],"l2cache",l2cache)
    #saverecords(sys.argv[2],"kernel",kernels)
    saveKernelConfig(sys.argv[2],kernels)
    
