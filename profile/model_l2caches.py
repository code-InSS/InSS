from numpy import DataSource
import pandas as pd
import copy
import math
import sys
import json


def ncompute(openfilepath):
    with open(openfilepath, encoding='utf-8') as f:
        id = 0
        reader = pd.read_csv(f)
        # print(reader)
        data = {}
        for i in range(0, len(reader)):
            id = reader.iloc[i]['ID']
            mn = reader.iloc[i]['Metric Name']
            if (mn == 'gpu__time_duration.sum'):
                data[id] = {}
            data[id][mn] = reader.iloc[i]['Metric Value']
        size = len(data)
        kernels = size# / 10
        metrics = ['lts__t_sectors.avg.pct_of_peak_sustained_elapsed']
        ans = []
        tmp = [0] * len(metrics)
        totaldurationtime = 0
        for i in range(size):
            if (i % kernels == 0 and i > 1):
                # print(totaldurationtime)
                tmp = [t / totaldurationtime for t in tmp]
                ans.append(copy.deepcopy(tmp))
                totaldurationtime = 0
                for j in range(len(tmp)):
                    tmp[j] = 0
            totaldurationtime += data[i]['gpu__time_duration.sum'] / 1000
            for j in range(len(metrics)):
                tmp[j] += data[i][metrics[j]] * data[i]['gpu__time_duration.sum'] / 1000
        tmp = [t / totaldurationtime for t in tmp]
        ans.append(copy.deepcopy(tmp))
        avgans = [0] * len(metrics)
        for i in range(len(ans)):
            for j in range(len(metrics)):
                avgans[j] += ans[i][j]
        #avgans = [i / 10 for i in avgans]
        print(avgans[0])
        print(size)
        return avgans[0]


def saverecords(kind, metrics, value):
    path = "config"
    records = {kind: {metrics: value}}
    json_str = json.dumps(records)
    # print(json_str)
    with open(path, "a") as file:
        file.write(json_str + "\n")


def solve(path1,path2,path3,path4,path5):
    return [ncompute(path1),ncompute(path2),ncompute(path3),ncompute(path4),ncompute(path5)]

def solve_1(path):
    return ncompute(path)

if __name__ == '__main__':
    """
    models = ["alexnet", "resnet50","vgg16","resnet18","vgg19","mnasnet","mobilenet_v3","efficientnet"]#["alexnet", "resnet50", "vgg16", "googlenet", "inception_v3", "densenet", "mobilenet_v3", "squeezenet"]
    for model in models:
        l2caches = solve("data/"+model+"_l2caches_1_10.csv", "data/"+model+"_l2caches_16_50.csv",
                         "data/"+model+"_l2caches_32_100.csv")
        # l2caches = solve(sys.argv[1],sys.argv[2],sys.argv[3])
        saverecords(model, "l2caches", l2caches)
    """
    model ='alexnet'
    res=[6,12,18,24,30,36,42,48,54,60,66,72,78,84,90,96,100]
    l2caches=[]
    for i in res:
        path="data/"+model+"_l2caches_"+str(i)+".csv"
        l2caches.append(solve_1(path))
    #l2caches = solve("data/alexnet_l2caches_1_10.csv","data/alexnet_l2caches_16_50.csv","data/alexnet_l2caches_32_100.csv")
    #l2caches = solve(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
    saverecords(model, "l2caches", l2caches)

