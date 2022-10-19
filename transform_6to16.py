import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
set1_1 = pd.read_csv('set1_1_6.csv', sep = ',')
set1_2 = pd.read_csv('set1_2_6.csv', sep = ',')
set1_3 = pd.read_csv('set1_3_6.csv', sep = ',')

sets = [set1_1, set1_2, set1_3]

count = 0
for s in sets:
    for i in range(len(set1_1)):
        if s["type"][i] =="網前小球":
            count+=1
            if prev == "殺球" or prev == "平球":
                s["type"][i] = "擋小球" 
                # 起落點區域是1-->1、2-->2
            elif (int(s["hit_area"][i])==1 and int(s["landing_area"][i])==1) or (int(s["hit_area"][i])==2 and int(s["landing_area"][i])==2):
                s["type"][i] = "勾球"
                # 起點區域是1, 2, 7，落點區域是1, 2, 7
            elif (int(s["hit_area"][i])==1 or int(s["hit_area"][i])==2 or int(s["hit_area"][i])==7) and (int(s["landing_area"][i])==1 or int(s["landing_area"][i])==2 or int(s["landing_area"][i])==7):
                s["type"][i] = "放小球"
            else:
                s["type"][i] = "小平球"
            prev =  "網前小球"
        elif s["type"][i] =="挑球":
            if int(s["hit_area"][i])==1 or int(s["hit_area"][i])==2 or int(s["hit_area"][i])==7:
                s["type"][i] = "挑球"
            else:
                s["type"][i] = "防守挑球"
            prev = "挑球"
        elif s["type"][i] =="平球":
            if prev == "殺球" or prev == "平球":
                s["type"][i] = "防守回抽"
            elif int(s["hit_area"][i])==1 or int(s["hit_area"][i])==2 or int(s["hit_area"][i])==7:
                s["type"][i] = "推球"
            elif int(s["hit_area"][i])==3 or int(s["hit_area"][i])==4 or int(s["hit_area"][i])==9:
                s["type"][i] = "後場抽平球"
            elif int(s["hit_area"][i])==5 or int(s["hit_area"][i])==6 or int(s["hit_area"][i])==8:
                s["type"][i] = "平球"
            prev = "平球"
        elif s["type"][i] =="殺球":
            if int(s["hit_area"][i])==1 or int(s["hit_area"][i])==2 or int(s["hit_area"][i])==7:
                s["type"][i] = "撲球"
            else:
                s["type"][i] = "殺球"
            prev = "殺球"
# print(count)
# print(set1_1["type"].unique())
set1_1.to_csv('./set1_1_15.csv')
set1_2.to_csv('./set1_2_15.csv')
set1_3.to_csv('./set1_3_15.csv')