import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
set1_1 = pd.read_csv('set1_1.csv', sep = ',')
set1_2 = pd.read_csv('set1_2.csv', sep = ',')
set1_3 = pd.read_csv('set1_3.csv', sep = ',')
RallySeg = pd.read_csv('RallySeg.csv', sep=',')

homography_matrix = [[1.4691360971722105, 0.6457341621204867, -768.5790300964597], [-0.036105424392318, 6.424840492559715, -1452.988265493345], [-4.7029847136960454e-05, 0.003690962462560561, 1.0]]

# def ComplicateToSimple(k):
#     global simple_cnt, cnt
#     if k == "放小球" or k == "勾球" or k == "擋小球" or k == "小平球":
#         simple_cnt["網前小球"] += cnt[k]
#     elif k == "推球" or k == "撲球":
#         simple_cnt["推撲球"] += cnt[k]
#     elif k == "挑球" or k =="防守回挑":
#         simple_cnt["挑球"] += cnt[k]
#     elif k == "平球" or k=="後場平抽球" or k=="防守回抽":
#         simple_cnt["平球"] += cnt[k]
#     elif k == "切球" or k=="過度切球":
#         simple_cnt["切球"] += cnt[k]
#     elif k=="殺球" or k=="點扣"
#         simple_cnt["殺球"] += cnt[k]
#     elif k=="長球":
#         simple_cnt["長球"] += cnt[k]

# A在下半場、B在上半場
def MAEofEvent():
    for index in range(1,4):
        rallySeg1 = set1_1[set1_1["server"]==index]["frame_num"]
        rallySeg2 = set1_2[set1_2["server"]==index]["frame_num"]
        rallySeg3 = set1_3[set1_3["server"]==index]["frame_num"]
        gt = (rallySeg1 + rallySeg2 + rallySeg3)/3

        gt = gt.reset_index(drop=True)
        rallySeg1 = rallySeg1.reset_index(drop=True)
        rallySeg2 = rallySeg2.reset_index(drop=True)
        rallySeg3 = rallySeg3.reset_index(drop=True)

        error1 = []
        error23 = []
        for i in range(len(gt)):
            error23.append(abs(gt[i]-rallySeg2[i])+abs(gt[i]-rallySeg3[i]))
        for i in range(len(gt)):
            error1.append(abs(gt[i]-rallySeg1[i]))

        print("MAE23 = ",sum(error23) / (len(gt)+len(gt)))#平均絕對誤差MAE
        print("MAE1 = ",sum(error1) / len(gt))#平均絕對誤差MAE

# A在下半場、B在上半場
# player:A/B
def AccofServe(player=None):
    serve1 = set1_1[set1_1["server"]==1]
    serve2 = set1_3[set1_2["server"]==1]
    serve3 = set1_3[set1_3["server"]==1]

    serve1 = serve1[serve1["player"]==player]
    serve2 = serve2[serve2["player"]==player]
    serve3 = serve3[serve3["player"]==player]

    serve1 = serve1.reset_index(drop=True)
    serve2 = serve2.reset_index(drop=True)
    serve3 = serve3.reset_index(drop=True)
    correct23 = 0
    correct1 = 0

    for i in range(0,len(serve1)):
        cnt = {"發短球":0,"發長球":0 }
        cnt[serve1["type"][i]]+=1
        cnt[serve2["type"][i]]+=1
        cnt[serve3["type"][i]]+=1

        fin_max = max(cnt, key=cnt.get)
        if serve2["type"][i]==fin_max and serve3["type"][i]==fin_max :
            correct23+=2
        elif serve2["type"][i]==fin_max or serve3["type"][i]==fin_max :
            correct23+=1
        if serve1["type"][i]==fin_max:
            correct1+=1

    print(correct23+correct1)
    print(correct23/(len(serve2)+len(serve3)))
    print(correct1/len(serve1))

# A在下半場、B在上半場
# player:A/B 
def AccofBallType(player=None):

    balltype1 = set1_1[set1_1["server"]!=1]
    balltype2 = set1_2[set1_2["server"]!=1]
    balltype3 = set1_3[set1_3["server"]!=1]
    if player!=None:
        balltype1 = balltype1[balltype1["player"]==player]
        balltype2 = balltype2[balltype2["player"]==player]
        balltype3 = balltype3[balltype3["player"]==player]
    balltype1 = balltype1.reset_index(drop=True)
    balltype2 = balltype2.reset_index(drop=True)
    balltype3 = balltype3.reset_index(drop=True)

    # print(balltype1)
    correct23 = 0
    correct1 = 0
    simple_correct23 = 0
    simple_correct1 = 0

    # confusion matrix
    y_true = []
    simple_y_true = []
    cnt = {"放小球":0,"勾球":0,"小平球":0,"推球":0,"挑球":0,"撲球":0,"平球":0,"擋小球":0,
                "點扣":0,"殺球":0,"防守回抽":0,"防守回挑":0,"切球":0,"過度切球":0,"長球":0,"後場抽平球":0 }
    balltype_order = ['放小球', '勾球', '小平球', '推球', '挑球', '撲球', '平球', '擋小球', '點扣', '殺球', '防守回抽', '防守回挑', 
    '切球', '過度切球', '長球', '後場抽平球']

    # simplify 16 -> 6
    # ball_type_mapping = {"放小球":"網前小球","勾球":"網前小球","小平球":"網前小球","推球":"平球","挑球": "挑球","撲球":"殺球","平球":"平球","擋小球":"網前小球",
    #             "點扣":"殺球","殺球":"殺球","防守回抽":"平球","防守回挑": "挑球","切球":"切球","過度切球":"切球","長球":"長球","後場抽平球":"平球"}
    # simple_cnt = {"網前小球":0, "挑球":0, "平球":0, "切球":0, "殺球":0, "長球":0}

    for i in range(0,len(balltype1)):
        # cnt = {"放小球":0,"勾球":0,"小平球":0,"推球":0,"挑球":0,"撲球":0,"平球":0,"擋小球":0,
        #         "點扣":0,"殺球":0,"防守回抽":0,"防守回挑":0,"切球":0,"過度切球":0,"長球":0,"後場抽平球":0 }
        cnt = cnt.fromkeys(cnt, 0) # reset value to 0

        if balltype1["type"][i]!="發長球" and balltype1["type"][i]!="發短球": # serve mistake
            cnt[balltype1["type"][i]]= cnt[balltype1["type"][i]]+1
            cnt[balltype2["type"][i]]= cnt[balltype2["type"][i]]+1
            cnt[balltype3["type"][i]]= cnt[balltype3["type"][i]]+1

            fin_max = max(cnt, key=cnt.get)
            if balltype2["type"][i]==fin_max and balltype3["type"][i]==fin_max :
                correct23 = correct23+2
            elif balltype2["type"][i]==fin_max or balltype3["type"][i]==fin_max :
                correct23 = correct23+1
            if balltype1["type"][i]==fin_max:
                correct1 = correct1+1
            # if ball_type_mapping[balltype2["type"][i]]==ball_type_mapping[fin_max] and ball_type_mapping[balltype3["type"][i]]==ball_type_mapping[fin_max] :
            #     simple_correct23 = simple_correct23+2
            # elif ball_type_mapping[balltype2["type"][i]]==ball_type_mapping[fin_max] or ball_type_mapping[balltype3["type"][i]]==ball_type_mapping[fin_max] :
            #     simple_correct23 = simple_correct23+1
            # if ball_type_mapping[balltype1["type"][i]]==ball_type_mapping[fin_max]:
            #     simple_correct1 = simple_correct1+1
            y_true.append(fin_max)
            # simple_y_true.append(ball_type_mapping[fin_max])

    print(correct23/(len(balltype2)+len(balltype3)))
    print(correct1/len(balltype1))
    print((correct23+correct1)/(len(balltype2)+len(balltype3)+len(balltype1)))

    # print(simple_correct23/(len(balltype2)+len(balltype3)))
    # print(simple_correct1/len(balltype1))
    # print((simple_correct23+simple_correct1)/(len(balltype2)+len(balltype3)+len(balltype1)))

    y1 = [i for i in balltype1["type"] if i!="發長球" and i!="發短球"]
    y2 = [i for i in balltype2["type"] if i!="發長球" and i!="發短球"]
    y3 = [i for i in balltype3["type"] if i!="發長球" and i!="發短球"]


    # simple_y1 = [ball_type_mapping[i] for i in balltype1["type"] if i!="發長球" and i!="發短球"]
    # simple_y2 = [ball_type_mapping[i] for i in balltype2["type"] if i!="發長球" and i!="發短球"]
    # simple_y3 = [ball_type_mapping[i] for i in balltype3["type"] if i!="發長球" and i!="發短球"]

    y1_mat = confusion_matrix(y_true, y1, labels=balltype_order)
    y2_mat = confusion_matrix(y_true, y2, labels=balltype_order)
    y3_mat = confusion_matrix(y_true, y3, labels=balltype_order)
    cnt = y1_mat+y2_mat+y3_mat
    ## print(mat)

    # simple_y1_mat = confusion_matrix(simple_y_true,simple_y1, labels = ["網前小球", "挑球", "平球", "切球", "殺球", "長球"])
    # simple_y2_mat = confusion_matrix(simple_y_true,simple_y2, labels = ["網前小球", "挑球", "平球", "切球", "殺球", "長球"])
    # simple_y3_mat = confusion_matrix(simple_y_true,simple_y3, labels = ["網前小球", "挑球", "平球", "切球", "殺球", "長球"])
    # simple_mat = simple_y1_mat+simple_y2_mat+simple_y3_mat
    ## print(simple_mat)

    ## ax = sns.heatmap(mat, annot=True, cmap='Blues', fmt="d")
    ax = sns.heatmap(cnt, annot=True, cmap='Blues', fmt="d")
    ## ax.set_title('{} Ball Types Confusion Matrix '.format(len(cnt)))
    ax.set_xlabel('Tagged Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels([i for i in range(1,len(cnt)+1)])
    ax.yaxis.set_ticklabels([i for i in range(1,len(cnt)+1)])

    ## Display the visualization of the Confusion Matrix.
    plt.show()

    df_y_true = pd.DataFrame(y_true, columns = ['type'])
    df_y_true.to_csv('major_vote.csv')

# A在下半場、B在上半場 ---> 0:上半場 1:下半場 
# B在下半場、A在上半場 ---> 1:上半場 0:下半場 
# court:0/1
def MAEofDeadbird(court=None):
    balltype1 = set1_1[set1_1["server"]==3]
    balltype2 = set1_2[set1_2["server"]==3]
    balltype3 = set1_3[set1_3["server"]==3]
    balltype1 = balltype1.reset_index(drop=True)
    balltype2 = balltype2.reset_index(drop=True)
    balltype3 = balltype3.reset_index(drop=True)

    correct23 = 0
    correct1 = 0

    c1 = 0
    c2 = 0
    c3 = 0
    y_true = []
    for i in range(0,len(balltype1)):
    #   對手場:出界，對手落地致勝，落點判斷失誤
        cnt = {"出界":0,"掛網":0,"未過網":0,"對手落地致勝":0,"落點判斷失誤":0 }

        if court==1:
            if (balltype1["player"][i]=="A"  and (balltype1["lose_reason"][i]=="掛網" or balltype1["lose_reason"][i]=="未過網")) or \
                (balltype1["player"][i]=="B" and (balltype1["lose_reason"][i]=="出界" or balltype1["lose_reason"][i]=="對手落地致勝" or balltype1["lose_reason"][i]=="落點判斷失誤")):
                c1 = c1+1
            if (balltype2["player"][i]=="A"  and (balltype2["lose_reason"][i]=="掛網" or balltype2["lose_reason"][i]=="未過網")) or \
                (balltype2["player"][i]=="B" and (balltype2["lose_reason"][i]=="出界" or balltype2["lose_reason"][i]=="對手落地致勝" or balltype2["lose_reason"][i]=="落點判斷失誤")):
                c2 = c2+1
            if (balltype3["player"][i]=="A"  and (balltype3["lose_reason"][i]=="掛網" or balltype3["lose_reason"][i]=="未過網")) or \
                (balltype3["player"][i]=="B" and (balltype3["lose_reason"][i]=="出界" or balltype3["lose_reason"][i]=="對手落地致勝" or balltype3["lose_reason"][i]=="落點判斷失誤")):
                c3 = c3+1
        else:
            if (balltype1["player"][i]=="B"  and (balltype1["lose_reason"][i]=="掛網" or balltype1["lose_reason"][i]=="未過網")) or \
                (balltype1["player"][i]=="A" and (balltype1["lose_reason"][i]=="出界" or balltype1["lose_reason"][i]=="對手落地致勝" or balltype1["lose_reason"][i]=="落點判斷失誤")):
                c1 = c1+1
            if (balltype2["player"][i]=="B"  and (balltype2["lose_reason"][i]=="掛網" or balltype2["lose_reason"][i]=="未過網")) or \
                (balltype2["player"][i]=="A" and (balltype2["lose_reason"][i]=="出界" or balltype2["lose_reason"][i]=="對手落地致勝" or balltype2["lose_reason"][i]=="落點判斷失誤")):
                c2 = c2+1
            if (balltype3["player"][i]=="B"  and (balltype3["lose_reason"][i]=="掛網" or balltype3["lose_reason"][i]=="未過網")) or \
                (balltype3["player"][i]=="A" and (balltype3["lose_reason"][i]=="出界" or balltype3["lose_reason"][i]=="對手落地致勝" or balltype3["lose_reason"][i]=="落點判斷失誤")):
                c3 = c3+1


        # for confusion matrix
        cnt[balltype1["lose_reason"][i]]+=1
        cnt[balltype2["lose_reason"][i]]+=1
        cnt[balltype3["lose_reason"][i]]+=1
        fin_max = max(cnt, key=cnt.get)
        if cnt[fin_max]!=0:
            if balltype2["lose_reason"][i]==fin_max and balltype3["lose_reason"][i]==fin_max :
                correct23 = correct23+2
            elif balltype2["lose_reason"][i]==fin_max or balltype3["lose_reason"][i]==fin_max :
                correct23 = correct23+1
            if balltype1["lose_reason"][i]==fin_max:
                correct1 = correct1+1
            y_true.append(fin_max)
    print(correct23/(c2+c3))
    print(correct1/c1)

    y1 = [i for i in balltype1["lose_reason"] ]
    y2 = [i for i in balltype2["lose_reason"] ]
    y3 = [i for i in balltype3["lose_reason"] ]

    y1_mat = confusion_matrix(y_true,y1, labels = ['出界', '掛網', '未過網', '對手落地致勝', '落點判斷失誤'])
    y2_mat = confusion_matrix(y_true,y2, labels = ['出界', '掛網', '未過網', '對手落地致勝', '落點判斷失誤'])
    y3_mat = confusion_matrix(y_true,y3, labels = ['出界', '掛網', '未過網', '對手落地致勝', '落點判斷失誤'])
    mat = y1_mat+y2_mat+y3_mat
    print(mat)

    ax = sns.heatmap(mat, annot=True, cmap='Blues', fmt="d")

    # ax.set_title('5 Dead Birds Confusion Matrix ')
    ax.set_xlabel('Tagged Values')
    ax.set_ylabel('Actual Values ')

    print(cnt.keys())
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels([i for i in range(1,6)])
    ax.yaxis.set_ticklabels([i for i in range(1,6)])

    ## Display the visualization of the Confusion Matrix.
    plt.show()

# A在下半場、B在上半場
# height:1/2(high/low)、player:A/B
def MAEofHeight(player=None, height = None):

    ball_height1 = set1_1[set1_1["hit_height"]==height]
    ball_height2 = set1_2[set1_2["hit_height"]==height]
    ball_height3 = set1_3[set1_3["hit_height"]==height]
    if player!=None:
        ball_height1 = ball_height1[ball_height1["player"]==player]
        ball_height2 = ball_height2[ball_height2["player"]==player]
        ball_height3 = ball_height3[ball_height3["player"]==player]

    ball_height1 = ball_height1.reset_index(drop=True)
    ball_height2 = ball_height2.reset_index(drop=True)
    ball_height3 = ball_height3.reset_index(drop=True)

    correct23 = 0
    correct1 = 0
    #用每個rally當作基準
    rally_num = set1_1.iloc[len(set1_1)-1]["rally"]
    for i in range(1,rally_num+1): 
        cnt ={k:0 for k in range(0,100)}
        ball_height1_rally = ball_height1[ball_height1["rally"]==i]
        ball_height2_rally = ball_height2[ball_height2["rally"]==i]
        ball_height3_rally = ball_height3[ball_height3["rally"]==i]
        ball_height1_rally = ball_height1_rally.reset_index(drop=True)
        ball_height2_rally = ball_height2_rally.reset_index(drop=True)
        ball_height3_rally = ball_height3_rally.reset_index(drop=True)
        # 看那些拍的ball height = height
        for j in range(0,len(ball_height1_rally)):
            cnt[ball_height1_rally["ball_round"][j]]= cnt[ball_height1_rally["ball_round"][j]]+1
        for j in range(0,len(ball_height2_rally)):
            cnt[ball_height2_rally["ball_round"][j]]= cnt[ball_height2_rally["ball_round"][j]]+1
        for j in range(0,len(ball_height3_rally)):
            cnt[ball_height3_rally["ball_round"][j]]= cnt[ball_height3_rally["ball_round"][j]]+1

        # 某一拍ball height = height的數量>2 -> gt
        for j in range(0,100):
            if cnt[j]>=2:
                for z in range(0,len(ball_height1_rally)):
                    if ball_height1_rally["ball_round"][z]==j:
                        correct1 = correct1+1
                for z in range(0,len(ball_height2_rally)):
                    if ball_height2_rally["ball_round"][z]==j:
                        correct23 = correct23+1
                for z in range(0,len(ball_height3_rally)):
                    if ball_height3_rally["ball_round"][z]==j:
                        correct23 = correct23+1
    print(correct23)
    print(correct1)
    print(correct23/(len(ball_height2)+len(ball_height3)))
    print(correct1/len(ball_height1))

# A在下半場、B在上半場
# player A/B
def MAEofPlayerLocation(player='A'):
    playerPosition1 = set1_1
    playerPosition2 = set1_2
    playerPosition3 = set1_3

    playerPosition1 = playerPosition1.reset_index(drop=True)
    playerPosition2 = playerPosition2.reset_index(drop=True)
    playerPosition3 = playerPosition3.reset_index(drop=True)
    error1 = []
    error23 = []
    # print(playerPosition1)
    for i in range(0,len(playerPosition1)): 
        #=="A":下半場, =="B":上半場
        if playerPosition1["player"][i]==player:
            location ="player_location"
        else:
            location = "opponent_location"

        # mapping on real court 

        img_coordinate1 = np.array([playerPosition1[location+"_x"][i], playerPosition1[location+"_y"][i], 1])
        world_coordinate1 = np.matmul(homography_matrix, img_coordinate1)
        img_coordinate2 = np.array([playerPosition2[location+"_x"][i], playerPosition2[location+"_y"][i], 1])
        world_coordinate2 = np.matmul(homography_matrix, img_coordinate2)
        img_coordinate3 = np.array([playerPosition3[location+"_x"][i], playerPosition3[location+"_y"][i], 1])
        world_coordinate3 = np.matmul(homography_matrix, img_coordinate3)
        gt =  (world_coordinate1[0]/world_coordinate1[2]+world_coordinate2[0]/world_coordinate2[2]+world_coordinate3[0]/world_coordinate3[2])/3 \
            +(world_coordinate1[1]/world_coordinate1[2]+world_coordinate2[1]/world_coordinate2[2]+world_coordinate3[1]/world_coordinate3[2])/3
        error23.append(abs(gt-world_coordinate2[0]/world_coordinate2[2]-world_coordinate2[1]/world_coordinate2[2])
                        +abs(gt-world_coordinate3[0]/world_coordinate3[2]-world_coordinate3[1]/world_coordinate3[2]))
        error1.append(abs(gt-world_coordinate1[0]/world_coordinate1[2]-world_coordinate1[1]/world_coordinate1[2]))

        # pixel in image

        # gt =  (playerPosition1[location+"_x"][i]+playerPosition2[location+"_x"][i]+playerPosition3[location+"_x"][i])/3\
        #     +(playerPosition1[location+"_y"][i]+playerPosition2[location+"_y"][i]+playerPosition3[location+"_y"][i])/3
        # error23.append(abs(gt-playerPosition2[location+"_x"][i]-playerPosition2[location+"_y"][i])
        #                 +abs(gt-playerPosition3[location+"_x"][i]-playerPosition3[location+"_y"][i]))
        # error1.append(abs(gt-playerPosition1[location+"_x"][i]-playerPosition1[location+"_y"][i]))
        # print(gt)

    print("MAE23 = ",sum(error23) / (len(playerPosition2)+len(playerPosition3)))#平均絕對誤差MAE
    print("MAE1 = ",sum(error1) / len(playerPosition1))#平均絕對誤差MAE


# B:上半場(標下半場球位置) A:下半場(標上半場球位置)
def MAEofShuttlecock(player=None):
    ballPosition1 = set1_1
    ballPosition2 = set1_2
    ballPosition3 = set1_3

    ballPosition1 = ballPosition1.reset_index(drop=True)
    ballPosition2 = ballPosition2.reset_index(drop=True)
    ballPosition3 = ballPosition3.reset_index(drop=True)
    error1 = []
    error23 = []
    # print(ballPosition1)
    court ="landing"
    for i in range(0,len(ballPosition1)): 
        if(ballPosition1["player"][i]==player):
            # mapping on real court 

            # img_coordinate1 = np.array([ballPosition1[court+"_x"][i], ballPosition1[court+"_y"][i], 1])
            # world_coordinate1 = np.matmul(homography_matrix, img_coordinate1)
            # img_coordinate2 = np.array([ballPosition2[court+"_x"][i], ballPosition2[court+"_y"][i], 1])
            # world_coordinate2 = np.matmul(homography_matrix, img_coordinate2)
            # img_coordinate3 = np.array([ballPosition3[court+"_x"][i], ballPosition3[court+"_y"][i], 1])
            # world_coordinate3 = np.matmul(homography_matrix, img_coordinate3)
            # gt =  (world_coordinate1[0]/world_coordinate1[2]+world_coordinate2[0]/world_coordinate2[2]+world_coordinate3[0]/world_coordinate3[2])/3 \
            #     +(world_coordinate1[1]/world_coordinate1[2]+world_coordinate2[1]/world_coordinate2[2]+world_coordinate3[1]/world_coordinate3[2])/3
            # error23.append(abs(gt-world_coordinate2[0]/world_coordinate2[2]-world_coordinate2[1]/world_coordinate2[2])
            #                 +abs(gt-world_coordinate3[0]/world_coordinate3[2]-world_coordinate3[1]/world_coordinate3[2]))
            # error1.append(abs(gt-world_coordinate1[0]/world_coordinate1[2]-world_coordinate1[1]/world_coordinate1[2]))
            
            # pixel in image
            gt =  (ballPosition1[court+"_x"][i]+ballPosition2[court+"_x"][i]+ballPosition3[court+"_x"][i])/3\
                +(ballPosition1[court+"_y"][i]+ballPosition2[court+"_y"][i]+ballPosition3[court+"_y"][i])/3
            error23.append(abs(gt-ballPosition2[court+"_x"][i]-ballPosition2[court+"_y"][i])
                            +abs(gt-ballPosition3[court+"_x"][i]-ballPosition3[court+"_y"][i]))
            error1.append(abs(gt-ballPosition1[court+"_x"][i]-ballPosition1[court+"_y"][i]))
        # print(gt)

    print("MAE23 = ",sum(error23) / (len(ballPosition2)+len(ballPosition3)))#平均絕對誤差MAE
    print("MAE1 = ",sum(error1) / len(ballPosition1))#平均絕對誤差MAE

# # ---------------------------------------------------------------------
# # ---------------------------------------------------------------------break time


def CalBreakTime():
    break_time = 0
    for i in range(0,len(RallySeg)):
        split_score = RallySeg["Score"][i].split('_')
        if split_score[0]=='1':
            break_time = break_time + (RallySeg["End"][i]-RallySeg["Start"][i] )
    # print((RallySeg["End"][len(RallySeg)-1]-RallySeg["Start"][0])/30 ) 
    # print(((RallySeg["End"][len(RallySeg)-1]-RallySeg["Start"][0])/30)%60 ) #seconds
    # print(((RallySeg["End"][len(RallySeg)-1]-RallySeg["Start"][0])/30)/60 ) # minutes
    # print(((RallySeg["End"][len(RallySeg)-1]-RallySeg["Start"][0])/30)%60 )
    seconds = (int(break_time/30))%60
    minutes = int((int(break_time/30)-seconds)/60)
    print("{}:{}".format(minutes, seconds))


if __name__=='__main__':
    pass