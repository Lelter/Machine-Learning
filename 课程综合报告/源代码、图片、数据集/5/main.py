import numpy as np
import random
import matplotlib.pyplot as plt


色泽 = np.array(["青绿", "乌黑", "乌黑", "青绿", "浅白", "青绿", "乌黑", "乌黑",
               "乌黑", "青绿", "浅白", "浅白", "青绿", "浅白", "乌黑", "浅白", "青绿"])
根蒂 = np.array(["蜷缩", "蜷缩", "蜷缩", "蜷缩", "蜷缩", "稍蜷", "稍蜷", "稍蜷",
               "稍蜷", "硬挺", "硬挺", "蜷缩", "稍蜷", "稍蜷", "稍蜷", "蜷缩", "蜷缩"])
敲声 = np.array(["浊响", "沉闷", "浊响", "沉闷", "浊响", "浊响", "浊响", "浊响",
               "沉闷", "清脆", "清脆", "浊响", "浊响", "沉闷", "浊响", "浊响", "沉闷"])
纹理 = np.array(["清晰", "清晰", "清晰", "清晰", "清晰", "清晰", "稍糊", "清晰",
               "稍糊", "清晰", "模糊", "模糊", "稍糊", "稍糊", "清晰", "模糊", "稍糊"])
脐部 = np.array(["凹陷", "凹陷", "凹陷", "凹陷", "凹陷", "稍凹", "稍凹", "稍凹",
               "稍凹", "平坦", "平坦", "平坦", "凹陷", "凹陷", "稍凹", "平坦", "稍凹"])
触感 = np.array(["硬滑", "硬滑", "硬滑", "硬滑", "硬滑", "软粘", "软粘", "硬滑",
               "硬滑", "软粘", "硬滑", "软粘", "硬滑", "硬滑", "软粘", "硬滑", "硬滑"])
密度 = np.array([0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437,
               0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719])
含糖率 = np.array([0.460, 0.376, 0.264, 0.318, 0.2215, 0.237, 0.149, 0.211,
                0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103])
好瓜 = np.array(["好", "好", "好", "好", "好", "好", "好", "好",
               "坏", "坏", "坏", "坏", "坏", "坏", "坏", "坏", "坏"])


attrArr = [色泽, 根蒂, 敲声, 纹理, 脐部, 触感, 密度, 含糖率]
Strattr = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖率"]

dict={"青绿":0, "乌黑":0.5, "浅白":1, "蜷缩":0, "稍蜷":0.5, "硬挺":1, "浊响":0, "沉闷":0.5,"清脆":1,
      "清晰":0,"稍糊":0.5,"模糊":1,"凹陷":0,"稍凹":0.5,"平坦":1,"硬滑":0, "软粘":1,"好":1,"坏":0}
mid_node_num = int(input("输入隐层神经元数"))
inputLevel=[] #用来输入元节点，此处固定8个
outputLevel=[] #用来输出神经元节点 固定2个
midLevel=[]    #用来存储神经元节点

u = 0.1  # 学习率

class nerveNode(object):
    def __init__(self):
        self.gap=random.random()
        #阈值
        self.w =[]
        for i in range(10):
            self.w.append(random.random())
        #每个节点往下走一层走的权重,表示每层最大有10个神经节点
        #初始随机赋值

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def Init_LevelNode():
    mid_node_num=int(input("输入隐层神经节点个数"))
    for i in range(mid_node_num):
        midLevel.append(nerveNode())
    for i in range(8):
        inputLevel.append(nerveNode())
    for i in range(2):
        outputLevel.append(nerveNode())

def input_case(num): #返回第num个例子
    result=[]
    for i in range(len(attrArr)):
        if i<6:  #离散的点
            result.append(dict[attrArr[i][num]])
        else:
            result.append(attrArr[i][num])
    if 好瓜[num]=="好":
        result.append(1)
    else:
        result.append(0)

    return result




def cal_y(k): #传入输入参数list  计算输出层最终的输出
    mid = []
    result=[]
    for i in range(mid_node_num):  #对于每一个隐层的点
        y=0
        for j in range(len(inputLevel)):
            y=y+k[j]*inputLevel[j].w[i]  #第j个点的输入*第j个点的第i个权重
        mid.append(sigmoid(y-midLevel[i].gap))
    for i in range(len(outputLevel)):
        y=0
        for j in range(len(midLevel)):
            y=y+mid[j]*midLevel[j].w[i]
        result.append(sigmoid(y-outputLevel[i].gap))
    return result

def cal_Gj(k):#计算输出层神经元梯度  cal_yj为输出层最终的输出  k为输入层输入
    #计算k
    result=[]
    cal_yj=cal_y(k)
    for j in range(len(outputLevel)):
       result.append( cal_yj[j] * (1 - cal_yj[j]) * (k[8+j] - cal_yj[j]))

    return result

def cal_Eh(h,k): #第h个点的Eh
    Dh=cal_Bh(h,k)
    re=0
    for j in range(len(outputLevel)):
        re=re+midLevel[h].w[j]*cal_Gj(k)[j]
    return Dh*(1-Dh)*re


def cal_Bh(h,k):#k传入输入参数list,结果为隐层第h个输出
    mid=[]
    for i in range(mid_node_num):  # 对于每一个隐层的点
        y = 0
        for j in range(len(inputLevel)):
            y = y + k[j] * inputLevel[j].w[i]  # 第j个点的输入*第j个点的第i个权重
        mid.append(y)
    for i in range(len(mid)):
        mid[i]=sigmoid(mid[i]-midLevel[i].gap)
    return mid[h]

def update(Bh,Gj,Eh,k,flag): #flag==1累计BP GJ,EH,x
    if flag==0:
        for j in range(len(outputLevel)):
            outputLevel[j].gap = outputLevel[j].gap - u * Gj[j]  # 阈值更新

        for h in range(len(midLevel)):
            midLevel[h].gap = midLevel[h].gap - u * Eh[h]  # 阈值更新
            for j in range(len(outputLevel)):  # 权
                midLevel[h].w[j] = midLevel[h].w[j] + u * Gj[j] * cal_Bh(h, k)

        for i in range(len(inputLevel)):
            for h in range(mid_node_num):
                inputLevel[i].w[h] = inputLevel[i].w[h] + u * Eh[h] * k[i]
    else:
        gj=np.array(Gj).sum()/ 17
        Eh=list(np.array(Eh).sum(axis=0)/ 17)

        for j in range(len(outputLevel)):
            outputLevel[j].gap = outputLevel[j].gap - u * gj  # 阈值更新

        for h in range(len(midLevel)):
            midLevel[h].gap = midLevel[h].gap - u * Eh[h]  # 阈值更新
            for j in range(len(outputLevel)):  # 权
                midLevel[h].w[j] = midLevel[h].w[j] + u * gj * Bh[h]
        for i in range(len(inputLevel)):
            for h in range(mid_node_num):
                inputLevel[i].w[h] = inputLevel[i].w[h] + u * Eh[h] * k[i]




def cal_Ek(k,cal_y):
    re=0
    for j in range(len(outputLevel)):
        re=re+(k[8+j]-cal_y[j])*(k[8+j]-cal_y[j])
    return re



ekx=[]
arr=[1,3,5,7,9,11,13,15,0,2,4,6,8,10,12,14,16]
def fun():
    n = 0
    counter = 0
    for i in range(8):
        inputLevel.append(nerveNode())
    for j in range(1):
        outputLevel.append(nerveNode())  # 初始化
    for h in range(mid_node_num):
        midLevel.append(nerveNode())
    Ekarr=[]
    while counter<2000:
        k = input_case(arr[n])
        calY = cal_y(k)
        Gj = cal_Gj(k)
        Eh = []
        for i in range(mid_node_num):
            Eh.append(cal_Eh(i, k))

        update( [],Gj, Eh, k,0)
        Ekarr.append( cal_Ek(k, calY))

        n = n + 1
        if n >=16:
            n = 0
            counter += 1
            Ek=np.array(Ekarr).sum() / 17
            Ekarr.clear()
            print(counter, Ek)
            #if counter==1999:

            ekx.append(Ek)
    plt.figure()
    plt.scatter(range(len(ekx)),ekx, c="red")
    plt.show()

def Accumulate():
    for i in range(8):
        inputLevel.append(nerveNode())
    for j in range(1):
        outputLevel.append(nerveNode())  # 初始化
    for h in range(mid_node_num):
        midLevel.append(nerveNode())
    ek=1
    s=0
    ekx.clear()
    while s <2000:
        Ek=[]
        GJ=[]
        EH=[]
        x=[]
        Bh=[]
        s+=1
        for n in range(17):
            k = input_case(arr[n])
            calY = cal_y(k)
            Gj = cal_Gj(k)
            bh=[]
            Eh = []
            for i in range(mid_node_num):
                Eh.append(cal_Eh(i, k))
                bh.append(cal_Bh(i,k))
            Bh.append(bh)
            GJ.append(Gj)
            EH.append(Eh)
            x.append(k)
            Ek.append(cal_Ek(k, calY))
        ek=np.array(Ek).sum()/len(Ek)
        Bh = list(np.array(Bh).sum(axis=0)/ 17 ) #
        x=list(np.array(x).sum(axis=0)/ 17)  #
        update(Bh, GJ, EH, x, 1)
        print(ek)
        if s==2000:
            print(" ")
        ekx.append(ek)
    plt.figure()
    plt.scatter(range(len(ekx)),ekx,c="red")
    plt.show()


Accumulate()
#fun()

# plt.figure()
# plt.plot(range(2,10),[0.04751,0.05784,0.05426,0.04243,0.04243,0.03683,0.045,0.0438], c="red")

# plt.show()