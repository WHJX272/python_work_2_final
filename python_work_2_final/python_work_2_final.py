import numpy as np
import random
import copy
from matplotlib import pyplot as plt
from scipy.misc import derivative
'''
@Author : WHJX_2072 NianWen Zeng 20200440816
说明：这里的xbits上限为2048，这是因为numpy的字符存储被我设置为了2048位
如果想要更改，只需将第39、66、110行内的U后数字改了就行。似乎是任意设置的。
这里的dim是因为老师提供的函数只能接受2维的，若想更改看achley()函数的注释
'''
dim = 2                              # 个体取值维度,列
xbits = 64                           # 与DNA长度相同
pop_size = 1000                      # 种群长度，行
cross_rate = 0.8                     # 交叉（配）概率
mutation_rate = 0.003                # 编译（混合）概率
n_generations = 1000                 # 迭代次数
bounds = [-3,3]
dis = 0
lowc = 0.
highc = 0.

def achley(X): 
    #老师给的只要2种特征，选最大最小值在selecctParent(),fit_index = fit_index[::-1]不注释就是最小值，注释就找最大值
    #如果以后要改成多dim版本只要把X改成(pop,dim)就可以了
    x = X[0]
    y = X[1]
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) \
           - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) \
           + np.exp(1) + 20

def rastrigin(x,y):
    return 20+ x**2-10*np.cos(2*np.pi*x)+y**2-10*np.cos(2*np.pi*y)

def mhumps(x):
    return abs(-1 / ((x - 0.3) ** 2 + 0.01)
               + 1 / ((x - 0.9) ** 2 + 0.04) - 6)

def createpop(size,dim,xbits,bounds):
    #xbits 上限2048
    pop = np.empty((size,dim),dtype = np.dtype('U2048')) #这里设置接收的字符长度
    global dis 
    dis = (bounds[1] - bounds[0]) / (2**xbits)
    global lowc
    lowc = bounds[0]
    global highc
    highc = bounds[1]
    #这里的dis,lowc,highc，是为了之后的二转十进制用的
    for s in range(0,size):
        for d in range(0,dim):
            stre=""
            for xb in range(0,xbits):
                stre += str(random.randint(0,1))
            pop[s][d]=stre
    return pop

def DtoB(pop):#2转10
    pop_x,pop_y = pop.shape
    for x in range(0,pop_x):
        for y in range(0,pop_y):
            arr10 = int(pop[x][y],2)
            yy = arr10 * dis + lowc
            pop[x][y] = yy
    return pop

def BtoD(pop): #10转2 , 这里没用上，最后为了保证精度，我用深复制存了及格数组
    pop_x,pop_y = pop.shape
    newpop = np.empty((pop_x,pop_y),dtype = np.dtype('U2048'))
    for x in range(0,pop_x):
        for y in range(0,pop_y):
            yy = (float(pop[x][y])- lowc) / dis
            yy = bin(int(yy))
            yy = yy[2:]
            need0 = xbits -len(yy)
            zero = ""
            if need0 > 0:
                for i in range(0,need0):
                    zero += "0"
            yy = zero + yy
            newpop[x][y] = yy
    return newpop

def calfunc(pop,func): #计算函数值
    pop_x = pop.shape[0] #获取pop的行数，即pop_size
    pop_value = np.zeros((pop_x,1),np.float64)  #生成全为0的矩阵，之所以只有一列，因为是求y值，只需一个

    if(func == "achley"):
        for i in range(0,pop_x):
             pop_value[i,0] = achley([float(pop[i,0]),float(pop[i,1])]) #对每一列求值然后给到,这里的pop[i,:]是一行单独作为一个数组传入ackley函数计算
    elif(func == "rastrigin"):
        for i in range(0,pop_x):
             pop_value[i,0] = rastrigin(float(pop[i,0]),float(pop[i,1])) 
    return pop_value

def calFitvalue(pop_value):#计算适应值
    value = 1/(pop_value + 0.0001) #防止分母为0
    max_value = max(value)
    min_value = min(value)
    fitvalue = (value - min_value)/(max_value - min_value + 0.001) #防止最大值最小值相同导致为0,ps:这个fit数组里必有一个数是0
    return fitvalue

def selectParent(pop,fitvalue):#选父辈 
    fit_index = np.argsort(fitvalue, axis =0) #按行从小到大排列，返回对应元素排之前的下标
    fit_index = fit_index[::-1] #此时是从大到小的坐标了

    pop_x,pop_y = pop.shape #pop_x获取行数，pop_y获取列数，或许写成 pop_x = pop.shape[0] pop_y = pop.shape[1]更直观一点？
    parent_num = int(pop_x * 0.8) #//表示向下取整的最大数 ，不是所有人都可以交配的拜托,选了80%

    if parent_num == 0 : #不能为0
        parent_num = 1

    new_pop = np.zeros((parent_num,pop_y),dtype = np.dtype('U2048')) #列数不变，但新的pop行数小了，也就是个体少了

    for i in range(0,parent_num):
        new_pop[i,:] = pop[fit_index[i],:] #直接让选的父代成为父母中最优的那几个

    new_need_born = pop_x - parent_num # 需要生的孩子数

    return new_pop,new_need_born

def crossover(pop,cross_rate,need_born):
    pop_x,pop_y = pop.shape
    
    if pop_x % 2 == 0 : #如果种群数能被2整除
        for x in range(0,int(pop_x / 2)): #配对父母
            for y in range(0,pop_y):
               if np.random.rand() < cross_rate: #如果在cross_rate的范围里则交配
                    pos = random.randrange(0,xbits) #选择杂交位置
                    pop_first = str(pop[x][y]) #这里及以下是杂交
                    pop_second = str(pop[x+1][y])
                    tmp = pop_first[pos:]
                    pop_first = pop_first[:pos] + pop_second[pos:]
                    pop_second = pop_second[:pos] + tmp
                    pop[x][y] = pop_first
                    pop[x+1][y] = pop_second
    else :
        for x in range(0,int((pop_x - 1)/ 2)): #如果多了一个单身狗，他就被踢了，因为最后一个即使备选也是这个里面适应值最低的
            for y in range(0,pop_y):
               if np.random.rand() < cross_rate:
                    pos = random.randrange(0,xbits)
                    pop_first = str(pop[x][y])
                    pop_second = str(pop[x+1][y])
                    tmp = pop_first[pos:]
                    pop_first = pop_first[:pos] + pop_second[pos:]
                    pop_second = pop_second[:pos] + tmp
                    pop[x][y] = pop_first
                    pop[x+1][y] = pop_second 
    return pop[:need_born,] #这里是保证返回需要补足的子代数个子代

def mutation(pop,mutation_rate):
    pop_x,pop_y = pop.shape
    for x in range(0,pop_x):
        for y in range(0,pop_y):
            if np.random.rand() < mutation_rate: #如果在mutation_rate的范围内
                pos = random.randrange(0,xbits) #选择变异位置
                tmp = str(pop[x][y]) 
                if tmp[pos] == "1" : #变异
                    tmp = tmp[:pos] + "0" + tmp[pos+1:]
                else :
                    tmp = tmp[:pos] + "1" + tmp[pos+1:]
                pop[x][y] = tmp
    return pop

def bestFit(pop,fitvalue,func):
    fit_index = np.argsort(fitvalue, axis =0) #按行从小到大排列，返回对应元素排之前的下标
    fit_index = fit_index[::-1] #此时是从大到小的坐标了
    print(pop[fit_index[0],:])
    print("其函数值为：")
    y = calfunc(DtoB(pop[fit_index[0],:]),func)
    y = float(y)
    print(y)
    return y



def GA(func,xrange,xbits,xdim):
    global bounds
    bounds = xrange
    global dim
    dim = xdim
    pop = createpop(pop_size,dim,int(xbits),bounds)

    copypop = copy.deepcopy(pop) #记住要深复制！！！
    pop_10 = DtoB(copypop)

    pop_value = calfunc(pop_10,func)

    pop_fit = calFitvalue(pop_value)

    y = [] #用来存最佳y值
    for g in range(0,n_generations):
        pop_pa,need_born = selectParent(pop,pop_fit)

        copypop_pa = copy.deepcopy(pop_pa)

        cross_pop = crossover(copypop_pa,cross_rate,need_born)

        new_pop = np.concatenate((pop_pa,cross_pop))

        mutation_pop = mutation(new_pop,mutation_rate)

        pop = copy.deepcopy(mutation_pop)

        copypop = copy.deepcopy(pop) #记住要深复制！！！
        pop_10 = DtoB(copypop)

        pop_value = calfunc(pop_10,func)

        pop_fit = calFitvalue(pop_value)

        print('第{}代最优基因型是'.format(g+1))
        y.append(bestFit(pop,pop_fit,func))

    return y

#=========================================GA===========================================


#=======================================newton=========================================

def newton(func,x0,k):
    K = k
    x = x0
    X = []
    while(k):
        k-=1
        X.append(x)
        x = x - derivative(func,x,dx = 1e-6 ,n = 1) / (derivative(func,x,dx = 1e-6,n = 2) + 0.000001)
        print('第{}代迭代坐标为：'.format(K - k))
        print(x)
        print()
    plt.plot(X)
    plt.title("每次迭代后的坐标")
    plt.show()
    return x
#=======================================Main===========================================
if __name__=='__main__':
    plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] =False #负号显示

    #=========GA===============

#==========================achley==================
    r_min, r_max = -32.768, 32.768
    xaxis = np.arange(r_min, r_max, 2.0)
    yaxis = np.arange(r_min, r_max, 2.0)
    x, y = np.meshgrid(xaxis, yaxis)
    results = achley([x, y])
    figure = plt.figure()
    axis = figure.gca( projection='3d')
    axis.plot_surface(x, y, results, cmap='jet', shade= "false")
    plt.title("achley函数的图像为：")
    plt.show()
    plt.contour(x, y, results)
    plt.title("achley函数的等高线图为")
    plt.show()
    y = GA("achley",[-3,3],64,2) #此处设置部分参数
    plt.plot(y)
    plt.title("最优解的的变化")
    plt.show()
#==========================rastrigin==================

    r_min, r_max = -5, 5
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)
    x, y = np.meshgrid(xaxis, yaxis)
    results1 = rastrigin(x, y)
    figure = plt.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results1, cmap='jet', shade="false")
    plt.title("rastrigin函数的图像为：")
    plt.show()

    plt.contour(x, y, results1)
    plt.title("rastrigin函数的等高线图为：")
    plt.show()
    y = GA("rastrigin",[-30,30],64,2) #此处设置部分参数
    plt.plot(y)
    plt.title("最优解的的变化")
    plt.show()

    #========newton============
    x = np.arange(-5, 5, 0.01)
    y = mhumps(x)
    plt.plot(x, y)
    
    plt.title("mhumps的图像为：")
    plt.show()

    # x0初始坐标
    x0 = 0.6
    # 求解
    point = newton(mhumps, x0, 1000) #此处设置参数
    # 输出坐标与极值
    print(f"坐标为： = {point}")
    print(f"函数值为：{mhumps(point)}")