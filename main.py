import numpy as np
import pandas as pd
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#加载数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    attribute = data.values[:, :-1] #获取所有样本属性的数据,3168*20
    label = data.values[:, -1] #获取所有样本标签,3168*1
    return attribute,label

#划分数据,7:3
def divid_data(attribute, label):
    #male
    male_data = attribute[0:1584]
    male_rand_array = np.arange(male_data.shape[0])
    np.random.shuffle(male_rand_array)
    male_train_attribute = attribute[male_rand_array[0:1109]]
    male_test_attribute = attribute[male_rand_array[1109:1584]]
    male_train_label = label[male_rand_array[0:1109]]
    male_test_label = label[male_rand_array[1109:1584]]
    #female
    female_data = attribute[1584:3168]
    female_rand_array = np.arange(female_data.shape[0])
    for i in range(len(female_rand_array)): female_rand_array[i] += 1584
    np.random.shuffle(female_rand_array)
    female_train_attribute = attribute[female_rand_array[0:1109]]
    female_test_attribute = attribute[female_rand_array[1109:1584]]
    female_train_label = label[female_rand_array[0:1109]]
    female_test_label = label[female_rand_array[1109:1584]]
    #male and fmale
    train_data = np.append(male_train_attribute, female_train_attribute, axis = 0)
    train_label = np.append(male_train_label, female_train_label, axis = 0)
    test_data = np.append(male_test_attribute, female_test_attribute, axis = 0)
    test_label = np.append(male_test_label, female_test_label, axis = 0)
    return train_data, train_label, test_data, test_label

#通过训练集计算男性和女性下各个属性的均值和标准差
#返回参数：male：para_male,female:para_female
def get_para(train_data):
    para_male = {} #male下的参数
    para_female = {} #female下的参数
    for i in range(20): #对20个属性分别求均值和方差
        mean_male = train_data[0:1109, i].mean()
        std_male = train_data[0:1109, i].std()
        para_male[i] = (mean_male, std_male)
        mean_female = train_data[1109:2218, i].mean()
        std_female = train_data[1109:2218, i].std()
        para_female[i] = (mean_female, std_female)
    return para_male, para_female

def gaussian(x, mean, std):
    return 1 / (math.sqrt(math.pi * 2) * std) * math.exp((-(x - mean) ** 2) / (2 * std * std))

#计算P(x|C)
def P(attribute_index, x, continuousPara):
    (mean, std) = continuousPara[attribute_index]
    return gaussian(x, mean, std)

#通过朴素贝叶斯根据属性预测结果
def Beyes(X, para_male, para_female):
    prediction = []
    row = len(X) #行数，包含所有属性
    for i in range(row):
        col = len(X[i])
        P_male = P_female = 0
        for j in range(col):
            P_male += math.log(P(j,X[i][j],para_male))
            P_female += math.log(P(j, X[i][j], para_female))
        if P_male > P_female:
            prediction.append('male')
        else:
            prediction.append('female')
    return prediction

#绘制混淆矩阵图
def show_result(cm):
    plt.figure(figsize = (6, 5))
    plt.imshow(cm)
    plt.title('Result', fontsize = 'xx-large', fontweight = 'heavy')
    plt.colorbar()
    labels = ['male', 'fmale']
    xlocations = np.array(range(len(labels)))
    #刻度
    plt.xticks(xlocations, labels)
    plt.yticks(xlocations, labels)
    #标签
    plt.xlabel('Prediction', size = 15)
    plt.ylabel('Truth', size = 15)
    #根据混淆矩阵写入图中的数据
    for first_index in range(len(cm)):
        for second_index in range(len(cm[first_index])):
            plt.text(first_index, second_index, format(cm[first_index][second_index],'.3f'),
                     horizontalalignment = "center", fontsize = 18)
    plt.tight_layout()
    plt.show()

def main():
    attribute, label = load_data('voice.csv')
    train_data, train_label, test_data, test_label = divid_data(attribute, label)
    para_male, para_female = get_para(train_data)
    prediction = Beyes(test_data, para_male, para_female)
    cm = confusion_matrix(test_label, prediction, labels = ['male', 'female']) #得出混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] #标准化,将数据化为概率
    accuracy_rate = (cm[0][0] + cm[1][1]) / len(test_label)
    accuracy_rate_male = cm_normalized[0][0]
    accuracy_rate_female = cm_normalized[1][1]
    #show_result(cm_normalized)
    return accuracy_rate,accuracy_rate_male,accuracy_rate_female

if __name__ == '__main__':
    n = input("训练次数：")
    accuracy_rate = []
    accuracy_rate_male = []
    accuracy_rate_female = []
    for i in range(int(n)):
        a, b, c = main()
        accuracy_rate.append(a)
        accuracy_rate_male.append(b)
        accuracy_rate_female.append(c)
    x = np.mean(accuracy_rate)
    y = np.mean(accuracy_rate_male)
    z = np.mean(accuracy_rate_female)
    cm = [[y, 1-z],[1-y,z]]
    show_result(cm)
    print('total:', x)
    print('male:', y)
    print('female', z)






