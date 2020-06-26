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

    #female
    female_data = attribute[1584:3168]
    female_rand_array = np.arange(female_data.shape[0])
    for i in range(len(female_rand_array)):
        female_rand_array[i] += 1584
    np.random.shuffle(female_rand_array)
    female_train_attribute = attribute[female_rand_array[0:1109]]
    #计算每个属性的标准差
    male_std_vector = []
    female_std_vector = []
    for i in range(20):
        male_std_vector.append(male_train_attribute[:, i].std())
        female_std_vector.append(female_train_attribute[:, i].std())
    #计算除数
    male_div = []
    female_div = []
    for i in range(len(male_std_vector)):
        if(male_std_vector[i] > female_std_vector[i]):
            male_div.append( male_std_vector[i] / female_std_vector[i])
            female_div.append(1)
        else:
            female_div.append( female_std_vector[i] / male_std_vector[i])
            male_div.append(1)

    male_mean_vector = male_train_attribute.mean(axis = 0) #男性列的平均值
    male_attribute_new = []
    for i in range(20):
        line = np.array((male_train_attribute[:, i] - male_mean_vector[i])
                        / male_div[i] + male_mean_vector[i])
        male_attribute_new.append(line)
    for i in range(20):
        male_train_attribute[: , i] = male_attribute_new[i]

    male_test_attribute = attribute[male_rand_array[1109:1584]]
    male_train_label = label[male_rand_array[0:1109]]
    male_test_label = label[male_rand_array[1109:1584]]


    female_mean_vector = female_train_attribute.mean(axis = 0) #列的平均值
    female_attribute_new = []
    for i in range(20):
        line = np.array((female_train_attribute[:, i] - female_mean_vector[i]) / female_div[i] + female_mean_vector[i])
        female_attribute_new.append(line)
    for i in range(20):
        female_train_attribute[:, i] = female_attribute_new[i]

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
#返回参数：male：continuousPara_1,female:continuousPara_2
def get_para(train_data):
    continuousPara_1 = {} #male下的参数
    continuousPara_2 = {} #female下的参数
    for i in range(20): #对20个属性分别求均值和方差
        mean_male = train_data[0:1109, i].mean()
        std_male = train_data[0:1109, i].std()
        continuousPara_1[i] = (mean_male, std_male)
        mean_female = train_data[1109:2218, i].mean()
        std_female = train_data[1109:2218, i].std()
        continuousPara_2[i] = (mean_female, std_female)
    return continuousPara_1, continuousPara_2

def gaussian(x, mean, std):
    a = 1 / (math.sqrt(math.pi * 2) * std) * math.exp((-(x - mean) ** 2) / (2 * std * std))
    if(a != 0):
        return a
    else:
        return math.exp(-80)
#计算P(x|C)
def P(attribute_index, x, continuousPara):
    (mean, std) = continuousPara[attribute_index]
    return gaussian(x, mean, std)

#通过朴素贝叶斯根据属性预测结果
def Beyes(X, continuousPara_1, continuousPara_2):
    prediction = []
    row = len(X) #行数，包含所有属性
    for i in range(row):
        col = len(X[i])
        P_male = P_female = 0
        for j in range(col):
            P_male += math.log(P(j, X[i][j], continuousPara_1))
            P_female += math.log(P(j, X[i][j], continuousPara_2))
        if P_male > P_female:
            prediction.append('male')
        else:
            prediction.append('female')
    return prediction

#绘制混淆矩阵
def show_result(cm):
    plt.figure(figsize = (8, 6))
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
    continuousPara_1, continuousPara_2 = get_para(train_data)
    prediction = Beyes(test_data, continuousPara_1, continuousPara_2)
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













