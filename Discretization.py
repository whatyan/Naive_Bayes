#将数据量化后进行操作
import numpy as np
import pandas as pd

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
    for i in range(len(female_rand_array)):
        female_rand_array[i] += 1584
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

def discretization(file_path):
    x , y = load_data(file_path)
    #数据离散化
    N = 15
    min_vector = x.min(axis = 0) #得到列的最小值向量
    max_vector = x.max(axis = 0) #得到列的最大值向量
    diff_vector = (max_vector - min_vector) / N #得到列的最大值减最小值向量
    x_new = []
    for i in range(len(x)):
        line = np.array((x[i] - min_vector) / diff_vector).astype(int)
        x_new.append(line)

    #对缺失值进行填补
    from sklearn.impute import SimpleImputer
    imp=SimpleImputer(missing_values=0,strategy='mean')
    x_new = imp.fit_transform(x_new).astype(int)
    return x_new, y

#计算male和fmale下每个属性下的概率
def get_para(x_train_male, x_train_female):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    #计算每一个属性下的P(X|C)
    p_male, p_female = {}, {}
    for i in range(20):
        feature_values_male = np.array(x_train_male[:, i]).flatten() #获取第i个属性下的所有样本值
        feature_values_female = np.array(x_train_female[:, i]).flatten() #获取第i个属性下的所有样本值
        l_male = []
        l_female = []
        for value in feature_values_male:
            l_male.append(value)
        for value in feature_values_female:
            l_female.append(value)
        a, b = {}, {}
        for j in x:
            a[j] = np.log((l_male.count(j) + 1) / (len(l_male) + 20)) #拉普拉斯修正
            b[j] = np.log((l_female.count(j) + 1) / (len(l_female) + 20))
        p_male[i] = a
        p_female[i] = b
    return p_male, p_female

def Bayes(x_test, p_male, p_female):
    result = []
    r1 = r2 = 0
    for i in range(len(x_test)):
        for j in range(len(x_test[i])):
            r1 += p_male[j][x_test[i][j]]
            r2 += p_female[j][x_test[i][j]]
        if(r1 > r2):
            result.append('male')
        else:
            result.append('female')
        r1 = r2 = 0
    return result


def main():
    #得到训练集和测试集
    x_new, y = discretization('voice.csv')
    x_train, y_train, x_test, y_test = divid_data(x_new, y)
    #训练样本分男女
    x_train_male = x_train[0:1109]
    x_train_female = x_train[1109:2218]

    p_male, p_female = get_para(x_train_male, x_train_female)
    result = Bayes(x_test, p_male, p_female)
    count = count_0 = count_1 = 0
    for i in range(len(result)):
        if(result[i] == y_test[i]):
            count = count + 1
    for i in range(int(len(result)/2)):
        if(result[i] == y_test[i]):
            count_0 = count_0 + 1
        if(result[i + 475] == y_test[i + 475]):
            count_1 = count_1 + 1
    accuracy_rate = count / len(result)
    accuracy_rate_male = 2 * count_0 / len(result)
    accuracy_rate_female = 2 * count_1 / len(result)
    return accuracy_rate, accuracy_rate_male, accuracy_rate_female

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
    print('total:', x)
    print('male:', y)
    print('female', z)
