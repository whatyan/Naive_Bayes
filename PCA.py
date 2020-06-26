import pandas as pd

voice_data=pd.read_csv('voice.csv')
x=voice_data.iloc[:,:-1]
y=voice_data.iloc[:,-1]

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
imp=SimpleImputer(missing_values=0,strategy='mean')
x=imp.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

#线性归一化
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
scaler1.fit(x_train)
x_train_std = scaler1.transform(x_train)
x_test_std = scaler1.transform(x_test)

#特征提取(PCA)
from sklearn.decomposition import PCA
pca_std = PCA(n_components = 11).fit(x_train_std)
x_train_std = pca_std.transform(x_train_std)
x_test_std = pca_std.transform(x_test_std)

#高斯过程
from sklearn.naive_bayes import GaussianNB
gnb_std = GaussianNB()
fit_std = gnb_std.fit(x_train_std, y_train)

#结果显示
from sklearn import metrics
pred_test_std = gnb_std.predict(x_test_std)
print('标准化后进行特征提取：')
print('准确率:','{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))
