import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

train=pd.read_csv("C:\\Users\\iPC\\PycharmProjects\\untitled2\\Assignment\\week 6\\train.csv")
test=pd.read_csv("test.csv")
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)

train.head()
train.describe()
train.info()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()


for column in train.columns:
        if train[column].dtype == np.number:
            continue
        train[column] = le.fit_transform(train[column])
for column in test.columns:
        if test[column].dtype == np.number:
            continue
        test[column] = le.fit_transform(test[column])

train['Attrition'].value_counts()


dataset1=train.copy()
dataset2=test.copy()
dataset1.drop(['Id','EmployeeNumber'],axis=1,inplace=True)
dataset2.drop(['Id','EmployeeNumber'],axis=1,inplace=True)


X=dataset1.loc[:,dataset1.columns!='Attrition']
y=dataset1['Attrition']



# Plot the Correlation map to see how features are correlated with target: Attrtition
corr_matrix = dataset1.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corr_matrix, vmax=0.9, square=True)
plt.show()


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X)
X_test = sc_X.fit_transform(dataset2)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y.values.reshape(-1,1))

import xgboost as xgb
data_dmatrix = xgb.DMatrix(data=X_train,label=y)
xgb_model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
xgb_model.fit(X_train, y)
y_pred = xgb_model.predict(X_test)


from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(solver='lbfgs',max_iter=2000,C=0.5,penalty='l2',random_state=1)
log_reg.fit(X_train,y)
y_pred = log_reg.predict(X_test)


log_reg = LogisticRegression(C = 1,max_iter=1000)
log_reg.fit(X_train,y)
print('For Logistic Regression')
y_pred = log_reg.predict(X_test)
#score = roc_auc_score(Y_train, log_reg.predict_proba(X_train)[:,1])
#print('Train roc_auc_score:',score)
#score = roc_auc_score(Y_test, log_reg.predict_proba(X_test)[:,1])
#print("Test roc_auc_score:",score)

y_Pred = pd.DataFrame(test['Id'])
y_Pred=y_Pred.set_index(test['Id'])

y_Pred['Attrition']=y_pred

y_Pred.to_csv("C:\\Users\\iPC\\submission.csv")



