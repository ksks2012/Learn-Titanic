import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn import linear_model


import sklearn.preprocessing as preprocessing

def read_data():
    return pd.read_csv("train.csv"), pd.read_csv("test.csv") 


def set_missing_ages(data):

    global rfr

    global run

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = data[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    #print(age_df)

    #print(age_df[age_df.Age.notna()].values)

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notna()].values
    unknown_age = age_df[age_df.Age.isna()].values

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    

    # fit到RandomForestRegressor之中

    #print(X)

    #print(known_age)
    #rfr = RandomForestRegressor()
    if(run == 0):
        X = known_age[:, 1:]
        rfr.fit(X, y)
        run = run + 1
    else :
        X = unknown_age[:, 1:]

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:])
    
    # 用得到的预测结果填补原缺失数据
    data.loc[ (data.Age.isna()), 'Age' ] = predictedAges 

def set_Cabin_type(data):
    data.loc[ (data.Cabin.notnull()), 'Cabin' ] = "Yes"
    data.loc[ (data.Cabin.isnull()), 'Cabin' ] = "No"

def trans_Cabin(data):
    
    dummies_Cabin = pd.get_dummies(data['Cabin'], prefix= 'Cabin')

    dummies_Embarked = pd.get_dummies(data['Embarked'], prefix= 'Embarked')

    dummies_Sex = pd.get_dummies(data['Sex'], prefix= 'Sex')

    dummies_Pclass = pd.get_dummies(data['Pclass'], prefix= 'Pclass')

    df = pd.concat([data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    return df

def scale_df(df):

    global age_scale_param
    global fare_scale_param

    #Age = data.iloc['Age'].values

    scaler = preprocessing.StandardScaler()

    if(run == 0):
        age_scale_param = scaler.fit(df['Age'])
        fare_scale_param = scaler.fit(df['Fare'])

    df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)

def linear(df):
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.values

    # y即Survival结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]

    # fit到RandomForestRegressor之中
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)

    return clf

def pre_processing(data):

    global run

    set_missing_ages(data)

    set_Cabin_type(data)

    df = trans_Cabin(data)

    scale_df(df)

    return df


if __name__ == "__main__":

    global rfr

    global run

    global age_scale_param
    global fare_scale_param

    age_scale_param = None
    fare_scale_param = None

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
   
    run = 0

    train, test = read_data();

    #train
    df = pre_processing(train)

    clf = linear(df)

    #test
    #print(test.info())
    test.loc[ (test.Fare.isnull()), 'Fare' ] = 0

    test_df = pre_processing(test)


    test_df = test_df.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(test_df)
    result = pd.DataFrame({'PassengerId':test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
    result.to_csv("./gender_submission.csv", index=False)
