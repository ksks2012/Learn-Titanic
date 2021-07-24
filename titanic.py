import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn import linear_model


import sklearn.preprocessing as preprocessing

class Titanic():

    def __init__(self):
        self.train, self.test = self.read_data()

        self.rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
        self.scaler = preprocessing.StandardScaler()

        self.scale_df = None

        self.age_scale_param = None
        self.fare_scale_param = None

    
    def train_data(self):
        self.set_missing_age(self.train)
        self.set_Cabin_type(self.train)
        self.dummies_df = self.trans_Cabin(self.train)
        self.scale_df = self.scale_data(self.dummies_df)
        #self.clf = self.linear(self.scale_df)
        self.linear(self.scale_df)

    def test_data(self):
        self.set_missing_age(self.test, self.rfr)
        self.set_missing_fare(self.test)
        self.set_Cabin_type(self.test)
        self.dummies_df = self.trans_Cabin(self.test)
        self.scale_df = self.scale_data(self.dummies_df, self.scale_df)
        self.predict_survived(self.scale_df)

    def read_data(self):
        return pd.read_csv("train.csv"), pd.read_csv("test.csv")

    def set_missing_age(self, data, rfr = None):
        
        age_df = data[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
        known_age = age_df[age_df.Age.notna()].values
        unknown_age = age_df[age_df.Age.isna()].values

        y = known_age[:, 0]

        if rfr is None:
            X = known_age[:, 1:]
            self.rfr.fit(X, y)
        else:
            X = unknown_age[:, 1:]

        predictedAges = self.rfr.predict(unknown_age[:, 1:])
        data.loc[ (data.Age.isna()), 'Age' ] = predictedAges

        return 0

    def set_missing_fare(self, data):
        data.loc[(data.Fare.isnull()), 'Fare'] = 0

    def set_Cabin_type(self, data):
        data.loc[ (data.Cabin.notnull()), 'Cabin' ] = "Yes"
        data.loc[ (data.Cabin.isnull()), 'Cabin' ] = "No"

    def trans_Cabin(self, data):
    
        dummies_Cabin = pd.get_dummies(data['Cabin'], prefix= 'Cabin')

        dummies_Embarked = pd.get_dummies(data['Embarked'], prefix= 'Embarked')

        dummies_Sex = pd.get_dummies(data['Sex'], prefix= 'Sex')

        dummies_Pclass = pd.get_dummies(data['Pclass'], prefix= 'Pclass')

        dummies_data = pd.concat([data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
        dummies_data.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

        return dummies_data

    def scale_data(self, data, scaler = None):

        age_1d = data['Age'].values.reshape(-1, 1)
        fare_1d = data['Fare'].values.reshape(-1, 1)

        if(scaler is not None):
            self.age_scale_param = self.scaler.fit(age_1d)
            self.fare_scale_param = self.scaler.fit(fare_1d)

        data['Age_scaled'] = self.scaler.fit_transform(age_1d, self.age_scale_param)
        data['Fare_scaled'] = self.scaler.fit_transform(fare_1d, self.fare_scale_param)

        return data

    def linear(self, data):
        train_df = data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
        train_np = train_df.values

        # y即Survival结果
        y = train_np[:, 0]

        # X即特征属性值
        X = train_np[:, 1:]

        # fit到RandomForestRegressor之中
        self.clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6, solver='liblinear')
        self.clf.fit(X, y)

    def predict_survived(self, data):
        test_df = data.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
        predictions = self.clf.predict(test_df)
        result = pd.DataFrame({'PassengerId':self.test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
        result.to_csv("./gender_submission.csv", index=False)
        

if __name__ == "__main__":

    titanic = Titanic()

    titanic.train_data()

    titanic.test_data()

    print("Done")