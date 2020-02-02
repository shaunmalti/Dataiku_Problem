import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold

from sklearn.metrics import accuracy_score
from sklearn import metrics

# other models
import lightgbm as lgb
pd.set_option('display.max_columns', 50)
from sklearn.utils import resample



def convertNominalFeatures(data):
    for datum in data:
        for col in datum.columns.values:
            if datum[col].dtype.name == 'object':
                le = LabelEncoder()
                datum[col] = le.fit_transform(datum[col])
    return data[0], data[1]


def replaceQMarks(data):
    QCols = ['MigCodeMSA', 'MigCodeRegDiff', 'MigCodeRegSame', 'MigResSunbelt', 'FatherBirthCountry', 'MotherBirthCountry', 'SelfBirthCountry', 'PrevState']
    # QCols = ['MigCodeMSA', 'MigCodeRegDiff', 'MigCodeRegSame', 'FatherBirthCountry', 'MotherBirthCountry', 'SelfBirthCountry']
    for datum in data:
        for col in QCols:
            if datum[col].value_counts().index.to_list()[0] == ' ?':
                datum.drop(col, axis=1, inplace=True)
            else:
                # instead of removing place unknown values as most occurring within column
                datum[col] = datum[col].replace(' ?', datum[col].value_counts().index.to_list()[0])
    return data[0], data[1]


def dropUnneeded(data):
    # dont check for numeric as some numeric cols are categorical
    # remove second industry code and occupation code as they are string versions of
    # the first ones seen in the dataset
    for datum in data:
        datum.drop(['IndustryCodeString', 'OccupationCodeString', 'InstanceWeight'], axis=1, inplace=True)

    return data[0], data[1]


def importData():
    train = pd.read_csv('./us_census_full/census_income_learn.csv')
    test = pd.read_csv('./us_census_full/census_income_test.csv')
    # allData = pd.DataFrame.merge(train, test)
    # allData = concatDf(train, test)
    return train, test

def dropMissingDataCols(train, test):
    # dropping colums with >90% missing data to see effect on accuracy
    train.drop(
        ['EducationLastWk', 'LabourUnion', 'UnempReas', 'PrevReg', 'PrevState', 'VeteranAdmQ'],
        axis=1, inplace=True)
    test.drop(
        ['EducationLastWk', 'LabourUnion', 'UnempReas', 'PrevReg', 'PrevState', 'VeteranAdmQ'],
        axis=1, inplace=True)
    return train, test


def binningRace(train, test):
    train['Race'] = train['Race'].replace([' Asian or Pacific Islander', ' Amer Indian Aleut or Eskimo',
                                           ' Black', ' Other'], 'Other')
    train['Race'] = train['Race'].replace([' White'], 'White')

    test['Race'] = test['Race'].replace([' Asian or Pacific Islander', ' Amer Indian Aleut or Eskimo',
                                         ' Black', ' Other'], 'Other')
    test['Race'] = test['Race'].replace([' White'], 'White')

    return train, test

def binningAge(train, test):
    train.loc[train['Age'] >= 40, 'NewAge'] = 4
    train.loc[(train['Age'] >= 25) & (train['Age'] < 30), 'NewAge'] = 3
    train.loc[(train['Age'] >= 30) & (train['Age'] < 35), 'NewAge'] = 2
    train.loc[(train['Age'] >= 35) & (train['Age'] < 40), 'NewAge'] = 1
    train.loc[train['Age'] < 25, 'NewAge'] = 0

    test.loc[test['Age'] >= 40, 'NewAge'] = 4
    test.loc[(test['Age'] >= 25) & (test['Age'] < 30), 'NewAge'] = 3
    test.loc[(test['Age'] >= 30) & (test['Age'] < 35), 'NewAge'] = 2
    test.loc[(test['Age'] >= 35) & (test['Age'] < 40), 'NewAge'] = 1
    test.loc[test['Age'] < 25, 'NewAge'] = 0

    train.drop('Age', axis=1, inplace=True)
    test.drop('Age', axis=1, inplace=True)
    return train, test


def binningWeeksWorked(train, test):
    train.loc[train['WeeksWorked'] >= 45, 'NewWeeksWorked'] = 5
    train.loc[(train['WeeksWorked'] >= 40) & (train['WeeksWorked'] < 45), 'NewWeeksWorked'] = 4
    train.loc[(train['WeeksWorked'] >= 35) & (train['WeeksWorked'] < 40), 'NewWeeksWorked'] = 3
    train.loc[(train['WeeksWorked'] >= 30) & (train['WeeksWorked'] < 35), 'NewWeeksWorked'] = 2
    train.loc[(train['WeeksWorked'] >= 25) & (train['WeeksWorked'] < 30), 'NewWeeksWorked'] = 1
    train.loc[train['WeeksWorked'] < 25, 'NewWeeksWorked'] = 0

    test.loc[test['WeeksWorked'] >= 45, 'NewWeeksWorked'] = 5
    test.loc[(test['WeeksWorked'] >= 40) & (test['WeeksWorked'] < 45), 'NewWeeksWorked'] = 4
    test.loc[(test['WeeksWorked'] >= 35) & (test['WeeksWorked'] < 40), 'NewWeeksWorked'] = 3
    test.loc[(test['WeeksWorked'] >= 30) & (test['WeeksWorked'] < 35), 'NewWeeksWorked'] = 2
    test.loc[(test['WeeksWorked'] >= 25) & (test['WeeksWorked'] < 30), 'NewWeeksWorked'] = 1
    test.loc[test['WeeksWorked'] < 25, 'NewWeeksWorked'] = 0

    train.drop('WeeksWorked', axis=1, inplace=True)
    test.drop('WeeksWorked', axis=1, inplace=True)
    return train, test

def downSampleMajorityClass(data):
    minorityDf = data[data['Target'] == 1]
    majorityDf = data[data['Target'] == 0]

    majorityDfDownsample = resample(majorityDf, replace=False, n_samples=len(minorityDf), random_state=1)

    return pd.concat([majorityDfDownsample, minorityDf])

def main():
    train, test = importData()
    # train, test = checkRemoveMissingData([train, test])
    # describe(train)

    train, test = dropUnneeded([train, test])
    train, test = replaceQMarks([train, test])
    train, test = convertNominalFeatures([train, test])
    # train, test = dropMissingDataCols(train, test)

    train, test = binningAge(train, test)
    train, test = binningWeeksWorked(train, test)

    # train = downSampleMajorityClass(train)

    # setting up training params
    y_data = train['Target']
    x_data = train.drop(['Target'], axis=1)
    y_test = test['Target']
    x_test = test.drop(['Target'], axis=1)


    lightGBMModel(x_data, y_data, x_test, y_test)

    kf = KFold(n_splits=10, shuffle=True, random_state=123)
    reg = LogisticRegression(solver='lbfgs', max_iter=500)
    cross_val_score(reg, x_data, y_data, cv=kf)
    preds = cross_val_predict(reg, x_test, y_test, cv=kf)
    print('Accuracy Logistic Reg: ', (accuracy_score(preds, y_test)) * 100)

    decTree(x_data, y_data, x_test, y_test)
    randFor(x_data, y_data, x_test, y_test)


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

def hyperParamTuningRF(x_data, y_data, x_test, y_test):
    print('****************************************')
    print('STARTED HYPERPARAM TUNING')
    print('****************************************')
    # trying out hyperparameter tuning
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    nEstimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    maxFeatures = ['auto', 'sqrt']
    maxDepth = [int(x) for x in np.linspace(10, 100, num=11)]
    maxDepth.append(None)
    minSampleSplit = [2, 5, 10]
    minSamplesLeaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': nEstimators,
                   'max_features': maxFeatures,
                   'max_depth': maxDepth,
                   'min_samples_split': minSampleSplit,
                   'min_samples_leaf': minSamplesLeaf,
                   'bootstrap': bootstrap}
    rfReg = RandomForestRegressor(random_state=123)
    rfRand = RandomizedSearchCV(estimator=rfReg, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                random_state=42, n_jobs=-1)
    # Fit the random search model
    rfRand.fit(x_data, y_data)
    print(rfRand.best_params_)

def lightGBMModel(x_data, y_data, x_test, y_test):
    # using lightgbm model to compare accuracy with normal rand forest
    train_x, valid_x, train_y, valid_y = train_test_split(
        x_data, y_data, test_size=0.33)

    train_data = lgb.Dataset(train_x, label=train_y)
    valid_data = lgb.Dataset(valid_x, label=valid_y)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'max_depth': 28,
        'num_estimators': 800
    }

    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    train_data,
                    num_boost_round=2000,
                    valid_sets=valid_data,
                    early_stopping_rounds=50)

    print('Starting predicting...')
    # predict
    y_prob = gbm.predict(x_test.values, num_iteration=gbm.best_iteration)
    y_pred = [round(x) for x in y_prob]
    print('Accuracy LGBM: ', (accuracy_score(y_pred, y_test)) * 100)

def logReg(x_data, y_data, x_test, y_test):
    reg = LogisticRegression(solver='lbfgs')
    reg.fit(x_data, y_data)
    preds = reg.predict(x_test)
    print('Accuracy Logistic Reg: ', (accuracy_score(preds, y_test)) * 100)

def decTree(x_data, y_data, x_test, y_test):
    # next try with decision tree
    d_t = DecisionTreeClassifier()
    # cross val
    d_t.fit(x_data, y_data)
    preds = d_t.predict(x_test)
    print('Accuracy DT Class: ', (accuracy_score(preds, y_test)) * 100)

def randFor(x_data, y_data, x_test, y_test):
    # final impl with random forests
    rf = RandomForestClassifier(n_estimators=800, min_samples_split=10, min_samples_leaf=2,
                           max_features='sqrt', max_depth=28, bootstrap=False)
    rf.fit(x_data, y_data)
    preds = rf.predict(x_test)
    print('Accuracy Rand For: ', (accuracy_score(preds, y_test)) * 100)


if __name__ == '__main__':
    main()