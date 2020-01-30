import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

pd.set_option('display.max_columns', 50)

def convertNominalFeatures(data):
    for col in data.columns.values:
        if data[col].dtype.name == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    return data

def replaceQMarks(data):
    # TODO check if changing these to mode/median improves accuracy
    QCols = ['MigCodeMSA', 'MigCodeRegDiff', 'MigCodeRegSame', 'MigResSunbelt', 'FatherBirthCountry', 'MotherBirthCountry', 'SelfBirthCountry', 'PrevState']
    for col in QCols:
        if data[col].value_counts().index.to_list()[0] == ' ?':
            data = data.drop([col], axis=1)
            # data[col] = data[col].replace(' ?', data[col].value_counts().index.to_list()[1])
        else:
            data[col] = data[col].replace(' ?', data[col].value_counts().index.to_list()[0])
    return data

def checkNulls(data):
    print(data.isnull().sum())

def dropUnneeded(data):
    # dont check for numeric as some numeric cols are categorical
    # remove second industry code and occupation code as they are string versions of
    # the first ones seen in the dataset
    return data.drop(['IndustryCodeString', 'OccupationCodeString', 'InstanceWeight'], axis=1)

def checkTargetSplit(data):
    plt.figure()
    sns.countplot(data['Target'])
    # plt.show()

def describe(data):
    checkTargetSplit(data)

    # correlationPlot(data)

def correlationPlot(data):
    k = 15  # number of variables for heatmap
    corrMatrix = data.corr()
    cols = corrMatrix.nlargest(k, 'Target')['Target'].index
    cm = np.corrcoef(data[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()

def concatDf(train, test):
    return pd.concat([train, test])

def importData():
    train = pd.read_csv('./us_census_full/census_income_learn.csv')
    test = pd.read_csv('./us_census_full/census_income_test.csv')
    # allData = pd.DataFrame.merge(train, test)
    # allData = concatDf(train, test)
    return train, test

def main():
    # TODO check aggregation of features, first that comes to mind is CapGains and CapLosses
    train, test = importData()
    describe(train)
    train = dropUnneeded(train)
    train = replaceQMarks(train)
    convertNominalFeatures(train)
    # correlationPlot(train)

    # this is but a test
    y_data = train['Target']
    x_data = train.drop(['Target'], axis=1)

    reg = LogisticRegression(max_iter=100)
    reg.fit(x_data, y_data)

    test = dropUnneeded(test)
    test = replaceQMarks(test)
    convertNominalFeatures(test)

    y_test = test['Target']
    x_test = test.drop(['Target'], axis=1)
    preds = reg.predict(x_test)

    index = 0
    score = 0
    for val in preds:
        if val == y_test.loc[index]:
            score += 1
        index += 1

    print('Accuracy Logistic Reg: ', (score / len(y_test)) * 100)

    # next try with decision tree
    d_t = DecisionTreeClassifier()
    d_t.fit(x_data, y_data)
    preds = d_t.predict(x_test)

    index = 0
    score = 0
    for val in preds:
        if val == y_test.loc[index]:
            score += 1
        index += 1

    print('Accuracy DT Class: ', (score / len(y_test)) * 100)

    # final impl with random forests
    rf = RandomForestClassifier()
    rf.fit(x_data, y_data)
    preds = rf.predict(x_test)

    index = 0
    score = 0
    for val in preds:
        if val == y_test.loc[index]:
            score += 1
        index += 1

    print('Accuracy Rand For: ', (score / len(y_test)) * 100)

#     post prediction - work on balancing train to have a more equal distribution of -50000/+50000

if __name__ == '__main__':
    main()