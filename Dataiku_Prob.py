import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import resample

import prince

from sklearn.model_selection import StratifiedKFold

pd.set_option('display.max_columns', 50)

def downSampleMajorityClass(data, amount):
    minorityDf = data[data['Target'] == 1]
    majorityDf = data[data['Target'] == 0]

    majorityDfDownsample = resample(majorityDf, replace=False, n_samples=amount, random_state=1)

    return pd.concat([majorityDfDownsample, minorityDf])

def upSampleMinorityClass(data, amount):
    minorityDf = data[data['Target']==1]
    majorityDf = data[data['Target']==0]

    minorityDfUpsampled = resample(minorityDf, replace=True, n_samples=amount, random_state=1)

    return pd.concat([minorityDfUpsampled, majorityDf])

def aggregateGains(data):
    gainCols = ['CapGains', 'CapLosses', 'StockDiv']
    for datum in data:
        datum['GainsOverall'] = datum['CapGains'] + datum['StockDiv'] - datum['CapLosses']
        datum.drop(gainCols, axis=1, inplace=True)
        # datum['WageYr'] = datum['Wage'] * datum['WeeksWorked']
        # datum.drop(['Wage', 'WeeksWorked'], axis=1, inplace=True)
    return data[0], data[1]


def scaleFeatures(data):
    numericalFeatures = ['Age', 'Wage', 'CapGains', 'CapLosses', 'StockDiv', 'NumWorkers', 'WeeksWorked']
    for datum in data:
        for col in numericalFeatures:
            scaler = MinMaxScaler()
            datum[col] = scaler.fit_transform(datum[col].values.reshape(-1,1))
    return data[0], data[1]


def convertNominalFeatures(data):
    for datum in data:
        for col in datum.columns.values:
            if datum[col].dtype.name == 'object':
                le = LabelEncoder()
                datum[col] = le.fit_transform(datum[col])
    return data[0], data[1]

# not used
def doMCA(data):
    QCols = ['MigCodeMSA', 'MigCodeRegDiff', 'MigCodeRegSame', 'MigResSunbelt']
             # 'FatherBirthCountry','MotherBirthCountry', 'SelfBirthCountry', 'PrevState']
    for datum in data:
        mca = prince.MCA()
        mca = mca.fit_transform(datum[QCols])
        datum['MCA1'] = mca[0]
        datum['MCA2'] = mca[1]
        datum.drop(QCols, axis=1, inplace=True)

    return data[0], data[1]


def replaceQMarks(data):
    # TODO check if changing these to mode/median improves accuracy
    QCols = ['MigCodeMSA', 'MigCodeRegDiff', 'MigCodeRegSame', 'MigResSunbelt', 'FatherBirthCountry', 'MotherBirthCountry', 'SelfBirthCountry', 'PrevState']
    for datum in data:
        for col in QCols:
            if datum[col].value_counts().index.to_list()[0] == ' ?':
                datum.drop(col, axis=1, inplace=True)
            else:
                # instead of removing place unknown values as most occurring within column
                datum[col] = datum[col].replace(' ?', datum[col].value_counts().index.to_list()[0])
    return data[0], data[1]

def checkNulls(data):
    print(data.isnull().sum())

def dropUnneeded(data):
    # dont check for numeric as some numeric cols are categorical
    # remove second industry code and occupation code as they are string versions of
    # the first ones seen in the dataset
    # TODO check effect of stratified sampling when doing statistics
    for datum in data:
        datum.drop(['IndustryCodeString', 'OccupationCodeString', 'InstanceWeight'], axis=1, inplace=True)

    return data[0], data[1]

def checkTargetSplit(data):
    plt.figure()
    sns.countplot(data['Target'])
    # plt.show()

def gridPlot(data):
    # gender pay gap first one
    # grid = sns.FacetGrid(data, col='Target', aspect=1.6)
    # grid.map(plt.hist, 'Sex', bins=2)
    # plt.show()

    # gender pay gap wrt industry
    grid = sns.FacetGrid(data, col='Sex', row='Target', aspect=1.6)
    # print(data['OccupationCodeString'].value_counts())
    grid.map(plt.hist, 'OccupationCodeString', bins=data['OccupationCodeString'].nunique())
    # grid.set_xlabels(rotation=45)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    plt.show()

def describe(data):
    # checkTargetSplit(data)
    gridPlot(data)

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

    # cols = corrMatrix.nsmallest(k, 'Target')['Target'].index
    # cm = np.corrcoef(data[cols].values.T)
    # sns.set(font_scale=1.25)
    # hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
    #                  xticklabels=cols.values)
    # plt.show()

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
    # describe(train)

    train, test = dropUnneeded([train, test])
    train, test = replaceQMarks([train, test])
    train, test = convertNominalFeatures([train, test])
    # train, test = aggregateGains([train, test])


    # doing under/oversampling
    # train = downSampleMajorityClass(train, 90000)
    # train = upSampleMinorityClass(train, 90000)

    # print(train.head(10))
    # print(train['Target'].value_counts())
    # exit()

    # correlationPlot(train)
    # print(train.head(10))
    # exit()

    # print(train.head(10))
    # exit()
    #
    # train, test = doMCA([train, test])
    # sns.boxplot(x='Target', y='WeeksWorked', data=train)
    # plt.show()
    # print(train.shape)
    # exit()

    # this is but a test
    y_data = train['Target']
    x_data = train.drop(['Target'], axis=1)



    # test = dropUnneeded(test)
    # test = replaceQMarks(test)
    # convertNominalFeatures(test)

    # TODO see effect of scaling on rf/dt
    # train, test = scaleFeatures([train, test])


    # check area under roc curve to better evaluate model performance
    y_test = test['Target']
    x_test = test.drop(['Target'], axis=1)
    #
    reg = LogisticRegression(solver='saga')
    reg.fit(x_data, y_data)
    preds = reg.predict(x_test)

    print('Accuracy Logistic Reg: ', (accuracy_score(preds, y_test)) * 100)
    pred_prob_train = reg.predict_proba(x_data)
    pred_prob_train = [p[1] for p in pred_prob_train]
    print('ROC Accuracy Log Reg: ', roc_auc_score(y_data, pred_prob_train))

    # # next try with decision tree
    # d_t = DecisionTreeClassifier()
    # d_t.fit(x_data, y_data)
    # preds = d_t.predict(x_test)
    # print('Accuracy DT Class: ', (accuracy_score(preds, y_test)) * 100)
    #
    # pred_prob_train = d_t.predict_proba(x_data)
    # pred_prob_train = [p[1] for p in pred_prob_train]
    # print('ROC Accuracy DT: ', roc_auc_score(y_data, pred_prob_train))
    #
    # # final impl with random forests
    # rf = RandomForestClassifier()
    # rf.fit(x_data, y_data)
    # preds = rf.predict(x_test)
    # print('Accuracy Rand For: ', (accuracy_score(preds, y_test)) * 100)
    #
    # pred_prob_train = rf.predict_proba(x_data)
    # pred_prob_train = [p[1] for p in pred_prob_train]
    # print('ROC Accuracy Rand For: ', roc_auc_score(y_data, pred_prob_train))

    # SGDclassifier
    sgd = SGDClassifier()
    sgd.fit(x_data, y_data)
    preds = sgd.predict(x_test)

    print('Accuracy SGD: ', (accuracy_score(preds, y_test)) * 100)

    pred_prob_train = sgd.predict_proba(x_data)
    pred_prob_train = [p[1] for p in pred_prob_train]
    print('ROC Accuracy SGD: ', roc_auc_score(y_data, pred_prob_train))

if __name__ == '__main__':
    main()