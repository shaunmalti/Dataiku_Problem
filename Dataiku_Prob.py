import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import resample

from scipy import stats

import prince

from sklearn.model_selection import StratifiedKFold

pd.set_option('display.max_columns', 50)

def checkMissingData(data):
    data = data.replace(' ?', np.nan)
    data = data.replace(' Not in universe', np.nan)

    # from this largest lack of data is in EducationLastWk, LabourUnion, UnempReason, PrevReg, PrevState, MigResSunbelt, VeteranAdmQ > 90% are nulls
    print(data.isnull().sum()/len(data) * 100)


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


# not used
def scaleFeatures(data):
    numericalFeatures = ['Age', 'Wage', 'CapGains', 'CapLosses', 'StockDiv', 'NumWorkersEmployer', 'WeeksWorked']
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
    for datum in data:
        datum.drop(['IndustryCodeString', 'OccupationCodeString', 'InstanceWeight'], axis=1, inplace=True)

    return data[0], data[1]


def checkTargetSplit(data):
    plt.figure()
    sns.countplot(data['Target'])
    # plt.show()


def weighted_hist(x, weights, **kwargs):
    plt.hist(x, weights=weights, **kwargs)
    plt.xticks(rotation=90)


def weighted_scatter(x, y, **kwargs):
    plt.scatter(x, y, **kwargs)


def getCapStats(data):
    # show distribution of non zero cap gains
    nonZeroCapGains = data.loc[data['CapGains'] > 0]
    nonZeroCapGains = data
    # sns.distplot(nonZeroCapGains['CapGains'], hist_kws={'weights': nonZeroCapGains['InstanceWeight']}, fit=stats.norm)
    # plt.show()
    # stats.probplot(nonZeroCapGains['CapGains'], plot=plt)
    # plt.show()

    # remove capgains larger than 50000, outliers
    nonZeroCapGains = nonZeroCapGains.loc[(nonZeroCapGains['CapGains'] < 50000) & (nonZeroCapGains['CapGains'] > 0)]
    # unskew
    nonZeroCapGains['CapGains'] = np.log(nonZeroCapGains['CapGains'])
    # sns.distplot(nonZeroCapGains['CapGains'], hist_kws={'weights': nonZeroCapGains['InstanceWeight']}, fit=stats.norm)
    # plt.show()
    # stats.probplot(nonZeroCapGains['CapGains'], plot=plt)
    # plt.show()


    nonZeroCapLosses = data.loc[data['CapLosses'] > 0]
    # sns.distplot(nonZeroCapLosses['CapLosses'], hist_kws={'weights': nonZeroCapLosses['InstanceWeight']}, fit=stats.norm)
    # plt.show()
    # stats.probplot(nonZeroCapLosses['CapLosses'], plot=plt)
    # plt.show()

    nonZeroCapLosses['CapLosses'] = np.log(nonZeroCapLosses['CapLosses'])
    # sns.distplot(nonZeroCapLosses['CapLosses'], hist_kws={'weights': nonZeroCapLosses['InstanceWeight']}, fit=stats.norm)
    # plt.show()
    # stats.probplot(nonZeroCapLosses['CapLosses'], plot=plt)
    # plt.show()

    return nonZeroCapGains, nonZeroCapLosses



def gridPlot(data):
    # show effect of occupation wrt target
    # grid = sns.FacetGrid(data, col='Target', aspect=1.6)
    # grid.map(weighted_hist, 'OccupationCodeString', 'InstanceWeight', bins=np.arange(data['OccupationCodeString'].nunique())-0.5)
    # plt.show()
    #
    # show effect of occupation and gender wrt target
    # grid = sns.FacetGrid(data, col='Target', row='Sex', aspect=1.6)
    # grid.map(weighted_hist, 'Age', 'InstanceWeight', bins=np.arange(100)-0.5)
    # plt.show()

    # show industry of highest scoring occupations wrt target
    # specificIndustryData = data.loc[
    #     data['OccupationCodeString'].isin([' Professional specialty', ' Executive admin and managerial', ' Sales', ' Precision production craft & repair'])
    # ]
    # grid = sns.FacetGrid(specificIndustryData, col='Target', row='OccupationCodeString', aspect=1.6)
    # grid.map(weighted_hist, 'IndustryCodeString', 'InstanceWeight', bins=np.arange(data['IndustryCodeString'].nunique())-0.5)
    # plt.show()

    # weeks worked is the highest correlated var wrt target, therefore check gridplot wrt target
    # remove 0 weeks worked since they are the overwhelming majority
    # weeksWorkedGreaterZero = data.loc[data['WeeksWorked'] > 0]
    # grid = sns.FacetGrid(weeksWorkedGreaterZero, col='Target', aspect=1.6)
    # grid.map(weighted_hist, 'WeeksWorked', 'InstanceWeight', bins=np.arange(weeksWorkedGreaterZero['WeeksWorked'].nunique())-0.5)
    # plt.show()

    # check capitalgains wrt target
    # changing capgains to ordinals to better view
    # ranges are based off of df['capitalgains'].describe() and taking into consideration min, 25%, 50%, 75% and max vals
    # capitalGainsDf = data.loc[data['CapGains'] > 0]
    # capitalGainsDf['CapTotal'] = capitalGainsDf['CapGains'] + capitalGainsDf['StockDiv'] - capitalGainsDf['CapLosses']
    # capitalGainsDf.loc[(capitalGainsDf['CapTotal'] > 100) & (capitalGainsDf['CapTotal'] <= 2500), 'newCapGains'] = '100-2500'
    # capitalGainsDf.loc[(capitalGainsDf['CapTotal'] > 2500) & (capitalGainsDf['CapTotal'] <= 5000), 'newCapGains'] = '2500-5000'
    # capitalGainsDf.loc[(capitalGainsDf['CapTotal'] > 5000) & (capitalGainsDf['CapTotal'] <= 10000), 'newCapGains'] = '5000-10000'
    # capitalGainsDf.loc[(capitalGainsDf['CapTotal'] > 10000) & (capitalGainsDf['CapTotal'] <= 50000), 'newCapGains'] = '10000-50000'
    # capitalGainsDf.loc[(capitalGainsDf['CapTotal'] > 50000), 'newCapGains'] = '50000+'
    # grid = sns.FacetGrid(capitalGainsDf, col='Target', aspect=1.6)
    # grid.map(weighted_hist, 'newCapGains', 'InstanceWeight', bins=np.arange(5))
    # plt.show()

    # next is numworkersemployer - highly corred (0.22), it is the number of workers working for the person's employer
    # print(data['NumWorkersEmployer'].value_counts()) - 7 distinct values
    # grid = sns.FacetGrid(data, col='Target', aspect=1.6)
    # # grid.map(weighted_hist, 'NumWorkersEmployer', 'InstanceWeight', bins=np.arange(7)-0.5)
    # grid.map(weighted_hist, 'IndustryCodeString', 'WeeksWorked', bins=np.arange(data['IndustryCodeString'].nunique())-0.5)
    # plt.show()

    # grid = sns.FacetGrid(capitalGainsDf, col='Target', aspect=1.6)
    # grid.map(sns.distplot, 'CapGains')
    # plt.show()

    # grid = sns.FacetGrid(data, col='Target', aspect=1.6)
    # grid.map(weighted_hist, 'MaritalStat', 'InstanceWeight', bins=np.arange(data['MaritalStat'].nunique())-0.5)
    # plt.show()

    # checking pair plots for continuous features
    # cols = ['Age', 'Wage', 'CapGains', 'CapLosses', 'StockDiv', 'NumWorkersEmployer', 'WeeksWorked']
    # sns.set()
    # sns.pairplot(data[cols], size=2.5)
    # plt.savefig('continuousVars')

    # rechecking age for histogram
    # sns.distplot(data['Age'], hist_kws={'weights':data['InstanceWeight']}, fit=stats.norm)
    # plt.figure()
    # stats.probplot(data['Age'], plot=plt)
    # plt.show()

    # capStats = getCapStats(data)
    data['CapTotal'] = data['CapGains'] - data['CapLosses'] + data['StockDiv']
    # TODO try statistics with captotal column

    # return


def describe(data):
    # checkTargetSplit(data)
    return gridPlot(data)


def correlationPlot(data):
    # k = 15  # number of variables for heatmap
    # corrMatrix = data.corr()
    #
    # cols = corrMatrix.nlargest(k, 'Target')['Target'].index
    # cm = np.corrcoef(data[cols].values.T)
    # sns.set(font_scale=1.25)
    # hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
    #                  xticklabels=cols.values)
    # plt.show()

    # tot corrmatrix
    corr = data.corr()
    hm = sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=True, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
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

def dropMissingDataCols(train, test):
    # dropping colums with >90% missing data to see effect on accuracy
    train.drop(
        ['EducationLastWk', 'LabourUnion', 'UnempReas', 'PrevReg', 'PrevState', 'VeteranAdmQ'],
        axis=1, inplace=True)
    test.drop(
        ['EducationLastWk', 'LabourUnion', 'UnempReas', 'PrevReg', 'PrevState', 'VeteranAdmQ'],
        axis=1, inplace=True)
    return train, test


def featureRanking(x_data, y_data):
    estimator = LogisticRegression()
    selector = RFE(estimator, 5, step=1)
    selector = selector.fit(x_data, y_data)
    print(selector.ranking_)
    print(x_data.columns.values)
    print(selector.n_features_)
    # exit()


def main():
    train, test = importData()
    # checkMissingData(train)
    # describe(train)
    # exit()
    train, test = dropUnneeded([train, test])
    train, test = replaceQMarks([train, test])
    train, test = convertNominalFeatures([train, test])
    # train, test = aggregateGains([train, test])
    # correlationPlot(train)
    # exit()
    train, test = dropMissingDataCols(train, test)

    # doing under/oversampling
    # train = downSampleMajorityClass(train, 90000)
    # train = upSampleMinorityClass(train, 90000)

    # this is but a test
    y_data = train['Target']
    x_data = train.drop(['Target'], axis=1)

    # featureRanking(x_data, y_data)

    # scaling no real increase
    # train, test = scaleFeatures([train, test])

    # check area under roc curve to better evaluate model performance
    y_test = test['Target']
    x_test = test.drop(['Target'], axis=1)


    reg = LogisticRegression()
    score = cross_val_score(reg, x_data, y_data, cv=4)
    reg.fit(x_data, y_data)
    preds = reg.predict(x_test)

    print(score.mean(), score.std() * 2)

    print('Accuracy Logistic Reg: ', (accuracy_score(preds, y_test)) * 100)

    pred_prob_train = reg.predict_proba(x_data)
    pred_prob_train = [p[1] for p in pred_prob_train]
    print('ROC Accuracy Log Reg: ', roc_auc_score(y_data, pred_prob_train))

    # next try with decision tree
    d_t = DecisionTreeClassifier()
    # cross val
    d_t.fit(x_data, y_data)
    preds = d_t.predict(x_test)
    print('Accuracy DT Class: ', (accuracy_score(preds, y_test)) * 100)

    pred_prob_train = d_t.predict_proba(x_data)
    pred_prob_train = [p[1] for p in pred_prob_train]
    print('ROC Accuracy DT: ', roc_auc_score(y_data, pred_prob_train))


    # final impl with random forests
    rf = RandomForestClassifier()
    rf.fit(x_data, y_data)
    preds = rf.predict(x_test)
    print('Accuracy Rand For: ', (accuracy_score(preds, y_test)) * 100)

    pred_prob_train = rf.predict_proba(x_data)
    pred_prob_train = [p[1] for p in pred_prob_train]
    print('ROC Accuracy Rand For: ', roc_auc_score(y_data, pred_prob_train))

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

    best_random = rfRand.best_estimator_
    print('****************************************')
    print('EVALUATING HYPERPARAM TUNED RANDOM FOREST')
    print('****************************************')
    random_accuracy = evaluate(best_random, x_test, y_test)
    print('****************************************')
    print('EVALUATING ORIG RANDOM FOREST')
    print('****************************************')
    random_acc_orig = evaluate(rf, x_test, y_test)

    print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - random_acc_orig) / random_acc_orig))



    # SGDclassifier
    sgd = SGDClassifier()
    # added kfolds
    score = cross_val_score(sgd, x_data, y_data, cv=5)
    print(score.mean(), score.std() * 2)
    sgd.fit(x_data, y_data)
    preds = sgd.predict(x_test)

    print('Accuracy SGD: ', (accuracy_score(preds, y_test)) * 100)


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


if __name__ == '__main__':
    main()