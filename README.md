# Dataiku_Problem

The problem at hand was to model a predictor that could accurately state whether a US citizen made 
more than 50,000 dollars in terms of annual wage. The data was presented in a stratified sampling way, 
which had to be taken into account when graphing and analysing.

## Data Ingestion and Initial Thoughts

The first stage in the process was to import the data via pandas in-built csv reader as follows.

```python
train = pd.read_csv('./us_census_full/census_income_learn.csv')
test = pd.read_csv('./us_census_full/census_income_test.csv')
```

The data was found to contain no headers, so a quick use of `df.head()` and some playing around in google sheets 
led to them being created and manually added both the train and test files.

After this, an initial describing of the data was done to see the amount of missing values together with
the class split. The describe `df.describe()` produced the following results. On further inspection the data was 
found to contain many instances of `not in universe` or `?` which were then understood as being the missing values.

| FeatureName          | Initial Number Missing | Actual Missing % |
|----------------------|------------------------|----------------|
| EducationLastWk      | 0                      | 93.694962      |
| FamMembersU18        | 0                      | 72.288408      |
| FatherBirthCountry   | 0                      | 3.364524       |
| LabourUnion          | 0                      | 90.445212      |
| MigCodeMSA           | 0                      | 50.726984      |
| MigCodeRegDiff       | 0                      | 50.726984      |
| MigCodeRegSame       | 0                      | 50.726984      |
| MigResSunbelt        | 0                      | 92.094646      |
| MotherBirthCountry   | 0                      | 3.066814       |
| OccupationCodeString | 0                      | 50.462353      |
| PrevReg              | 0                      | 92.094646      |
| PrevState            | 0                      | 92.449492      |
| SelfBirthCountry     | 0                      | 1.700556       |
| UnempReas            | 0                      | 96.957744      |
| VeteranAdmQ          | 0                      | 99.005628      |
| WorkerClas           | 0                      | 50.242328      |

The amount of missing data seen in certain columns will be revisited later on in this README.

Next the class split was graphed as seen in the following image.

![image](./plots/train_class_split.png)

This shows a large imbalance between the -50,000 and +50,000 classes, where 93.8% of the training examples are
of one class (-50,000). An attempt to rectify this will be seen later using 
downsampling and upsampling of the training data.

### Analyses and Correlations

Initially, a correlation plot was created for the data through the seaborn library and pandas in built `df.corr()` 
function.

```python
corr = data.corr()
hm = sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=True, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.show()
```

Which produced the following result:

![image](./plots/all_correlations.png)

After this, the same type of graph was plotted but this time taking into consideration the top 15 variable 
correlations with respect to the target variable. 

```python
k=15
cols = corr.nsmallest(k, 'Target')['Target'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.show()
```

![image](./plots/target_correlations.png)

This gives a better idea as to what kind of relationships each individual feature has with the target variable, ideal
for us to create models that can take advantage of these correlations.

Next the continuous variables were taken into account, these being:

```
Age, Wage, CapGains, CapLosses, StockDiv, NumWorkersEmployer, WeeksWorked
```

These were placed in a seaborn pairplot as follows to check for any relationships between them.

```python
cols = ['Age', 'Wage', 'CapGains', 'CapLosses', 'StockDiv', 'NumWorkersEmployer', 'WeeksWorked']
sns.set()
sns.pairplot(data[cols], size=2.5)
plt.show()
```

![image](./plots/continuousVars.png)

### Other Analyses

Other analyses were also carried out, looking at variables with respect to the target class, to again try and 
see any relationships.

```python
# Example histogram code
sns.distplot(data['Age'], hist_kws={'weights':data['InstanceWeight']}, fit=stats.norm)
plt.figure()
# A facet grid created with a wrapper written around pyplots histogram function so as to accept InstanceWeight 
# due to stratified sampling
grid = sns.FacetGrid(data, col='Target', row='Sex', aspect=1.6)
grid.map(weighted_hist, 'Age', 'InstanceWeight', bins=np.arange(100)-0.5)
plt.show()

# the wrapper function weighted_hist
def weighted_hist(x, weights, **kwargs):
    plt.hist(x, weights=weights, **kwargs)
    plt.xticks(rotation=90)
```

The age was first graphed with respect to the target.

![image](./plots/age_wrt_target.png)

This was then re-graphed to take into consideration sex.

![image](./plots/age_wrt_target_sex.png)

Interestingly, a large spike is seen at the extreme end of the age (~90), maybe all entries with ages larger 
than this were bunched to 90? 

The weeks worked in a year was then graphed, again with respect to the target variable.

![image](./plots/weeksworked_wrt_target.png)

As was expected for the 50,000+ class, as the number of weeks worked in the year increased, so did
the proportion of people earning more than 50,000 dollars.

The wage per hour feature was graphed as a distribution plot against a normal distribution, showing a large amount 
of entries as being 0. 

![image](./plots/wage_dist_plot.png)

This entries were then removed to see the non-0 wage distribution, with the same being done here as for in previous 
features.

![image](./plots/non_zero_wage_wrt_target.png)

Surprisingly, some people earning more than 2,000 dollars per hour don't seem to make more than 50,000 dollars per
year. Could this be because they work less than 25 hours in the whole year?

Marital status was then taken into consideration.

![image](./plots/maritalStat_wrt_target.png)

It seems as though married people with a present spouse make up the bulk of people that earn more than 50,000 
dollars per year. 

The occupation of each citizen was also taken into consideration.

![image](./plots/occupationcodestring_wrt_target.png)

The top 4 industries, `Executive admin and managerial, Sales, Precision production craft & repair and 
Professional speciality` were then paired with the industry in question to produce the following plot.

![image](./plots/industrycodestring_wrt_target_occupationcodestring.png) 

Finally, the capital gains feature was graphed in a seaborn distribution plot.

![image](./plots/capgains_distplot.png)

Interestingly, although the -50,000 class has the majority of people having 0 or close to 0 capital gains, 
there are some people that an abnormally high capital gains (~100,000) but still are not classified as making over 
50,000 dollars per year. The other class contains a higher distribution of people having a larger capital gains which 
to be expected.

Further analysis will be revisited later on in this README.

## Cleaning and Setup

Initially three columns were dropped, namely `IndustryCodeString, OccupationCodeString and InstanceWeight` the latter
of which was not needed to model. The other two contained information directly related to `IndustryCode and OccupationCode` 
and therefore could be discarded.

After this, the columns mentioned previously that consisted/contained `?` and `not in universe` were dealt with. To 
start off with, columns where more than 90% of values were missing, together with having a low correlation score, 
were discarded. When these values were seen to occur in less than 90% of the dataframe, the values were replaced 
with the next most popular result.

An example of this would be `SelfBirthCountry` where ~100,000 of the observed values were `North America` (>50%) so where
`?` was seen it was replaced with `North America`. In other cases, such as `MigCodeMSA`, where more than 90% of values 
were found to be `?`, the column was dropped entirely.

The code to accomplish this is as follows:

```python
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
``` 

With data[0] being the train data and data[1] being that used for testing.

Finally, the features were converted into numerical values for our models to be able to comprehend them. This was done 
using the sklearn LabelEncoder(), with the already numerical features being ignored.

```python
def convertNominalFeatures(data):
    for datum in data:
        for col in datum.columns.values:
            if datum[col].dtype.name == 'object':
                le = LabelEncoder()
                datum[col] = le.fit_transform(datum[col])
    return data[0], data[1]
```

## Initial Modelling

Before a better look was taken at the analysis, an initial run of modelling was performed to get an idea of 
a baseline with which future models were to be compared to.

The models taken into consideration were a Logistic Regression, Decision Trees and a Random Forest. Decision Trees and 
Random Forest were both used just to see the difference ensembling makes in the final prediction accuracy. These models 
were implemented as follows:

```python
# first implementation of Logistic Regression
reg = LogisticRegression(solver='lbfgs')
reg.fit(x_data, y_data)
preds = reg.predict(x_test)
print('Accuracy Logistic Reg: ', (accuracy_score(preds, y_test)) * 100)

# first implementation of Decision Trees
d_t = DecisionTreeClassifier()
# cross val
d_t.fit(x_data, y_data)
preds = d_t.predict(x_test)
print('Accuracy DT Class: ', (accuracy_score(preds, y_test)) * 100)

# first implementation of Random Forest
rf = RandomForestClassifier()
rf.fit(x_data, y_data)
preds = rf.predict(x_test)
print('Accuracy Rand For: ', (accuracy_score(preds, y_test)) * 100)
```

Which produced the following accuracies for the test set.

| Logistic Regression | Decision Tree | Random Forest |
|---------------------|---------------|---------------|
| 94.376              | 93.017        | 95.513        |

Great initial results!

### Next Steps

#### Replacing not Removing

The first question that came to mind was the decision to drop the columns having `?` for more than 90% of the data (
ex: `MigCodeMSA`). The same initial models were kept, this time instead of dropping the columns the next most frequently 
seen value was used as a replacement much in the same way the other column's values were replaced.

```python
datum[col] = datum[col].replace(' ?', datum[col].value_counts().index.to_list()[1])
```

This time picking the second value as the first is the 'null' value (`?`), resulting in accuracies of:

| Logistic Regression | Decision Tree | Random Forest |
|---------------------|---------------|---------------|
| 94.284              | 93.091        | 95.507        |

Slight decreases in the tree models, not the desired results.

#### Keeping Features but Reducing Dimensionality

A method called multiple correspondence analysis was tried so as to see if the previously removed features would yield 
any information that could be benefitial to our models. This was done using a python package called `prince`. The reduction 
was done on the migration features, namely: `MigCodeMSA, MigCodeRegDiff, MigCodeRegSame and MigResSunbelt`. 

```python
def doMCA(data):
    QCols = ['MigCodeMSA', 'MigCodeRegDiff', 'MigCodeRegSame', 'MigResSunbelt']
    for datum in data:
        mca = prince.MCA()
        mca = mca.fit_transform(datum[QCols])
        datum['MCA1'] = mca[0]
        datum['MCA2'] = mca[1]
        datum.drop(QCols, axis=1, inplace=True)

    return data[0], data[1]
```

Producing the following results.

| Logistic Regression | Decision Tree | Random Forest |
|---------------------|---------------|---------------|
| 94.376              | 92.9          | 95.49        |


#### Engineering a New Feature - TotalGains

The capital gains was used together with capital losses and stock dividends to create a new feature called `TotalGains`. 
This was done by adding the `Stock Dividends` to the `Capital Gains` and taking away the `Capital Losses`.

After this was done the new feature was graphed with respect to the target variable.

![image](./plots/captotal_wrt_target_bins.png)

This shows how most of the people to make over 50,000 dollars in a year have a total gain in between 10,000-50,000 dollars.
This new feature was used in place of the `CapGains`, `CapLosses` and `StockDiv` features, producing the following results.

| Logistic Regression | Decision Tree | Random Forest |
|---------------------|---------------|---------------|
| 94.297              | 92.907        | 95.335        |

Still a slight decrease, not what was wanted.

#### Engineering a New Feature - TotalWage

Since the wage per hour metric wasn't really seen to be an adequate descriptor of if a person makes more than 50,000 dollars 
per year (see Analysis part), it was used together with the number of weeks worked to create a new feature called `WageYr`. 
This was calculated by multiplying the `Wage` with the `WeeksWorked` and 40, the average number of hours spent working by 
a full time adult. The columns used to make up this new feature were dropped.

```python
datum['WageYr'] = datum['Wage'] * datum['WeeksWorked'] * 40
datum.drop(['Wage', 'WeeksWorked'], axis=1, inplace=True)
```

This was used together with the previously created `TotalGains` feature and yielded the following result.

| Logistic Regression | Decision Tree | Random Forest |
|---------------------|---------------|---------------|
| 94.13               | 92.988        | 95.357        |

Again, a slight decreased when compared to the base model results.

#### UpSampling/DownSampling

The next step was to use sklearn's `resample()` function to better balance the dataset for training. This was done as 
follows:

```python
def downSampleMajorityClass(data):
    minorityDf = data[data['Target'] == 1]
    majorityDf = data[data['Target'] == 0]

    majorityDfDownsample = resample(majorityDf, replace=False, n_samples=len(minorityDf), random_state=1)

    return pd.concat([majorityDfDownsample, minorityDf])


def upSampleMinorityClass(data):
    minorityDf = data[data['Target']==1]
    majorityDf = data[data['Target']==0]

    minorityDfUpsampled = resample(minorityDf, replace=True, n_samples=len(majorityDf), random_state=1)

    return pd.concat([minorityDfUpsampled, majorityDf])
```

Which were both tried separately and together on the original data.

##### UpSample Minority Class to Majority Class

First the minority class (>50,000) were upsampled to be of the same length as the majority class (<50,000) producing the 
following results.

(Original data)

| Logistic Regression | Decision Tree | Random Forest |
|---------------------|---------------|---------------|
| 87.245               | 93.218        | 95.141        |

(With engineered features)

| Logistic Regression | Decision Tree | Random Forest |
|---------------------|---------------|---------------|
| 87.426              | 93.142         | 95.07        |

It seems that adding so many points to the under represented class isn't such a good idea, especially for logistic 
regression.

##### DownSample Majority Class to Minority Class

The same was done but this time reducing the majority class to be of the same length as that of the minority class (around 
12,000 samples each).

(Original data)

| Logistic Regression | Decision Tree | Random Forest |
|---------------------|---------------|---------------|
| 85.63               | 81.51         | 85.84         |

(With engineered features)

| Logistic Regression | Decision Tree | Random Forest |
|---------------------|---------------|---------------|
| 84.31               | 81.328        | 85.823        |

This is even worse, as we are reducing the amount of data that the models have to learn.

#### Using both

Finally a combination of both was used, upsampling the minority class while downsampling the majority class so that 
they the lengths are the same and in between the original lengths.

(With engineered features)

| Logistic Regression | Decision Tree | Random Forest |
|---------------------|---------------|---------------|
| 86.45               | 91.416        | 94.19         |

More information therefore higher scores, but at the same time we are still removing the 'real' information and adding 
fakes by upsampling. Leading to a worse overall accuracy.

#### Further Removing >90% Null Features

Previously it was mentioned how certain columns where more than 90% of data was null were removed. This was now extended 
so that all columns were taken into consideration. If a column had more than 90% of its values as null, it was discarded.

```python
def checkRemoveMissingData(data):
    testData = data.copy()
    testData[0] = testData[0].replace(' ?', np.nan)
    testData[0] = testData[0].replace(' Not in universe', np.nan)

    # from this largest lack of testData[0] is in EducationLastWk, LabourUnion, UnempReason, PrevReg, PrevState, MigResSunbelt, VeteranAdmQ > 90% are nulls
    nullSeries = testData[0].isnull().sum()/len(testData[0]) * 100
    for item in nullSeries.iteritems():
        if item[1] >= 90:
            data[0].drop([item[0]], axis=1, inplace=True)
            data[1].drop([item[0]], axis=1, inplace=True)
    return data[0], data[1]
```

Resulting in:

| Logistic Regression | Decision Tree | Random Forest |
|---------------------|---------------|---------------|
| 94.375              | 93.12         | 95.528        |

Our first slight gain in accuracy! Now we know that no information is better than `?`. 

#### Introducing KFolds 

Here KFolds cross validation was introduced to the Logistic Regression model to see its impact on accuracy. This was 
done using sklearn's `KFolds()` and `cross_val_score()` methods.

```python
kf = KFold(n_splits=10, shuffle=True, random_state=123)
reg = LogisticRegression(solver='lbfgs', max_iter=500)
cross_val_score(reg, x_data, y_data, cv=kf)
preds = cross_val_predict(reg, x_test, y_test, cv=kf)
print('Accuracy Logistic Reg: ', (accuracy_score(preds, y_test)) * 100)
```

| Logistic Regression |
|---------------------|
| 94.59               |

The best accuracy seen so far in Logistic Regression!

#### Hyperparameter Tuning

Since the random forest model was performing the best until now, it was decided that its parameters should be tuned 
programmatically. This involved using sklearn's `RandomizedSearchCV` and `RandomForestRegressor` to iteratively go 
over different hyperparameter options and test them at each stage, keeping the one that produced the best score.

```python
def hyperParamTuningRF(x_data, y_data, x_test, y_test):
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
    rfRand.fit(x_data, y_data)
    print(rfRand.best_params_)
```

Which produced the following result.

| Random Forest       |
|---------------------|
| 95.595              |

Another slight improvement!

After this experiment the random forest model was kept with these hyperparameters.

#### Going back to Feature Selection

Another, more statistics based approach was taken to selecting features. Three different feature selection methods 
were used to obtain a list of the most important features. These methods were recursive feature elimination, chi square 
test and mutual information metric. 

Chi-Square Graph Scores

![image](./plots/chi_test_features.png)

Mutual Information Metric Scores

![image](./plots/mutual_info_test_features.png)

These, together with recursive feature elimination available through sklearn's `RFE` method were bunched together to 
produce a dataframe containing the 15 most important features. The results for this data are as follows:

| Logistic Regression | Decision Tree | Random Forest |
|---------------------|---------------|---------------|
| 94.13               | 94.224        | 95.212        |

Less information is worse!

### Introducing New Models

Two new models were added, namely Microsoft's LightGBM and XGBoost Gradient Boosting Models. Due to both of these models 
being tree based, some of the hyperparameters such as `n_estimators` and `max_depth` were carried over from the random 
forest model mentioned previously. These were done using the python `lightgbm` and `xgboost` packages.

#### LightGBM Implementation

```python
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
```

#### XGBoost Implementation
```python
def xgBoost(x_data, y_data, x_test, y_test):
    # using model with tuned hyperparams (tuned for random forest classifier)
    # produces better results ~0.5% better but takes substantially longer to compute
    # model = XGBClassifier(n_estimators=800, max_depth=28)

    # using base model
    model = XGBClassifier()
    model.fit(x_data, y_data)
    y_pred = model.predict(x_test)
    print('Accuracy XGBoost: ', (accuracy_score(y_pred, y_test)) * 100)
```

Which produced the following results:

| LightGBM | XGBoost |
|---------------------|---------------|
| 95.714               |    95.653     |

The best results seen so far!

#### Stacked Regression

Another model was tried which included ensembling different regression techniques and stacking them, such that a first 
layer of regressors feeds their results to another regressor that uses the previous outputs as features. This can be 
better understood by looking at the following graphic.

![image](./plots/stacked.jpg)

The implementation for this model can be seen in the `AveragingModel.py` file, which was carried over from a previous 
project. The stacked model contained an Random Forest, Lasso Regressor, Gradient Boosting Regressor and normal Logistic
Regressor

| Stacked Model |
|---------------------|
| 94.736               |

This needs to be investigated and tuned further.

### Another Final Look at the Statistics

A final metric was looked at to again rank features and their importance, this being the SHAP ranking. This ranking (
SHaplet Additive exPlanations) measures the contributions of the predictors via Shapley values which are the average of 
the marginal contributions across all permutations.

Initially all features were ranking against one another.

![image](./plots/shap_summary.png)

This graph has a lot of information, ranking features from most important at the top, together with the value of each feature 
within its distribution. For Age, for example, we can see how a large number of points are bunched up towards the right.

Expanding on the Age feature and looking at the SHAP values for Age specifically, we can plot the distribution of each 
data point's SHAP value. This was done for both the Age and Weeks Worked features.

![image](./plots/shap_age.png)

![image](./plots/shap_weeks_worked.png)

These show quite evident relationships to the SHAP values for the respective feature. Using this information, both the 
Age feature and Weeks Worked were binned (based on their positive SHAP values) and the models were re-run. A new python 
file was created for this (Dataiku_Second.py).

The values were binned as follows:

```python
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
```

These produced the following results:

| Logistic Regression | Decision Tree | Random Forest |
|---------------------|---------------|---------------|
| 94.614               | 93.27       | 95.528       |

Slight improvements, but improvements nonetheless. 

## Further Work

The results shown here are quite positive, but at the same time they can be improved on. Following are some further steps 
that can be taken to improve the project overall:

* A greater effort at feature engineering, creating new features and possibly also reducing features to carry more 
information.
* A greater understanding of the underlying statistics used to define feature importance, which also contributes to 
the previous point.
* Looking at other models to solve the problem (neural networks, SVM)
* Taking greater care to plan solving the problem from before hand. I.E. Revisiting feature selection repeatedly throughout 
the project does not help.

### Edit - More Further Work

* Use pred to fill in missing values, some form of regression?
* Tune xgboost and lightgbm, was not done due to the time it would take - Random Forest hyperparameter tuning took 
around 7 hours.
* Log transform of continuous variables to reduce skew and make more normally distributed. - done
* One hot some categorical vars to test.
* Pivot to create new feature, wage bin as column name then number of weeks worked in column 
* Adaptive binning via df.quantile() to get corresponding quant list 