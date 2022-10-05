from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import joblib

#Data Preprocessing
housing = pd.read_csv() #파일 주소 지정
housing['income_cat'] = pd.cut(housing['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train = housing.loc[train_index]
    strat_test = housing.loc[test_index]

for data in (strat_train, strat_test):
    data.drop('income_cat', axis=1, inplace=True)

housing = strat_train.drop('median_house_value', axis=1)
housing_labels = strat_train['median_house_value']
housing_num = housing.drop('ocean_proximity', axis=1)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
    ])

num_att = list(housing_num)
cat_att = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', pipeline, num_att),
    ('cat', OneHotEncoder(), cat_att)
    ])

housing_prepared = full_pipeline.fit_transform(housing)

#Sample Data
some_data = housing.iloc[:]
some_labels = housing_labels.iloc[:]
some_data_prepared = full_pipeline.transform(some_data)

#Linear Regression -> looks Underfitting
"""
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

lin_predict = lin_reg.predict(some_data_prepared)
lin_mse = mean_squared_error(lin_predict, some_labels)
lin_rmse = np.sqrt(lin_mse)

#print('Predict:', lin_predict)
#print('Labels:', list(some_labels))
print('lin_mse:', lin_mse, 'lin_rmse:', lin_rmse)
"""

#Decision Tree Regressor -> looks Overfitting
"""
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

scores = cross_val_score(tree_reg, some_data_prepared, some_labels, cv=10)

tree_predict = tree_reg.predict(some_data_prepared)
tree_mse = mean_squared_error(tree_predict, some_labels)
tree_rmse = np.sqrt(tree_mse)

#print('Predict:', tree_predict)
#print('Labels:', list(some_labels))
print('10-KFold Acc: {0:g}%'.format(100*scores.mean()))
"""

#Random Forest Regressor
forest_reg = RandomForestRegressor()
forest_reg.fit(some_data_prepared, some_labels)

scores = cross_val_score(forest_reg, some_data_prepared, some_labels, cv=2)

forest_predict = forest_reg.predict(some_data_prepared)
forest_mse = mean_squared_error(some_labels, forest_predict)
forest_rmse = np.sqrt(forest_mse)

print('10-KFold Acc: {0:g}%'.format(100*scores.mean()))
