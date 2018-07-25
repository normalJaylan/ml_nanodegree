import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import fbeta_score, accuracy_score, make_scorer
from sklearn import linear_model


def read_data():
	data = pd.read_csv('census.csv')

	income_raw = data['income']
	features_raw = data.drop('income', axis=1)

	return data, features_raw, income_raw

def deal_data(features_raw, data):
	skewed = ['capital-gain', 'capital-loss']
	features_raw[skewed] = data[skewed].apply(lambda x:np.log(x+1))

	scaler = MinMaxScaler()
	numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
	features_raw[numerical] = scaler.fit_transform(data[numerical])

	return features_raw

def one_hot(features_raw, income_raw):
	features = pd.get_dummies(features_raw)
	income = income_raw.replace(['>50K', '<=50K'], [1, 0])

	return features, income

def do_split(features, income):
	x_train, x_test, y_train, y_test = train_test_split(features, income, test_size=0.2, random_state=0)
	
	return x_train, x_test, y_train, y_test

def create_model():
	model = linear_model.LogisticRegression(random_state=0)

	return model

def get_result(model, x_train, x_test, y_train, y_test):
	predictions_test = model.predict(x_test)
	predictions_train = model.predict(x_train)

	result = {}
	result['train_acc'] = accuracy_score(y_train, predictions_train)
	result['test_acc'] = accuracy_score(y_test, predictions_test)
	result['train_f'] = fbeta_score(y_train, predictions_train, beta=0.5)
	result['test_f'] = fbeta_score(y_test, predictions_test, beta=0.5)

	return result

def main():
	data, features_raw, income_raw = read_data()
	features_raw = deal_data(features_raw, data)
	features, income = one_hot(features_raw, income_raw)
	x_train, x_test, y_train, y_test = do_split(features, income)

	model = create_model()

	parameters = {'C':[0.1, 1, 10]}
	scorer = make_scorer(fbeta_score, beta=0.5)
	model = GridSearchCV(model, parameters, scorer, cv=10)

	model.fit(x_train, y_train)
	model = model.best_estimator_


	result = get_result(model, x_train, x_test, y_train, y_test)

	print(result)


main()
	




