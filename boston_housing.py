import numpy as np
import pandas as pd
import visuals as vs

from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV

data = pd.read_csv('housing.csv')
prices = data['MDEV']
features = data.drop('MDEV', axis=1)

# Statistics

minimum_price = np.amin(data.values)
maximum_price = np.amax(data.values)
mean_price = np.mean(data.values)
std_price = np.std(data.values)
median_price = np.median(data.values)

def performance_metrics(y_true, y_predict):
	score = r2_score(y_true, y_predict)
	return score

X_train, X_test, y_train, y_test = train_test_split(features, prices, random_state=40, test_size=0.2)

def fit_model(X,y):
	cv_sets = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.20, random_state=0)
	regressor = DecisionTreeRegressor()
	params = {'max_depth': range(1,11)}
	scoring_func = make_scorer(performance_metrics)
	grid = GridSearchCV(regressor, params, cv=cv_sets, scoring=scoring_func)
	grid.fit(X,y)
	return grid.best_estimator_

reg = fit_model(X_train, y_train)

print "Best value for max_depth is {}".format(reg.get_params()['max_depth'])

client_data = [[5,17,15],[4,32,22],[8,3,12]]

for i,price in enumerate(reg.predict(client_data)):
	print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)
