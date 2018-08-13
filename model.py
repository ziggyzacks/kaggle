import xgboost
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from utils import Logging
import warnings

warnings.simplefilter("ignore")
log = Logging(__name__)

##TODO: add randomized search (as a new function)

class Model(object):
    DEFAULT_ALGO='RF'
    DEFAULT_INPUTS= open('inputs.txt').read().splitlines()
    DEFAULT_PARAMS={'max_depth':20, 'max_features':5, 'n_estimators':100, 'n_jobs':-1}
    DEFAULT_KMEANS = 'clusters.pkl'
    DEFAULT_DAY_DENSITY = 'day_density.csv'
    DEFAULT_DATA_NAME = 'process.csv'
    RANDOM_PARAMS = {
        "max_depth": [3, 5, 10, 15, 20,25],
        "n_estimators": [80, 100, 200, 300]
    }
    SEARCHITERS = 10

    def __init__(self,re_process=False,algo=DEFAULT_ALGO,inputs=DEFAULT_INPUTS,params=DEFAULT_PARAMS,nrows=10000,clustering_model=DEFAULT_KMEANS,redo_clustering=False,
             n_clusters=10,redo_day_density=False,day_density=DEFAULT_DAY_DENSITY,save_data=False,data_name=DEFAULT_DATA_NAME,random_params=RANDOM_PARAMS,searchiters=SEARCHITERS):
        self.algo=algo
        self.re_process=re_process
        self.inputs=inputs
        self.params=params
        self.redo_clustering = redo_clustering
        self.n_rows = nrows
        self.n_clusters = n_clusters
        self.save_data = save_data
        self.clustering_model = clustering_model
        self.redo_day_density = redo_day_density
        self.day_density = day_density
        self.data_name = data_name
        self.searchiters=searchiters
        self.random_params=random_params

    def load_training_data(self):
        log.info('Fetching training data')
        if self.re_process:
            p = Process(redo_clustering=self.redo_clustering, n_rows=self.nrows, n_clusters=self.n_clusters,
                        save_data=self.save_data, clustering_model=self.clustering_model,
                        redo_day_density=self.redo_day_density,
                        day_density=self.day_density, name=self.name)
            data = p.transform()
        else:
            data = pd.read_csv(self.data_name, nrows=self.n_rows)

        return data

    def randomized_search(self):
        data=self.load_training_data()
        param_dist = self.random_params
        if self.algo=='RF':
            model=RandomForestRegressor(n_jobs=-1)
        if self.algo=='XGB':
            model=xgboost.XGBRegressor(n_jobs=-1)
        log.info('Reshaping data into arrays')

        y = data['fare_amount'].dropna(axis=0, how='any').as_matrix()
        X = (data[self.inputs]
             ._get_numeric_data()
             .dropna(axis=0, how='any')
             .as_matrix())

        random_search = RandomizedSearchCV(model,n_jobs=-1,
                                           param_distributions=param_dist,
                                           n_iter=self.searchiters,
                                           scoring='mean_squared_error',verbose=100)
        log.info('Searching parameter space using RMSE as scoring metric for model {}'.format(self.algo))
        # Doing the randomized search
        random_search.fit(X, y)
        return random_search.best_params_

    def fit(self):
        data = self.load_training_data()
        log.info('Reshaping data into arrays')
        y = data['fare_amount'].dropna(axis=0, how='any').as_matrix()
        X = (data[self.inputs]
             ._get_numeric_data()
             .dropna(axis=0, how='any')
             .as_matrix())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
        if self.algo=='RF':
            model=RandomForestRegressor(**self.params)
        if self.algo=='XGB':
            model=xgboost.XGBRegressor(**self.params)
        log.info('Fitting the model')
        model.fit(X_train,y_train)
        rmse=np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        log.info('RMSE on test set is '+str(rmse))
        log.info('Saving the model as '+self.algo+'.pkl')
        joblib.dump(model, self.algo+'.pkl')

    def predict(self):
        log.info('Fetching test data')
        data=pd.read_csv('test.csv')
        log.info('Transforming test data')
        data_1=Process(clustering_model=self.clustering_model,day_density=self.day_density).transform()
        missing_inputs = list(set(self.inputs) - set(list((data_1.columns))))
        for i in missing_inputs:
            data_1[i] = 0
        log.info('Reshaping data into arrays')
        X = (data_1[self.inputs]
             ._get_numeric_data()
             .dropna(axis=0, how='any')
             .as_matrix())
        log.info('Fetching trained model')
        model=joblib.load(self.algo+'.pkl')
        log.info('Computing predictions')
        predicted = model.predict(X)
        df_test['preds'] = predicted
        df_test_keys = df_test[['key', 'preds']]
        log.info('Saving new predictions to a csv')
        df_final = pd.read_csv('sample_submission.csv')
        df_final.fare_amount = df_final.merge(df_test_keys, on='key')['preds']
        df_final[['key', 'fare_amount']].to_csv('results_nathan.csv', index=False)



