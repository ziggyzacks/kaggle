import pandas as pd
import geopy.distance
from datetime import datetime
import numpy as np
import pickle
import joblib
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.cluster import KMeans
import glob
from utils import Logging
import warnings

warnings.simplefilter("ignore")
log = Logging(__name__)


class Process(object):
    DEFAULT_KMEANS='clusters.pkl'
    DEFAULT_DAY_DENSITY='day_density.csv'
    DEFAULT_NAME='processed.csv'


    def __init__(self,nrows=10000,clustering_model=DEFAULT_KMEANS,redo_clustering=False,
             n_clusters=10,redo_day_density=False,day_density=DEFAULT_DAY_DENSITY,save_data=False,name=DEFAULT_NAME):
        self.redo_clustering=redo_clustering
        self.n_rows=nrows
        self.n_clusters=n_clusters
        self.save_data=save_data
        self.clustering_model=clustering_model
        self.redo_day_density=redo_day_density
        self.day_density=day_density
        self.name=name


    def get_data(self):
        log.info('Fetching the data')
        return pd.read_csv('~/Documents/Codes/Kaggle/all/train.csv',nrows=self.n_rows)

    def remove_outliers(self,data):
        log.info('Removing fare amount outliers')
        data = data[data.fare_amount >= 0]
        log.info('Removing location outliers')
        data = data[(data.pickup_longitude < -70) & (data.pickup_longitude >= -75) &
                      (data.dropoff_longitude < -70) & (data.dropoff_longitude >= -75) &
                      (data.pickup_latitude < 42) & (data.pickup_latitude >= 39) &
                      (data.dropoff_latitude < 42) & (data.dropoff_latitude >= 39)]
        return data

    def compute_distance(self,data):
        """
        computes lat, long and straight distances
        :type data: pandas DataFrame
        """

        def dist(lat1, long1, lat2, long2):
            try:
                coords_1 = (long1, lat1)
                coords_2 = (long2, lat2)

                return (geopy.distance.vincenty(coords_1, coords_2).km)
            except:
                return -10

        def lat_dist(lat1, long1, lat2, long2):
            return dist(lat1, long1, lat2, long1)

        def long_dist(lat1, long1, lat2, long2):
            return dist(lat1, long1, lat1, long2)

        log.info('Computing straight distances')
        data['distance'] = data.apply(
            lambda x: dist(x.pickup_latitude, x.pickup_longitude, x.dropoff_latitude, x.dropoff_longitude), axis=1)
        data = data[data.distance >0]
        log.info('Computing lat distances')
        data['latdistance'] = data.apply(
            lambda x: lat_dist(x.pickup_latitude, x.pickup_longitude, x.dropoff_latitude, x.dropoff_longitude), axis=1)
        log.info('Computing long distances')
        data['longdistance'] = data.apply(
            lambda x: long_dist(x.pickup_latitude, x.pickup_longitude, x.dropoff_latitude, x.dropoff_longitude), axis=1)
        data = data[data.distance > 0]
        return data

    def split_dates(self,data):
        # Extract date attributes and then drop the pickup_datetime column
        log.info('Extracting date info')
        data['pickup_datetime'] = data['pickup_datetime'].str.slice(0, 16)
        data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
        data['hour'] = data['pickup_datetime'].dt.hour
        data['day'] = data['pickup_datetime'].dt.day
        data['month'] = data['pickup_datetime'].dt.month
        data['year'] = data['pickup_datetime'].dt.year
        data['weekday'] = data['pickup_datetime'].dt.weekday
        return data

    def make_cluster(self,data):
        log.info('Fitting kmeans with '+str(self.n_clusters)+' clusters')
        # Training the kmeans algo
        X_cluster = np.vstack([data[['pickup_longitude',
                                      'pickup_latitude']].dropna(axis=0, how='any').as_matrix(),
                               data[['dropoff_longitude',
                                      'dropoff_latitude']].dropna(axis=0, how='any').as_matrix()])
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(X_cluster)
        log.info('Saving kmeans as '+self.clustering_model)
        joblib.dump(kmeans, self.clustering_model)
        return kmeans

    def assign_clusters(self,data):
        if self.redo_clustering:
            kmeans=self.make_cluster(data)
        else:
            log.info('Fetching kmeans model')
            kmeans=joblib.load(self.clustering_model)
        log.info('Assigning clusters')
        data['pickup_cluster'] = kmeans.predict(data[['pickup_longitude',
                                                        'pickup_latitude']].dropna(axis=0, how='any').as_matrix())
        data['pickup_cluster'] = data['pickup_cluster'].astype(str)
        data['dropoff_cluster'] = kmeans.predict(data[['dropoff_longitude',
                                                         'dropoff_latitude']].dropna(axis=0, how='any').as_matrix())
        data['dropoff_cluster'] = data['dropoff_cluster'].astype(str)
        return data

    def add_day_density(self,data):
        data['pickup_datetime_2'] = data['pickup_datetime'].apply(lambda x: x.date())
        if self.redo_day_density:
            log.info('Computing day density')
            day_density=data[['pickup_datetime_2']].groupby(data.pickup_datetime_2).count().add_suffix('_Count').reset_index()
            log.info('saving new day density')
            day_density.to_csv(self.day_density)
        else:
            day_density=pd.read_csv(self.day_density)
        log.info('Adding day density')
        day_density['pickup_datetime_2'] = pd.to_datetime(day_density['pickup_datetime_2'])
        data['pickup_datetime_2'] = pd.to_datetime(data['pickup_datetime_2'])
        data = data.merge(day_density, how='left', on='pickup_datetime_2')
        data['day_density'] = data.pickup_datetime_2_Count
        return data

    def add_holidays(self,data):
        log.info('Adding holidays')
        cal = calendar()
        holidays = cal.holidays(start=data['pickup_datetime'].min(), end=data['pickup_datetime'].max())
        hol = []
        for i in holidays:
            hol.append(i.date())
        data['holiday'] = data['pickup_datetime_2'].isin(hol)
        return data

    def transform(self):
        raw_data=self.get_data()
        data=self.remove_outliers(raw_data)
        data=self.compute_distance(data)
        data=self.split_dates(data)
        data=self.assign_clusters(data)
        data=self.add_day_density(data)
        data=self.add_holidays(data)
        log.info('Transform data into dummies (for clusters)')
        data = pd.get_dummies(data[['fare_amount','pickup_longitude',
                                      'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                                      'passenger_count', 'distance', 'year', 'month', 'day', 'hour',
                                      'weekday', 'latdistance', 'longdistance', 'holiday', 'day_density',
                                      'pickup_cluster', 'dropoff_cluster']])
        if self.save_data:
            log.info('Saving the processed data as '+self.name)
            data.to_csv(self.name)
        return data






