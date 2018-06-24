# Udacity MLND Capstone Project and Report
## Purpose
This project’s objective is to define directionality of the closing price of a stock/ETF N days in the future relative to its daily opening price i.e. whether its price is predicted to increase (up), or decrease (down). Given the binary nature of the outcome, supervised learning techniques will be used to achieve the objective; specifically, classification techniques such as Decision Tress, Gaussian Naïve Bayes, Support Vector Machine and Neural Networks (in order of complexity) will be used. It used sensitivity analysis to define best parameter set for maximizing prediccion accuracy (AUC_ROC)


## Data
This project use the following datasets for ETF and Indexes:
* “Daily and Intraday Stock Price Data” from Kaggle.com(https://www.kaggle.com/borismarjanovic/daily-and-intraday-stock-price-data). It contains full historical daily (an intraday) price and volume data for all U.S.-based stocks and ETFs trading on the NYSE, NASDAQ, and NYSE MKT (Date, Open, High, Low, Close, Volume, OpenInt). This data set is adjusted for splits and dividends. Specifically, the daily data will be used.

* NASDAQ-100 Technology Index (NDXT) https://finance.yahoo.com/quote/%5ENDXT/history?p=^NDXT
* Volatility Index (VIX) http://www.cboe.com/products/vix-index-volatility/vix-options-and-futures
* NASDAQ Volatility Index (VXN) http://www.cboe.com/products/vix-index-volatility/volatility-on-stock-indexes/cboe-nasdaq-100-volatility-index-vxn

* List of best performing ETFs (http://etfdb.com/etfdb-category/technology-equities/%23etfs&sort_name=ytd_percent_return&sort_order=desc&page=1)

Engineered Features:
* Up_Down: defines whether the stock increased or decreased its price each day
* ETF’s Price Momentum
* ETF’s Price Volatility
* Sector’s Momentum - NDXT
* Sector’s Volatility - NDXT
* Market’s Momentum - VIX
* Market’s Volatility - VIX
* Market’s Momentum - VXN
* Market’s Volatility - VXN

These datasets will be combined into one dataframe for easy training and testing afterwards. This is, data from VIX, NXDT and VXN data sets will be brought in into one dataframe. The dataset contains 38921 rows for 13 “Tickers” with a data range from 2/23/2016 through 11/10/17.

## Libraries used
It uses the following libraries which required no special installation procedure:
### Core Libraries
import numpy as np 
import pandas as pd 
import random
import datetime as dt
from pandas.core import datetools
import time
from matplotlib import pyplot as pyplot
import csv
import os

### Preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer

### Cross-Validation and Hyper-Parameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

### Evaluation Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc,roc_auc_score

### Classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

### GGPLOT2-like library
from plotnine import *

## Additional Documents
* MLND_Capstone_Stock_LP_2.ipynb: contains code and exploratory data analysis, results and comments. Written in Python (2.7.12) in Jupyter Notebooks
* 20180310 Capstone Report.pdf: project report, containing analyses and following project rubric items
* 20180201 Proposal.pdf: Approved proposal for Capstone Project