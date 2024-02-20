import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import scipy.stats as stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,RobustScaler
from sklearn.compose import ColumnTransformer
import statsmodels.formula.api as smf 
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.arima.model import ARIMA
import pylab

from sqlalchemy import create_engine

user = 'root'
pw = 'Lucky143'
db = 'medical_db'

engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

medical = pd.read_excel("C:/Users/lucky/OneDrive/Desktop/Optimization of Medical Inventory/Model Building/Medical Inventory Optimaization Dataset.xlsx")
medical.to_sql('medical_data', con = engine,if_exists='replace',chunksize=1000,index=False)

sql = "select * from medical_data"
medical = pd.read_sql_query(sql, engine)

medical.info()

medical.describe()

### EDA business moments
medical_mean = medical.mean()
medical_median = medical.median()
medical_var = medical.var()
medical_std = medical.std()
numerical_columns = medical.select_dtypes(include='number')
medical_range = numerical_columns.max()-numerical_columns.min()
medical_skew = medical.skew()
medical_kurt = medical.kurtosis()


#####AutoEDA---Before Preprocessing ---
#pip install sweetviz
#pip install autoviz
#pip install dtale

import sweetviz as sv
from autoviz.AutoViz_Class import AutoViz_Class 
%matplotlib inline
import dtale

report_sweetviz = sv.analyze(medical)
report_sweetviz.show_html('sweetviz_report.html')

av = AutoViz_Class()
report_autoviz = av.AutoViz('C:/Users/lucky/OneDrive/Desktop/Optimization of Medical Inventory/Model Building/Medical Inventory Optimaization Dataset.xlsx')

d = dtale.show(medical)
d.open_browser()


#### --------------Data Preprocessing ----------
###--Type casting
medical['Patient_ID'] = medical['Patient_ID'].astype(str)
medical['Dateofbill'] = medical['Dateofbill'].astype('datetime64')

##Handling duplicate values

duplicate_value = medical.duplicated()
sum(duplicate_value)
medical.drop_duplicates(inplace = True,keep = 'first')

###Handling missing values

medical.isnull().sum()


group_cols = ['Typeofsales', 'Specialisation', 'Dept']

imputation_columns = ['Formulation', 'DrugName', 'SubCat', 'SubCat1']

for col in imputation_columns:
    medical[col] = medical.groupby(group_cols)[col].apply(
            lambda x: x.fillna(x.mode().iloc[0])if not x.mode().empty else x)

    
medical.dropna(inplace=True)

medical.reset_index(drop = True, inplace = True)

medical.to_csv('C:/Users/lucky/medical_preprocessed_data.csv', index=False)



####Finding and Handling Outliers

medical.plot(kind = 'box', subplots = True, sharey = False, figsize = (15,8))

numerical_columns = ['Quantity','ReturnQuantity','Final_Cost','Final_Sales','RtnMRP']
for i in numerical_columns:
    IQR = medical[i].quantile(0.75) - medical[i].quantile(0.25)
    lower_limit = medical[i].quantile(0.25)-(1.5*IQR)
    upper_limit = medical[i].quantile(0.75) +(1.5*IQR)
    medical[i] = pd.DataFrame(np.where(medical[i] > upper_limit,upper_limit,
                                       np.where(medical[i] < lower_limit,lower_limit,medical[i])))

medical.plot(kind = 'box', subplots = True, sharey = False, figsize = (15,8))


#####----After data Preprocessing-----

medical_mean = medical.mean()
medical_median = medical.median()
medical_var = medical.var()
medical_std = medical.std()
numerical_columns = medical.select_dtypes(include='number')
medical_range = numerical_columns.max()-numerical_columns.min()
medical_skew = medical.skew()
medical_kurt = medical.kurtosis()

#####AutoEDA -----After preprocessing -----

report_sweetviz = sv.analyze(medical)
report_sweetviz.show_html('sweetviz_report1.html')

av = AutoViz_Class()
report_autoviz = av.AutoViz('C:/Users/lucky/medical_preprocessed_data.csv')

# Save the visualizations as HTML
html_path = 'C:/Users/lucky/autoviz_report.html'
report_autoviz.to_html(html_path)

d = dtale.show(medical)
d.open_browser()

#####-----Model Building-----
from statsmodels.tsa.stattools import adfuller
result = adfuller(medical['Quantity'])
test_statistics,p_value,lags_used, num_obs,critical_values,icbest = result
if p_value <=0.05:
    print("The time series is likely stationary")
else:
    print("The time series is likely non-stationary")


medical['Dateofbill'] = pd.to_datetime(medical['Dateofbill'])
medical['billof month'] = medical['Dateofbill'].dt.month_name()
medical.loc[:,'billof month'] = medical['billof month'].str.slice(stop=3)
medical_month = medical.groupby('billof month')['Quantity'].sum().reset_index()

medical['weekofbill'] = medical['Dateofbill'].dt.isocalendar().week
medical.reset_index(drop = True, inplace = True)
medical_week = medical.groupby('weekofbill')['Quantity'].sum().reset_index()


data2 = pd.get_dummies(medical_week['weekofbill'], prefix='weekofbill')


medical_data = pd.concat([medical_week,data2],axis=1) 

medical_data["t"] = np.arange(1,53) # linear trend
medical_data["t-square"] = medical_data["t"] * medical_data["t"]  #Quadratic trend
medical_data["log_Quantity"] = np.log(medical_data["Quantity"]) # exponential trend

medical_data.to_csv(r'C:/Users/lucky/weekly_medical_data.csv', index=True)
medical_data.Quantity.plot()

train = medical_data
test = medical_data

def MAPE(pred,actual):
    temp = np.abs((pred-actual)/actual)*100
    return np.mean(temp)

##----------Linear model--------
linear_model = smf.ols('Quantity ~ t', data = train).fit()
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(test['t'])))
lin = MAPE(np.array(test['Quantity']),np.array(pred_linear))
lin  #  13.555620286044217

#-------Exponential Smoothing-------
exp = smf.ols('log_Quantity ~ t',data=train).fit()
pred_exp = pd.Series(exp.predict(pd.DataFrame(test['t'])))
exp = MAPE(np.array(test['Quantity']),np.array(np.exp(pred_exp)))
exp # 13.757858780817035

#-------Moving average------

mv_pred = medical_data['Quantity'].rolling(5).mean()
MV = MAPE(mv_pred, test.Quantity)
MV # 11.100788158757247

#--------Simple Exponential Smoothing-------
ses_model = SimpleExpSmoothing(train['Quantity']).fit()
pred_ses = ses_model.predict(start = test.index[0],end = test.index[-1])
ses = MAPE(pred_ses,test.Quantity)
ses # 13.852943

#-------Holt's Winter--------
hw_model = Holt(train['Quantity']).fit()
pred_hw = hw_model.predict(start = test.index[0],end = test.index[-1])
hw = MAPE(pred_hw,test.Quantity)
hw # 17.774981323113632

#--------ARIMA----------
model_full = smf.ols('Quantity ~ weekofbill_1 + weekofbill_2 + weekofbill_3 + weekofbill_4 + weekofbill_5 + weekofbill_6 + weekofbill_7 + weekofbill_8 + weekofbill_9 + weekofbill_10 + weekofbill_11 + weekofbill_12 + weekofbill_13 + weekofbill_14+ weekofbill_15 + weekofbill_16 + weekofbill_17 + weekofbill_18 + weekofbill_19 + weekofbill_20 + weekofbill_21 + weekofbill_22 + weekofbill_23 + weekofbill_24 + weekofbill_25+ weekofbill_26 + weekofbill_27 + weekofbill_28 + weekofbill_29 + weekofbill_30+ weekofbill_31 + weekofbill_32 + weekofbill_33 + weekofbill_34 + weekofbill_35 + weekofbill_36 + weekofbill_37 + weekofbill_38 + weekofbill_39+ weekofbill_40 + weekofbill_41 + weekofbill_42 + weekofbill_43 + weekofbill_44 + weekofbill_45 + weekofbill_46+ weekofbill_47 + weekofbill_48+ weekofbill_49 + weekofbill_50 + weekofbill_51 + weekofbill_52+t+t-square',data=train).fit()
predict_data = medical_data
pred_new  = pd.Series(model_full.predict(predict_data))
pred_new
predict_data["forecasted_Quantity"] = pd.Series(pred_new)
model_full.save("model.pickle")

from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model.pickle")

full_res = medical_data.Quantity - model_full.predict(medical_data)

import statsmodels.graphics.tsaplots as tsa_plots

tsa_plots.plot_acf(full_res,lags =12)
tsa_plots.plot_pacf(full_res,lags =5)
arima = ARIMA(train.Quantity, order = (1,0,10))
res1 = arima.fit()
pred_arima = res1.predict(start = test.index[0], end = test.index[-1])
ari = MAPE(pred_arima, test.Quantity[1:])
ari  # 11.448012

#-------------SARIMA-------------
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(train['Quantity'])
plot_pacf(train['Quantity'])
sarima_model = SARIMAX(train['Quantity'], order=(1,0,10), seasonal_order=(0,0,1,52))
sarima_result = sarima_model.fit()
pred_sarima = sarima_result.predict(start = test.index[0], end = test.index[-1])
sari = MAPE(pred_sarima, test.Quantity)
sari # 13.640154

#----------SARIMAX---------

from sklearn import metrics
inventory = pd.read_excel(r"C:\Users\lucky\OneDrive\Desktop\Optimization of Medical Inventory\Model Building\Medical Inventory Optimaization Dataset.xlsx")

inventory.info()


inventory['Patient_ID'] = inventory['Patient_ID'].astype(str)

inventory['Dateofbill'] = inventory['Dateofbill'].astype('datetime64[ns]')

inventory.duplicated().sum()

inventory.drop_duplicates(inplace = True,keep = 'first')

sarimax_data = inventory.drop(['Patient_ID'],axis =1)

inventory['weekofbill'] = inventory['Dateofbill'].dt.isocalendar().week

sarimax_data.drop(['Typeofsales','DrugName','RtnMRP','Final_Sales','Dateofbill'], axis =1, inplace=True)
sarimax_data.drop(['SubCat'], axis =1, inplace=True) #When we drop this feature i got MAPE = 5.89% 
# and it not drops this i got the 2 approximately
sarimax_data.drop(['Specialisation'], axis =1, inplace=True)
sarimax_data.drop(['Quantity'], axis =1, inplace=True)
sarimax_data.info()

numerical = sarimax_data.select_dtypes(['int64','float64']).columns
categorical = sarimax_data.select_dtypes(['object']).columns
num = Pipeline(steps= [('scaling',RobustScaler())])
cat = Pipeline([('encoding',OneHotEncoder())])
preprocess = ColumnTransformer([('scaling',num,numerical),
                                ('encoding',cat,categorical)],remainder=  'passthrough')
preprocess_fit =  preprocess.fit(sarimax_data)


sarimax_data_preprocess = pd.DataFrame(preprocess_fit.transform(sarimax_data).toarray() ,columns=preprocess_fit.get_feature_names_out())

sarimax_data = pd.concat([sarimax_data_preprocess,inventory[['weekofbill','Quantity']]],axis =1)

sarimax_data = sarimax_data.groupby('weekofbill').sum().reset_index()



train = sarimax_data
test = sarimax_data
from statsmodels.tsa.statespace.sarimax import SARIMAX
sarimax_model = SARIMAX(train['Quantity'],exog=train.drop(['Quantity','weekofbill'],axis=1), order=(4, 1, 10),seasonal_order=(1,0,1,52))
sarimax_result = sarimax_model.fit()
pred_sarimax = sarimax_result.predict(start = test.index[0], end = test.index[-1])
sarimax = metrics.mean_absolute_percentage_error(pred_sarimax, test.Quantity)
sarimax = sarimax*100

#-----------VAR model-----------
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_percentage_error

VAR_data = medical_data[['Quantity','t-square', 'log_Quantity']]
model = VAR(VAR_data)
model_fitted = model.fit()
lag_order = model_fitted.k_ar

forecast = model_fitted.forecast(VAR_data.values[-lag_order:], steps=10)
VAR_mape = MAPE(VAR_data[-10:], forecast)
VAR_mape # 17.920312


#--------------FTS MODEL---------------

#pip install scikit-fuzzy
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import mean_absolute_error
# Extract the target column
train_series = train['Quantity'].values
test_series = test['Quantity'].values

# Define fuzzy sets and antecedents
antecedent = ctrl.Antecedent(np.arange(min(train_series), max(train_series), 1), 'input')
consequent = ctrl.Consequent(np.arange(min(train_series), max(train_series), 1), 'output')

# Create fuzzy membership functions
antecedent['low'] = fuzz.trimf(antecedent.universe, [min(train_series), min(train_series), np.median(train_series)])
antecedent['medium'] = fuzz.trimf(antecedent.universe, [min(train_series), np.median(train_series), max(train_series)])
antecedent['high'] = fuzz.trimf(antecedent.universe, [np.median(train_series), max(train_series), max(train_series)])

consequent['low'] = fuzz.trimf(consequent.universe, [min(train_series), min(train_series), np.median(train_series)])
consequent['medium'] = fuzz.trimf(consequent.universe, [min(train_series), np.median(train_series), max(train_series)])
consequent['high'] = fuzz.trimf(consequent.universe, [np.median(train_series), max(train_series), max(train_series)])

# Create fuzzy rules
rule1 = ctrl.Rule(antecedent['low'], consequent['low'])
rule2 = ctrl.Rule(antecedent['medium'], consequent['medium'])
rule3 = ctrl.Rule(antecedent['high'], consequent['high'])
# Create Fuzzy Control System
fts_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
fts_simulation = ctrl.ControlSystemSimulation(fts_ctrl)
# Fit the model with training data
fts_simulation.input['input'] = train_series
fts_simulation.compute()
# Make predictions on test data
fts_predictions = fts_simulation.output['output']
# Calculate MAPE
FTS_mape = MAPE(test_series, fts_predictions)
FTS_mape # 10.280207



#---------------GRU model--------------

from sklearn.preprocessing import MinMaxScaler
train_series = train['Quantity'].values
test_series = test['Quantity'].values
# Normalize the data
scaler = MinMaxScaler()
train_series_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
test_series_scaled = scaler.transform(test_series.reshape(-1, 1))
# Define a function to create sequences for training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

sequence_length = 10  
X_train = create_sequences(train_series_scaled, sequence_length)
y_train = train_series_scaled[sequence_length:]
#pip install tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
# Create a GRU model
model = Sequential()
model.add(GRU(units=50, input_shape=(sequence_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=32)

X_test = create_sequences(test_series_scaled, sequence_length)

predictions_scaled = model.predict(X_test)

predictions = scaler.inverse_transform(predictions_scaled)

GRU_mape = MAPE(test_series[sequence_length:], predictions.flatten())
GRU_mape  # 12.683471




#---------------LSTM model----------------

from sklearn.preprocessing import MinMaxScaler
train_series = train['Quantity'].values
test_series = test['Quantity'].values
# Normalize the data
scaler = MinMaxScaler()
train_series_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
test_series_scaled = scaler.transform(test_series.reshape(-1, 1))
# Define a function to create sequences for training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

sequence_length = 10  
X_train = create_sequences(train_series_scaled, sequence_length)
y_train = train_series_scaled[sequence_length:]
#pip install tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
# Create a LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(sequence_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=10, batch_size=10)

X_test = create_sequences(test_series_scaled, sequence_length)

predictions_scaled = model.predict(X_test)

predictions = scaler.inverse_transform(predictions_scaled)

LSTM_mape = MAPE(test_series[sequence_length:], predictions.flatten())
LSTM_mape   # 12.974484


#------------RNN model--------------------

from sklearn.preprocessing import MinMaxScaler
train_series = train['Quantity'].values
test_series = test['Quantity'].values
# Normalize the data
scaler = MinMaxScaler()
train_series_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
test_series_scaled = scaler.transform(test_series.reshape(-1, 1))
# Define a function to create sequences for training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

sequence_length = 10  
X_train = create_sequences(train_series_scaled, sequence_length)
y_train = train_series_scaled[sequence_length:]
#pip install tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN
model = Sequential()
model.add(SimpleRNN(units=50, input_shape=(sequence_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=10, batch_size=10)

X_test = create_sequences(test_series_scaled, sequence_length)

predictions_scaled = model.predict(X_test)

predictions = scaler.inverse_transform(predictions_scaled)

RNN_mape = MAPE(test_series[sequence_length:], predictions.flatten())
RNN_mape    # 10.363019


#--------comparing all mape's---------
di = pd.Series({'Linear model':lin,'Exponential model':exp,'Moving Average':MV,'Simple Exponential':ses,'Holts Winter':hw,
                'ARIMA':ari,'SARIMA':sari,'SARIMAX':sarimax,'FTS':FTS_mape,'GRU':GRU_mape,'LSTM':LSTM_mape,'RNN':RNN_mape,'VAR':VAR_mape})
mape = pd.DataFrame(di, columns=['mape'])
mape


mape.to_csv('C:/Users/lucky/MAPE Values.csv', index=True)

import os
os.chdir(r"C:\Users\lucky\OneDrive\Desktop\Optimization of Medical Inventory\Model Building")
import joblib
best_model = sarimax_result
joblib.dump(best_model, 'best_model_sarimax.pkl')
joblib.dump(preprocess_fit,'preprocess_fit.pkl')

