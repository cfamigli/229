
import numpy as np
from pandas import read_csv, DataFrame
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

data = read_csv('fire_data_2001_2017.csv')
print(list(data.columns.values))

data_cond = DataFrame()
for i in range(2001,2018):
    subset = data.loc[data['year']==i].drop_duplicates(subset=['lat', 'lon'], keep='first')
    data_cond = data_cond.append(subset)

clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(10, 2), random_state=1)

data_cond_train = data_cond.loc[data_cond['year']!=2017]
data_cond_test = data_cond.loc[data_cond['year']==2017]

X_train = data_cond_train.loc[:, (data_cond_train.columns != 'system.index') & (data_cond_train.columns != 'fire')
     & (data_cond_train.columns != 'lat')  & (data_cond_train.columns != 'lon') & (data_cond_train.columns != 'doy')
     & (data_cond_train.columns != 'year')].values
X_train_refl = data_cond_train[['B_1m',	'B_1w',	'B_3m',	'G_1m',	'G_1w',	'G_3m', 'NBR1_1m',
    'NBR1_1w', 'NBR1_3m', 'NBR2_1m', 'NBR2_1w',	'NBR2_3m',	'NDMI_1m',	'NDMI_1w',	'NDMI_3m',	'NDVI_1m',	'NDVI_1w',
    'NDVI_3m',	'NDWI_1m', 'NDWI_1w', 'NDWI_3m', 'GCVI_1m',	'GCVI_1w', 'GCVI_3m', 'NIR_1m', 'NIR_1w', 'NIR_3m',
    'R_1m',	'R_1w', 'R_3m', 'SWIR1_1m', 'SWIR1_1w', 'SWIR1_3m', 'SWIR2_1m',	'SWIR2_1w',
    'SWIR2_3m',	'TCG_1m', 'TCG_1w', 'TCG_3m', 'TCW_1m', 'TCW_1w', 'TCW_3m',
    'GCVI_1m', 'GCVI_1w', 'GCVI_3m', 'LC']]
X_train_climate = data_cond_train[['ET_1d', 'ET_1d_clim', 'ET_1m', 'ET_1m_clim', 'ET_1w', 'ET_1w_clim', 'ET_3m', 'ET_3m_clim',
    'LC', 'LST_Day_1km_1m', 'LST_Day_1km_1w', 'LST_Day_1km_3m', 'SM_0_10_1d', 'SM_0_10_1d_clim', 'SM_0_10_1m', 'SM_0_10_1m_clim',
    'SM_0_10_1w', 'SM_0_10_1w_clim', 'SM_0_10_3m', 'SM_0_10_3m_clim', 'SM_100_200_1d', 'SM_100_200_1d_clim', 'SM_100_200_1m',
    'SM_100_200_1m_clim', 'SM_100_200_1w', 'SM_100_200_1w_clim', 'SM_100_200_3m', 'SM_100_200_3m_clim', 'SM_10_40_1d',
    'SM_10_40_1d_clim', 'SM_10_40_1m', 'SM_10_40_1m_clim', 'SM_10_40_1w', 'SM_10_40_1w_clim', 'SM_10_40_3m', 'SM_10_40_3m_clim',
    'SM_40_100_1d', 'SM_40_100_1d_clim', 'SM_40_100_1m', 'SM_40_100_1m_clim', 'SM_40_100_1w', 'SM_40_100_1w_clim', 'SM_40_100_3m',
    'SM_40_100_3m_clim', 'WS_1d', 'WS_1d_clim', 'WS_1m', 'WS_1m_clim', 'WS_1w', 'WS_1w_clim', 'WS_3m', 'WS_3m_clim',
    'precip_1d', 'precip_1d_clim', 'precip_1m', 'precip_1m_clim', 'precip_1w', 'precip_1w_clim', 'precip_3m', 'precip_3m_clim']]
X_test = data_cond_test.loc[:, (data_cond_test.columns != 'system.index') & (data_cond_test.columns != 'fire')
     & (data_cond_test.columns != 'lat')  & (data_cond_test.columns != 'lon') & (data_cond_test.columns != 'doy')
     & (data_cond_test.columns != 'year')].values
X_test_refl = data_cond_test[['B_1m',	'B_1w',	'B_3m',	'G_1m',	'G_1w',	'G_3m', 'NBR1_1m',
     'NBR1_1w', 'NBR1_3m', 'NBR2_1m', 'NBR2_1w',	'NBR2_3m',	'NDMI_1m',	'NDMI_1w',	'NDMI_3m',	'NDVI_1m',	'NDVI_1w',
     'NDVI_3m',	'NDWI_1m', 'NDWI_1w', 'NDWI_3m', 'GCVI_1m',	'GCVI_1w', 'GCVI_3m', 'NIR_1m', 'NIR_1w', 'NIR_3m',
     'R_1m',	'R_1w', 'R_3m', 'SWIR1_1m', 'SWIR1_1w', 'SWIR1_3m', 'SWIR2_1m',	'SWIR2_1w',
     'SWIR2_3m',	'TCG_1m', 'TCG_1w', 'TCG_3m', 'TCW_1m', 'TCW_1w', 'TCW_3m',
     'GCVI_1m', 'GCVI_1w', 'GCVI_3m', 'LC']]
X_test_climate = data_cond_test[['ET_1d', 'ET_1d_clim', 'ET_1m', 'ET_1m_clim', 'ET_1w', 'ET_1w_clim', 'ET_3m', 'ET_3m_clim',
    'LC', 'LST_Day_1km_1m', 'LST_Day_1km_1w', 'LST_Day_1km_3m', 'SM_0_10_1d', 'SM_0_10_1d_clim', 'SM_0_10_1m', 'SM_0_10_1m_clim',
    'SM_0_10_1w', 'SM_0_10_1w_clim', 'SM_0_10_3m', 'SM_0_10_3m_clim', 'SM_100_200_1d', 'SM_100_200_1d_clim', 'SM_100_200_1m',
    'SM_100_200_1m_clim', 'SM_100_200_1w', 'SM_100_200_1w_clim', 'SM_100_200_3m', 'SM_100_200_3m_clim', 'SM_10_40_1d',
    'SM_10_40_1d_clim', 'SM_10_40_1m', 'SM_10_40_1m_clim', 'SM_10_40_1w', 'SM_10_40_1w_clim', 'SM_10_40_3m', 'SM_10_40_3m_clim',
    'SM_40_100_1d', 'SM_40_100_1d_clim', 'SM_40_100_1m', 'SM_40_100_1m_clim', 'SM_40_100_1w', 'SM_40_100_1w_clim', 'SM_40_100_3m',
    'SM_40_100_3m_clim', 'WS_1d', 'WS_1d_clim', 'WS_1m', 'WS_1m_clim', 'WS_1w', 'WS_1w_clim', 'WS_3m', 'WS_3m_clim',
    'precip_1d', 'precip_1d_clim', 'precip_1m', 'precip_1m_clim', 'precip_1w', 'precip_1w_clim', 'precip_3m', 'precip_3m_clim']]

y_train = data_cond_train.fire.values
y_test = data_cond_test.fire.values

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
print('ALL TEST ACCURACY:')
print(accuracy_score(y_test, pred))
print('ALL TRAIN ACCURACY:')
print(accuracy_score(y_train, clf.predict(X_train)))

clf.fit(X_train_refl, y_train)
pred = clf.predict(X_test_refl)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
print('REFL TEST ACCURACY:')
print(accuracy_score(y_test, pred))
print('REFL TRAIN ACCURACY:')
print(accuracy_score(y_train, clf.predict(X_train_refl)))

clf.fit(X_train_climate, y_train)
pred = clf.predict(X_test_climate)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
print('CLIM TEST ACCURACY:')
print(accuracy_score(y_test, pred))
print('CLIM TRAIN ACCURACY:')
print(accuracy_score(y_train, clf.predict(X_train_climate)))
