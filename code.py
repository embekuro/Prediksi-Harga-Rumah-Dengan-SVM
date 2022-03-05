# Mengimpor library
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# impor dataset
mulai = time.time()
dataset = pd.read_csv('kc_house_data.csv')
x = dataset.iloc[:, 0:18].values # Mengambil data dari kolom 1 sampai 18
y = dataset.iloc[:, 18].values # Mengambil data dari kolom 19

# Split dataset ke dalam Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.25, random_state = 0)
print('\nTraining Set [ok]\nTest Set [ok]')

# Preprocessing
mm=MinMaxScaler()
x_train = mm.fit_transform(x_train)
x_test = mm.transform(x_test)
print('Feature Scaling [ok]')

# Metode SVR
from sklearn.svm import SVR
regresi = SVR(kernel='rbf', C=100000, gamma=1)
regresi.fit(x_train, y_train)
print('Model SVR [ok]')
y_pred = regresi.predict(x_test)

# Akurasi
o = int(len(y_test))
print('Prediksi [ok]\nData test',o)
akurasi = 0
for i in range(o):
    z = (y_pred[i]/y_test[i])*100
    if z>100 and z<=200:
        z = z-100
        z = 100-z
    elif z>200 or z<0:
        z = 0
    akurasi = akurasi+z
print('Harga Prediksi : ', y_pred, '\nHarga Asli : ', y_test, '\nAkurasi : ', akurasi/o)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rdua = r2_score(y_test, y_pred)
waktu = time.time()-mulai

print('Harga Prediksi : ', y_pred, '\nHarga Asli : ', y_test, '\nRata-rata perbandingan harga prediksi\
harga asli(persen) : ', akurasi/o, '\nMAE:', mae, '\nMSE:', mse, '\nr2:', rdua, '\nwaktu(detik):', waktu)
