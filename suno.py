# ASSIGNMENT _ 01 UBER.CSV
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
# reading dataset
df = pd.read_csv("uber.csv")
df
# drop the rows with null values
df.dropna(inplace=True)
df.isnull().sum()
# drop irrelevant columns
df.drop(columns=['key', 'pickup_datetime'], axis=1, inplace=True)
# frequency distribution of fare
sns.distplot(df['fare_amount'])
# outlier removal for fare
upper_limit = df['fare_amount'].quantile(0.88)
lower_limit = df['fare_amount'].quantile(0.01)
print(upper_limit, lower_limit)
df1 = df[(df['fare_amount'] <= upper_limit) & (df['fare_amount'] >= lower_limit)]
df1
sns.boxplot(df1['fare_amount'])
# frequency distribution for passenger_count
sns.distplot(df1['passenger_count'])
# unique values in passenger_counts and their frequency
df1['passenger_count'].value_counts()
# removing 208 passenger_count record
df2 = df1[(df1['passenger_count'] >= 0) & (df1['passenger_count'] <= 6)]
sns.distplot(df2['passenger_count'])
# calculating distance from pickup and dropoff latitudes-longitudes

# create a new column 'dist'
df2['dist'] = None

# convert decimal degrees to radians
pick_lat = np.radians(df2['pickup_latitude'])
pick_lon = np.radians(df2['pickup_longitude'])
drop_lat = np.radians(df2['dropoff_latitude'])
drop_lon = np.radians(df2['dropoff_longitude'])

# difference between latitudes
dlat = drop_lat - pick_lat
# difference between longitudes
dlon = drop_lon - pick_lon

# haversine formula
a = np.sin (dlat / 2)**2 + np.cos(pick_lat) * np.cos(drop_lat) * np.sin(dlon / 2)**2
c = 2 * np.arcsin(np.sqrt(a)) 

# radius of earth in km
r = 6371

# adding the calculated values to dist column
df2['dist'] = c*r

df2
# drop the latitude-longitude columns
df2.drop(columns=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], axis=1, inplace=True)
df2
# correlation
corr = 100 * df2.corr()
sns.heatmap(corr, annot=True, fmt='.1f')
# independent variables
X = df2.drop(columns=['fare_amount'], axis=1)

# dependent variable
y = df2['fare_amount']

# training-testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
print(X_train.shape, X_test.shape)
# model training

lr = LinearRegression()
rf = RandomForestRegressor(random_state=21)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# prediction
lr_preds = lr.predict(X_test)
rf_preds = rf.predict(X_test)
# preformance metrics

lr_r2 = r2_score(y_test, lr_preds)
rf_r2 = r2_score(y_test, rf_preds)

lr_rmse = mean_squared_error(y_test, lr_preds, squared=False)
rf_rmse = mean_squared_error(y_test, rf_preds, squared=False)

print("R2 scores:")
print("Linear Regression:", lr_r2)
print("Random Forest Regression:", rf_r2)
print("RMSE:")
print("Linear Regression:", lr_rmse)
print("Random Forest Regression:", rf_rmse)


# ASSIGNMENT -02 EMAIL SPAM
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# reading dataset
df = pd.read_csv('emails.csv')
df
# independent variables
X = df.drop(columns=['Prediction', 'Email No.'], axis=1)

# dependent variable
y = df['Prediction']

# training_testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
print(X_train.shape, X_test.shape)
# knn model training
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)

# prediction
knn_pred = knn_classifier.predict(X_test)
# knn performance metrics
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)

print("K-Nearest Neighbors Classifier:")
print(f"Accuracy: {knn_accuracy}")
print(f"Precision: {knn_precision}")
print(f"Recall: {knn_recall}")
print(f"F1 Score: {knn_f1}")
# svm model training
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)

# prediction
svm_pred = svm_classifier.predict(X_test)
# svm performance metrics
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)

print("Support Vector Machine Classifier:")
print(f"Accuracy: {svm_accuracy}")
print(f"Precision: {svm_precision}")
print(f"Recall: {svm_recall}")
print(f"F1 Score: {svm_f1}")
# confusion matrices

knn_cm = confusion_matrix(y_test, knn_pred)
sns.heatmap(knn_cm, annot=True, fmt='d')
plt.ylabel("Predicted Labels")
plt.xlabel("True Labels")
plt.title("KNN Confusion Matrix")
plt.show()

svm_cm = confusion_matrix(y_test, svm_pred)
sns.heatmap(svm_cm, annot=True, fmt='d')
plt.ylabel("Predicted Labels")
plt.xlabel("True Labels")
plt.title("SVM Confusion Matrix")
plt.show()

# ASSIGNMENT -03 NEURAL NETWORK_BANK CUSTOMER

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# reading dataset
data = pd.read_csv('Churn_Modelling.csv')
data
data.info()
data.isnull().sum()
# visualization - categorical variables

def count(x, fig):
    plt.subplot(6, 3, fig)
    sns.countplot(x=x, data=data, hue='Exited')

plt.figure(figsize=(15,30))

count('Geography',1)
count('Gender',2)
count('IsActiveMember',3)
count('HasCrCard',4)
count('Tenure',5)
count('NumOfProducts',6)
# visualization - continous variables

def count(x,num_bins, fig):
    plt.subplot(6,2,fig)
    sns.countplot(x=(pd.cut(data[x], bins=num_bins)),data=data,hue='Exited')

plt.figure(figsize=(15,30))

count('Age',5,1)
count('Balance',3,2)
count('EstimatedSalary',4,3)
# drop first 3 columns i.e. row_number, customer_id, surname
df=data.iloc[:,3:]
df
# one hot encoding 'geography' and 'gender' columns

df=pd.concat([df,pd.get_dummies(df['Geography'])],axis=1)
df=df.drop(columns=['Geography'])

df=pd.concat([df,pd.get_dummies(df['Gender'])],axis=1)
df=df.drop(columns=['Gender'])

# removing redundant 'female' column
df=df.rename(columns = {'Male':'Gender'})
df=df.drop(columns=['Female'])
# correlation
corr = 100 * df.corr()
sns.heatmap(corr, annot=True, fmt='.1f')
# standardization of continous variables
scaler = StandardScaler()
cols_to_scale = ['Tenure','EstimatedSalary','NumOfProducts','Age','Balance','CreditScore']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
# independent variables
X=df.drop(columns=['Exited'])

# dependent variable
y=df.Exited

X.shape,y.shape
# training-testing split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
X_train.shape, X_test.shape
# building neural network

model = keras.Sequential([
    # input layer
    keras.layers.Dense(24, activation='relu', input_shape=(12,)),
    # hidden layer
    keras.layers.Dense(48, activation='relu'),
    # output layer
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=32)
# prediction
y_pred = model.predict(X_test)

# converting perdictions to binary form (0 or 1)
y_pred_binary = (y_pred >= 0.5).astype(int)
# performance metrics

accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
# confusion matrix

cm = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d')
plt.ylabel("Predicted Labels")
plt.xlabel("True Labels")
plt.show()


# ASSIGNMENT -04 KNN Diabetes.csv
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# reading dataset
df = pd.read_csv("diabetes.csv")
df
df.info()
df.isnull().sum()
# visualization - all columns
for i in df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age']]:
    fig, axes = plt.subplots(1, 2, figsize = (18,5))
    sns.histplot(df[i], ax=axes[0])
    sns.boxplot(df[i], ax=axes[1], orient='h')
    plt.show()
# below values cannot be 0
zero_features = ['Glucose','BloodPressure','SkinThickness',"Insulin",'BMI']

# replacing the 0 values with mean value
for i in zero_features:
    df[i]=df[i].replace(0, df[i].mean())

# visualization
for i in df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]:
    fig, axes = plt.subplots(1, 2, figsize = (18,5))
    sns.histplot(df[i], ax=axes[0])
    sns.boxplot(df[i], ax=axes[1], orient='h')
    plt.show()
# outlier removal

def remove_outliers(col_name, df):
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    df = df[(df[col_name] <= upper_limit) & (df[col_name] >= lower_limit)]
    return df

col_list = ['SkinThickness','Insulin', 'BMI']

for i in col_list:
    df = remove_outliers(i, df)

# visualiation
for i in df[['SkinThickness', 'Insulin', 'BMI']]:
    fig, axes = plt.subplots(1, 2, figsize = (18,5))
    sns.histplot(df[i], ax=axes[0])
    sns.boxplot(df[i], ax=axes[1], orient='h')
    plt.show()
# independent variables
X = df.drop(columns='Outcome', axis=1)

# dependent variable
y = df['Outcome']

# standardization of independent variables
scaler = StandardScaler()
X = scaler.fit_transform(X)

# training-testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
print(X_train.shape, X_test.shape)
# formula for optimal values of k
k = round(np.sqrt(X_train.shape[0]))

# training model
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

# prediction
y_pred = knn_classifier.predict(X_test)
# performance metrics

knn_accuracy = accuracy_score(y_test, y_pred)
knn_error_rate = 1 - knn_accuracy
knn_precision = precision_score(y_test, y_pred)
knn_recall = recall_score(y_test, y_pred)
knn_f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {knn_accuracy}")
print(f"Error Rate: {knn_error_rate}")
print(f"Precision: {knn_precision}")
print(f"Recall: {knn_recall}")
print(f"F1 Score: {knn_f1}")
#confusion matrix

knn_cm = confusion_matrix(y_test, y_pred)
sns.heatmap(knn_cm, annot=True, fmt='d')
plt.ylabel("Predicted Labels")
plt.xlabel("True Labels")
plt.title("KNN Confusion Matrix")
plt.show()

# ASSIGNMENT 5 - SALES DATA SAMPLE
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
# reading dataset
df =  pd.read_csv("sales_data_sample.csv", encoding='latin')
df
df.info()
# dropping irrelevant columns
cols = ['ORDERNUMBER', 'ORDERDATE', 'STATUS', 'PRODUCTCODE', 'CUSTOMERNAME', 'PHONE', 'ADDRESSLINE1', 'ADDRESSLINE2', 'CITY', 'STATE', 'POSTALCODE', 'TERRITORY', 'CONTACTLASTNAME', 'CONTACTFIRSTNAME']
df = df.drop(cols, axis=1)

# checking null values
df.isnull().sum()
# unique values
print('COUNTRY')
print(df['COUNTRY'].unique())
print('PRODUCTLINE')
print(df['PRODUCTLINE'].unique())
print('DEALSIZE')
print(df['DEALSIZE'].unique())
# one hot encoding
productline = pd.get_dummies(df['PRODUCTLINE'])
dealsize = pd.get_dummies(df['DEALSIZE'])
df = pd.concat([df,productline, dealsize], axis = 1)

# dropping redundant columns and 'country' - too much classes
df_drop  = ['COUNTRY','PRODUCTLINE','DEALSIZE']
df = df.drop(df_drop, axis=1)
# finding optimal value of k - elbow method

wcss = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df)
    wcss.append(kmeanModel.inertia_)

plt.plot(K, wcss, 'bx-') 
plt.xlabel('k') 
plt.ylabel('wcss') 
plt.title('The Elbow Method showing the optimal k') 
plt.show() 
# training set
X_train = df.values
X_train.shape
# model training
model = KMeans (n_clusters=2, random_state=21)
model = model.fit(X_train) 

# prediction
predictions = model.predict(X_train) 
predictions
# no. of data points in each cluster

unique, counts = np.unique(predictions, return_counts=True)
counts = counts.reshape(1,2)
counts_df = pd.DataFrame (counts, columns=['Cluster1', 'Cluster2'])
counts_df.head()
# centroids of the clusters
model.cluster_centers_
# dendrogram
dendrogram = sch.dendrogram(sch.linkage(X_train, method = 'ward')) 
plt.title('Dendrogram') 
plt.xlabel('Customers') 
plt.ylabel('Euclidean distances')
plt.show()
# heirarchical clusternig - model trainning
model = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')

# prediction
predictions = model.fit_predict(X_train)
# no. of data points in clusters
unique, counts = np.unique(predictions, return_counts=True)
counts = counts.reshape(1,2)
counts_df = pd.DataFrame (counts, columns=['Cluster1', 'Cluster2'])
counts_df.head()
