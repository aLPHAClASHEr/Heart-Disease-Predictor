import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

heart_data = pd.read_csv('heart.csv')
print(heart_data.head(3))

print(heart_data.shape)

# Checking for missing values in the dataset
print(heart_data.isnull().sum())

# Statistical analysis of the dataset
print(heart_data.describe())

# Checking the Dataset for target values
print(heart_data['target'].value_counts())

# Splitting features and targets
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print(X)
print(Y)


# Splitting the Dataset into training data and testing data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,stratify=Y,random_state=2)

print(X_train.shape, X_test.shape, X.shape)

# MODEL TRAINING
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, Y_train)

# ACCURACY TEST
X_train_prediction = model.predict(X_train)
training_data_prediction = accuracy_score(X_train_prediction, Y_train)
print('Training Data Accuracy:', training_data_prediction*100 , '%')

X_test_prediction = model.predict(X_test)
test_data_prediction = accuracy_score(X_test_prediction, Y_test)
print('Testing Data Accuracy:', test_data_prediction*100 , '%')


# MODEL BUILDING FOR PREDICTION
input_data = (54,1,0,122,286,0,0,116,1,3.2,1,2,2)
#input_data = tuple(float(a) for a in input_data.split(","))
#print(input_data)

# Changing input data to numpy array
new_array = np.asarray(input_data)

# Reshaping the numpy array as we are predicting for only one instance
input_data_reshaped = new_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The person does not have a heart Disease')
else:
    print('The person has Heart Disease')



filename = 'heart_disease_model.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))

input_data = (54,1,0,122,286,0,0,116,1,3.2,1,2,2)
#input_data = tuple(float(a) for a in input_data.split(","))
#print(input_data)

# Changing input data to numpy array
new_array = np.asarray(input_data)

# Reshaping the numpy array as we are predicting for only one instance
input_data_reshaped = new_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The person does not have a heart Disease')
else:
    print('The person has Heart Disease')
