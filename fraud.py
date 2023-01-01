import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Load and preprocess the data
data = pd.read_csv("fraud_data.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversample the training set using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Convert the data to a TensorFlow dataset
ds_train = tf.data.Dataset.from_tensor_slices((X_train_resampled, y_train_resampled))
ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Shuffle and batch the data
ds_train = ds_train.shuffle(1000).batch(32)
ds_test = ds_test.batch(32)
# Train a logistic regression model
clf1 = LogisticRegression(random_state=42)
clf1.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred1 = clf1.predict(X_test)

# Train a decision tree model
clf2 = DecisionTreeClassifier(random_state=42)
clf2.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred2 = clf2.predict(X_test)

# Define a neural network using TensorFlow
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_resampled.shape[1],)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model to the training data
history = model.fit(ds_train, epochs=10)

# Make predictions on the test set
y_pred3 = model.predict(ds_test)

# Evaluate the models using precision, recall, and the F1 score
precision1 = precision_score(y_test, y_pred1)
recall1 = recall_score(y_test, y_pred1)
f1_score1 = f1_score(y_test, y_pred1)

precision2 = precision_score(y_test, y_pred2)
recall2 = recall_score(y_test, y_pred2)
f1_score2 = f1_score(y_test, y_pred2)

precision3 = precision_score(y_test, y_pred3.round())
recall3 = recall_score(y_test, y_pred3.round())
f1_score3 = f1_score(y_test, y_pred3.round())

print("Logistic Regression:")
print("Precision: ", precision1)
print("Recall: ", recall1)
print("F1 Score: ", f1_score1)

print("Decision Tree:")
print("Precision: ", precision2)
print("Recall: ", recall2)
print("F1 Score: ", f1_score2)

print("Neural Network:")
print("Precision: ", precision3)
print("Recall: ", recall3)
print("F1 Score: ", f1_score3)
