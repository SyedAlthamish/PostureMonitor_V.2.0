import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle
import tensorflow as tf

# Load dataset
dataset = pd.read_csv(r"C:\Users\saisr\Desktop\X_COMP.csv")

# Separate features and target
X = dataset.iloc[:, :-1].values  # All columns except last as features
y = dataset.iloc[:, -1].values  # Last column as target

# Label encoding for the target
label = LabelEncoder()
y = label.fit_transform(y)

# One-hot encoding the target labels for multiclass classification
y = to_categorical(y)

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=200)

# Feature scaling for the independent variables
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Reshaping the input data to 3D for LSTM: [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=y.shape[1], activation='softmax'))

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Saving the model to a file
model.save('lstm_model.h5')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("lstm_model.tflite", "wb") as f:
    f.write(tflite_model)

# Saving the scaler and label encoder for future use
pickle.dump(sc, open('scaler.pkl', 'wb'))
pickle.dump(label, open('label_encoder.pkl', 'wb'))

# Predicting with the model (example)
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Map predictions back to original labels
predicted_labels = label.inverse_transform(predicted_classes)

# Print predictions vs true values
print(np.concatenate((predicted_labels.reshape(len(predicted_labels), 1), 
                      np.argmax(y_test, axis=1).reshape(len(y_test), 1)), 1))

# Accuracy score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Classification Report:")
print(classification_report(np.argmax(y_test, axis=1), predicted_classes))

# Confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), predicted_classes)
print("Confusion Matrix:")
print(cm)
