import joblib
import serial
import numpy as np

# Load the exported models
svm_clf = joblib.load('svm_model.pkl')
rf_clf = joblib.load('rf_model.pkl')
knn_clf = joblib.load('knn_model.pkl')
elm_clf = joblib.load('elm_model.pkl')
# Load the fifth classifier if available:
# fifth_clf = joblib.load('fifth_model.pkl')

# Setup the serial connection (adjust the COM port, baud rate, etc., as needed)
ser = serial.Serial('COM3', 9600, timeout=1)

print("Waiting for data from the microcontroller...")

while True:
    # Read a line from the serial port
    line = ser.readline().decode('utf-8').strip()
    if line:
        # Assume the incoming line is a comma-separated string of feature values.
        try:
            features = [float(val) for val in line.split(',')]
        except ValueError:
            print("Received malformed data:", line)
            continue

        # Reshape the features to match the classifier's expected input shape (1 sample, n_features)
        input_data = np.array(features).reshape(1, -1)
        
        # Get predictions from each classifier
        svm_pred = svm_clf.predict(input_data)
        rf_pred = rf_clf.predict(input_data)
        knn_pred = knn_clf.predict(input_data)
        elm_pred = elm_clf.predict(input_data)
        # If using a fifth classifier, uncomment the line below:
        # fifth_pred = fifth_clf.predict(input_data)
        
        # Print out the predictions from each classifier
        print("Predictions:")
        print("SVM:", svm_pred[0])
        print("Random Forest:", rf_pred[0])
        print("k-NN:", knn_pred[0])
        print("ELM:", elm_pred[0])
        # print("Fifth Classifier:", fifth_pred[0])
