import numpy as np
from keras.models import load_model
import firebase_admin
from firebase_admin import credentials, db
import time

# Initialize Firebase with your credentials
cred = credentials.Certificate('D:\BCU Final\ML\Dataset\Wesad Acc\datasrc\heartrate-883af-firebase-adminsdk-bi4j9-ada56e6d3a.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://heartrate-883af-default-rtdb.europe-west1.firebasedatabase.app/'
})


model = load_model('D:\BCU Final\ML\Dataset\Wesad Acc\datasrc\84precent.h5')

# Create a reference to the 'sensor_data' node in your Firebase database
ref = db.reference('/sensor_data/acceleration')
ref2 = db.reference('/sensor_data/Prediction')

while True:
# Retrieve data from the database
    acceleration_data = ref.get()

    # Process the retrieved data
    for key, data in acceleration_data.items():
        chest_acc_x = acceleration_data.get('x', None)
        chest_acc_y = acceleration_data.get('y', None)
        chest_acc_z = acceleration_data.get('z', None)

        print(f"Key: {key}, Acceleration - x: {chest_acc_x}, y: {chest_acc_y}, z: {chest_acc_z}")

        # Preprocess input data
        input_data = np.array([[chest_acc_x, chest_acc_y, chest_acc_z, 0.0, 0.0]])
        # Apply any necessary preprocessing steps (e.g., normalization, reshaping)

        # Make predictions
        predictions = model.predict(input_data)

        # Postprocess output (this depends on your model's output format)
        # For example, if it's a classification task and predictions are probabilities
        predicted_label = int(np.argmax(predictions))  # Assuming the highest probability is the predicted class

        # If it's a regression task, you might directly use predictions

        # Display the result to the user
        print(f"Predicted Label: {predicted_label}")
        # Push the accelerometer data to the database
        new_sensor_data = ref2.set({
                    'Prediction': {
                                'pred': predicted_label,
        
                                    }
                            }   )
    
    time.sleep(60)








