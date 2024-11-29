import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time

#############################################
# CAMERA SETTINGS
frameWidth = 640         # Camera resolution
frameHeight = 480
brightness = 180         # Brightness level
threshold = 0.75         # Probability threshold
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP VIDEO CAPTURE
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)  # Set width
cap.set(4, frameHeight)  # Set height
cap.set(10, brightness)  # Set brightness

# LOAD TRAINED MODEL
model = load_model("model.h5")

# IMAGE PREPROCESSING FUNCTIONS
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255  # Normalize the image
    return img

# FUNCTION TO MAP CLASS INDEX TO CLASS NAME
def getClassName(classNo):
    classNames = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
        'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
        'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    return classNames[classNo]

# MAIN LOOP
while True:
    # Capture frame from video
    success, imgOriginal = cap.read()
    
    if not success:
        break

    start_time = time.time()

    # Process image
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    # Predict using the model
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.amax(predictions)

    # Display results
    if probabilityValue > threshold:
        cv2.putText(imgOriginal, str(classIndex) + " " + str(getClassName(classIndex)), 
                    (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", 
                    (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(imgOriginal, "Uncertain", (120, 35), font, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(imgOriginal, f"FPS: {int(fps)}", (500, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    # Display instructions to quit
    cv2.putText(imgOriginal, "Press 'q' to quit", (20, 450), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Show results
    cv2.imshow("Result", imgOriginal)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
