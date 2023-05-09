import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ASL_SIGN_MAP = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Load and preprocess the Sign Language MNIST dataset
train_data = pd.read_csv('sign_mnist_train.csv')
test_data = pd.read_csv('sign_mnist_test.csv')


def arrange_data(data):
    y = data['label']
    X = data.drop(columns=['label'])
    X = X.values.reshape(-1, 28, 28, 1) / 255.0
    y = to_categorical(y)
    return X, y

X_train, y_train = arrange_data(train_data)
X_test, y_test = arrange_data(test_data)

# Defining the CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(25, activation='softmax')
])

# Compiling and training the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
datagen.fit(X_train)

# Train the model with the augmented data
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=15, validation_data=(X_test, y_test))

# Evaluating the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

# Initializing MediaPipe Hands (referenced from Google MediaPipe page)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)


def detect_and_crop_hand(frame, hands):
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        # Draw hand landmarks
        annotated_frame = frame.copy()
        mp_drawing.draw_landmarks(annotated_frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # Get the bounding box
        hand_landmarks = result.multi_hand_landmarks[0].landmark
        x_coords = [landmark.x for landmark in hand_landmarks]
        y_coords = [landmark.y for landmark in hand_landmarks]
        x_min, x_max = int(min(x_coords) * frame.shape[1]), int(max(x_coords) * frame.shape[1])
        y_min, y_max = int(min(y_coords) * frame.shape[0]), int(max(y_coords) * frame.shape[0])

        # Check if the cropped region is valid before resizing
        if x_min < x_max and y_min < y_max and (x_max - x_min) > 5 and (y_max - y_min) > 5:
            cropped_frame = frame[y_min:y_max, x_min:x_max]

            # Ensure the cropped frame is not empty before resizing
            if cropped_frame.size > 0:
                resized_frame = cv2.resize(cropped_frame, (28, 28))
                return resized_frame, annotated_frame
    return None, frame


# Access the camera and capture frames
cap = cv2.VideoCapture(0)


def predict_sign(image, model):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = img.reshape(1, 28, 28, 1)
    predictions = model.predict(img)
    sign_class = np.argmax(predictions)
    return ASL_SIGN_MAP[sign_class]


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirroring the frame (either it's my camera or the program itself)
    flipped_frame = cv2.flip(frame, 1)

    # Detecting by MediaPipe and Cropping Hand in Frame
    cropped_hand, annotated_frame = detect_and_crop_hand(flipped_frame, hands)

    if cropped_hand is not None:
        # Making predictions using the trained model
        sign_prediction = predict_sign(cropped_hand, model)
        cv2.putText(annotated_frame, sign_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print(f'Predicted sign: {sign_prediction}')

    # Displaying frame with hand landmarks
    cv2.imshow('Sign Language Recognition', annotated_frame)

    # Enabling user-exit by pressing q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Closing the MediaPipe Hands instance
hands.close()

