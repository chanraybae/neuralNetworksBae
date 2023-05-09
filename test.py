import os
import os.path
import cv2
import mediapipe as mp
import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Increased image size, can scale for optimal output
IMAGE_SIZE = 64

def preprocess_image(image):
    # Converting to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resizing the image
    resized_image = cv2.resize(gray_image, (IMAGE_SIZE, IMAGE_SIZE))
    # Normalizing the image
    normalized_image = resized_image / 255.0
    return normalized_image

# non-essential identifier function for this script
def predict_ASL(image, model):
    preprocessed_image = preprocess_image(image)
    input_image = np.expand_dims(preprocessed_image, axis=(0, -1))
    prediction = model.predict(input_image)
    prediction_number = np.argmax(prediction)

    return ASL_SIGNS[prediction_number]


# Adding the detect_and_crop_hand function here
def detect_and_crop_hand(image, hands):
    height, width, _ = image.shape
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_hand_landmarks:
        return None, None

    hand_landmarks = results.multi_hand_landmarks[0]
    hand_points = [tuple(np.round(np.array([landmark.x * width, landmark.y * height])).astype(int)) for landmark in hand_landmarks.landmark]

    x_coords, y_coords = zip(*hand_points)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)

    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image, hand_landmarks


def load_images_from_folder(folder):
    images = []
    labels = []
    # Using the MediaPipe hand detector (using Google reference pages) to capture hand
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if not os.path.isdir(label_path):
            continue

        for filename in os.listdir(label_path):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in ['.jpg', '.jpeg']:
                continue

            img_path = os.path.join(label_path, filename)
            img = cv2.imread(img_path)

            cropped_hand, _ = detect_and_crop_hand(img, hands)
            if cropped_hand is not None:
                images.append(cropped_hand)
                labels.append(label)

    hands.close()
    return images, labels


# Loading data from folders
train_folder = "train"
test_folder = "test"
valid_folder = "valid"

# Creating a label mapping
label_mapping = {label: i for i, label in enumerate(sorted(os.listdir(train_folder)))}

x_train, y_train = load_images_from_folder(train_folder)
x_test, y_test = load_images_from_folder(test_folder)
x_valid, y_valid = load_images_from_folder(valid_folder)

# Preprocessing images
x_train = np.array([preprocess_image(img) for img in x_train]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
x_test = np.array([preprocess_image(img) for img in x_test]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
x_valid = np.array([preprocess_image(img) for img in x_valid]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

# Using list comprehension for training sets
y_train = [label_mapping[label] for label in y_train]
y_test = [label_mapping[label] for label in y_test]
y_valid = [label_mapping[label] for label in y_valid]

# Preparing labels
y_train = to_categorical(y_train, num_classes=26)
y_test = to_categorical(y_test, num_classes=26)
y_valid = to_categorical(y_valid, num_classes=26)


def print_data_shapes(x_train, y_train, x_test, y_test, x_valid, y_valid):
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    print("x_valid shape:", x_valid.shape)
    print("y_valid shape:", y_valid.shape)

# Defining the CNN model, adjusted as accuracy was initially too low
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(26, activation='softmax'))

print_data_shapes(x_train, y_train, x_test, y_test, x_valid, y_valid)

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Configure early stopping and checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=20)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Data augmentation (ChatGPT suggested as mentioned below)
data_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                    height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)

# Training the model, enhanced from basic to data augmented using ChatGPT
history = model.fit(data_generator.flow(x_train, y_train, batch_size=32),
                    epochs=100,
                    validation_data=(x_valid, y_valid),
                    callbacks=[early_stopping, model_checkpoint])


# Loading the best model
model.load_weights('best_model.h5')

# Evaluating the model on test data
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}, Test accuracy: {accuracy}")