import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# === Hyperparameters ===
EPOCHS = 50  # Number of epochs for training
BATCH_SIZE = 32  # Batch size for training
LEARNING_RATE = 0.001  # Learning rate for optimizer
DROPOUT_RATE = 0.3  # Dropout rate for preventing overfitting
VALIDATION_SPLIT = 0.2  # Validation split ratio
OPTIMIZER_TYPE = 'adam'  # Optimizer choice ('adam' or 'sgd')
LOSS_WEIGHTS = {'luminosity_output': 1.0, 'type_output': 1.0}  # Loss weights for multi-output

# Set paths
DATASET_PATH = "star_dataset.csv"
IMAGE_DATA_PATH = "image_data.npy"  # Replace with your preprocessed image file path
MODEL_SAVE_PATH = "star_analysis_ai.h5"

# Step 1: Load and Inspect Data
def load_data():
    data = pd.read_csv(DATASET_PATH)
    print("Dataset Loaded:")
    print(data.head())
    print(data.info())

    # Load images (replace with actual preprocessing)
    images = np.load(IMAGE_DATA_PATH)
    print(f"Loaded {images.shape[0]} images.")

    return data, images

# Step 2: Data Preprocessing
def preprocess_data(data):
    scaler = MinMaxScaler()
    data[["Diameter", "Luminosity (Absolute Magnitude)"]] = scaler.fit_transform(
        data[["Diameter", "Luminosity (Absolute Magnitude)"]]
    )

    label_encoder = LabelEncoder()
    data["Type (OBAFGKM Scale)"] = label_encoder.fit_transform(data["Type (OBAFGKM Scale)"])
    num_classes = len(label_encoder.classes_)

    return data, scaler, label_encoder, num_classes

def preprocess_image(image_path):
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image) / 255.0
    return image

# Step 3: Split Data
def split_data(data, images):
    numerical_data = data[["Diameter"]].values
    targets_luminosity = data["Luminosity (Absolute Magnitude)"].values
    targets_type = data["Type (OBAFGKM Scale)"].values

    (
        X_train_img, X_test_img, 
        X_train_num, X_test_num, 
        y_train_lum, y_test_lum, 
        y_train_type, y_test_type
    ) = train_test_split(
        images, numerical_data, targets_luminosity, targets_type, 
        test_size=0.2, random_state=42
    )

    return X_train_img, X_test_img, X_train_num, X_test_num, y_train_lum, y_test_lum, y_train_type, y_test_type

# Step 4: Build the Model
def build_model(num_classes):
    image_input = tf.keras.Input(shape=(128, 128, 3), name="image_input")
    x = layers.Conv2D(32, (3, 3), activation="relu")(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    numerical_input = tf.keras.Input(shape=(1,), name="numerical_input")
    y = layers.Dense(32, activation="relu")(numerical_input)
    y = layers.Dense(64, activation="relu")(y)

    combined = layers.Concatenate()([x, y])
    combined = layers.Dense(128, activation="relu")(combined)
    combined = layers.Dense(64, activation="relu")(combined)

    luminosity_output = layers.Dense(1, name="luminosity_output")(combined)
    type_output = layers.Dense(num_classes, activation="softmax", name="type_output")(combined)

    model = tf.keras.Model(inputs=[image_input, numerical_input], outputs=[luminosity_output, type_output])

    if OPTIMIZER_TYPE == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    elif OPTIMIZER_TYPE == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)

    model.compile(
        optimizer=optimizer,
        loss={
            "luminosity_output": "mse",
            "type_output": "sparse_categorical_crossentropy",
        },
        metrics={
            "luminosity_output": "mse",
            "type_output": "accuracy",
        },
        loss_weights=LOSS_WEIGHTS
    )
    return model

# Step 5: Train the Model
def train_model(model, X_train_img, X_train_num, y_train_lum, y_train_type):
    history = model.fit(
        {"image_input": X_train_img, "numerical_input": X_train_num},
        {"luminosity_output": y_train_lum, "type_output": y_train_type},
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    return history

# Step 6: Evaluate the Model
def evaluate_model(model, X_test_img, X_test_num, y_test_lum, y_test_type):
    results = model.evaluate(
        {"image_input": X_test_img, "numerical_input": X_test_num},
        {"luminosity_output": y_test_lum, "type_output": y_test_type},
    )
    print("Test Loss and Metrics:", results)

# Step 7: Save the Model
def save_model(model):
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

# Step 8: Predict on New Data
def predict_star(model, scaler, label_encoder):
    new_diameter = 2400000
    dummy_luminosity = 0.0
    new_image_path = "C:/Users/Jyoti/OneDrive/Desktop/Coding/SciRe 2024-25 STAHZAI/download.jpg"

    new_image = preprocess_image(new_image_path)
    new_image = np.expand_dims(new_image, axis=0)

    new_features = np.array([[new_diameter, dummy_luminosity]])
    new_features_scaled = scaler.transform(new_features)
    new_diameter_scaled = new_features_scaled[:, 0].reshape(-1, 1)

    predicted_luminosity, predicted_type = model.predict([new_image, new_diameter_scaled])
    predicted_type_label = label_encoder.inverse_transform([np.argmax(predicted_type)])

    print(f"Predicted Luminosity: {predicted_luminosity[0][0]}")
    print(f"Predicted Type: {predicted_type_label[0]}")

# Main Script
if __name__ == "__main__":
    data, images = load_data()
    data, scaler, label_encoder, num_classes = preprocess_data(data)

    X_train_img, X_test_img, X_train_num, X_test_num, y_train_lum, y_test_lum, y_train_type, y_test_type = split_data(data, images)

    model = build_model(num_classes)

    train_model(model, X_train_img, X_train_num, y_train_lum, y_train_type)

    evaluate_model(model, X_test_img, X_test_num, y_test_lum, y_test_type)

    save_model(model)

    predict_star(model, scaler, label_encoder)
