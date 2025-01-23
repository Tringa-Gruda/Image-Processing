import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# --- 1. Data Preprocessing ---
data_dir = "C:/Users/tring/Desktop/Image Processing/skin-disease-datasaet/train_set"
img_size = (299, 299)  

# Load images and labels
def load_and_preprocess_images(data_dir, img_size):
    img_data = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    class_indices = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for file in os.listdir(class_dir):
            if file.endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(class_dir, file)
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img) / 255.0
                img_data.append(img_array)
                labels.append(class_indices[class_name])

    return np.array(img_data), np.array(labels), class_names

# Load images and labels
images, labels, class_names = load_and_preprocess_images(data_dir, img_size)
print(f"Loaded {len(images)} images.")

# --- 2. Feature Extraction Using InceptionV3 ---
base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# Extract features
features = base_model.predict(images)
print(f"Extracted features shape: {features.shape}")

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --- 3. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42, stratify=labels)

# --- 4. Train SVM Classifier ---
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)
print("\nSVM model trained successfully.")

# --- 5. Model Evaluation ---
# Predict on test set
y_pred = svm.predict(X_test)

# Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# --- 6. Interactive Prediction ---
def interactive_prediction(svm_model, scaler, base_model, img_size, class_names):
    while True:
        img_path = input("\nEnter the image path (or type 'exit' to quit): ").strip()
        print(f"Received path: '{img_path}'")

        if img_path.lower() == 'exit':
            print("Exiting the program.")
            break

        if not os.path.isfile(img_path):
            print("Error: File not found. Please enter a valid image path.")
            continue

        try:
            # Load and preprocess the image
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Extract features
            feature = base_model.predict(img_array)
            feature = scaler.transform(feature)

            # Predict the class
            prediction = svm_model.predict(feature)
            predicted_class = class_names[prediction[0]]
            confidence = np.max(svm_model.predict_proba(feature)) * 100

            print(f'\nPredicted class: {predicted_class} ({confidence:.2f}% confidence)\n')

        except Exception as e:
            print(f"An error occurred: {e}")

# Call the interactive prediction function
interactive_prediction(svm, scaler, base_model, img_size, class_names)
