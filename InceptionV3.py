import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import os

# --- 1. Data Preprocessing ---
train_dir = "C:/Users/tring/Desktop/Image Processing/skin-disease-datasaet/train_set"
test_dir = "C:/Users/tring/Desktop/Image Processing/skin-disease-datasaet/test_set"
img_size = (299, 299)  
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# --- 2. Model Creation (InceptionV3) ---
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 3. Model Training ---
num_epochs = 10
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator
)

# --- 4. Model Evaluation ---
val_generator.reset()
predictions = model.predict(val_generator)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = val_generator.classes
class_names = list(val_generator.class_indices.keys())

# Classification Report
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - InceptionV3')
plt.show()

# --- 5. Save the Model ---
model.save('skin_disease_detector_inceptionv3.h5')
print("\nModel saved as 'skin_disease_detector_inceptionv3.h5'.")

# --- 6. Interactive Prediction ---
def interactive_prediction(model_path, img_size, class_names):
    # Load the saved model
    model = load_model(model_path)

    # Interactive loop
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
            img = image.load_img(img_path, target_size=img_size)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict the class
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            print(f'\nPredicted class: {predicted_class} ({confidence:.2f}% confidence)\n')

        except Exception as e:
            print(f"An error occurred: {e}")

# Call the interactive prediction function
interactive_prediction('skin_disease_detector_inceptionv3.h5', img_size, class_names)
