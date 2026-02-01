import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, applications, callbacks
from sklearn.metrics import confusion_matrix, classification_report
import sys

# --- CONFIGURATION ---
DATASET_DIR = "."  # Current directory contains Training and Testing folders
IMG_SIZE = (224, 224) # MobileNetV2 prefers 224x224
BATCH_SIZE = 32
EPOCHS = 20
MODEL_SAVE_PATH = 'brain_tumor_model.h5'

def train_model():
    # Paths
    train_dir = os.path.join(DATASET_DIR, "Training")
    val_dir = os.path.join(DATASET_DIR, "Testing")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"❌ Error: 'Training' or 'Testing' folders not found in {os.getcwd()}")
        sys.exit(1)

    print(f"📂 Training data: {train_dir}")
    print(f"📂 Testing data: {val_dir}")

    # Data Augmentation (Aggressive for better generalization)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    print("⏳ Loading data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Save class indices for app.py
    print(f"✅ Class Indices: {train_generator.class_indices}")
    # Write class names to a file so we can read it later
    with open("class_names.txt", "w") as f:
        f.write(str(list(train_generator.class_indices.keys())))

    # --- TRANSFER LEARNING: MobileNetV2 ---
    print("🧠 Building MobileNetV2 model...")
    base_model = applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Fine-tuning: Freeze mostly, but allow top layers if needed. 
    # For now, freezing all is safer for small datasets.
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),  # Higher dropout to prevent overfitting
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    print("🚀 Starting training...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr]
    )

    print("💾 Saving model...")
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

    # Evaluate
    print("📊 Evaluating on Test Set...")
    loss, accuracy = model.evaluate(validation_generator)
    print(f"🏆 Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    train_model()
