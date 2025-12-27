import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from datetime import datetime


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)


DATASET_DIR = "dataset"
MODEL_SAVE_PATH = "emotion_model_mobilenet.h5"
HISTORY_PLOT_PATH = "training_history_mobilenet.png"
IMG_SIZE = 224
BATCH_SIZE = 16  
EPOCHS = 50
INITIAL_LR = 0.001  

EMOTIONS = ['Angry', 'Happy', 'Neutral', 'Sad', 'Fear']

print("🚀 Emotion Recognition Training - Optimized Version")
print(f"Backbone: MobileNetV2 (memory-efficient)")
print(f"Dataset: {DATASET_DIR}")
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}\n")


def load_dataset():
    X = []
    y = []
    
    for idx, emotion in enumerate(EMOTIONS):
        emotion_path = os.path.join(DATASET_DIR, emotion)
        if not os.path.exists(emotion_path):
            print(f"⚠️ Warning: {emotion_path} not found!")
            continue
            
        images = os.listdir(emotion_path)
        print(f"Loading {emotion}: {len(images)} images")
        
        for img_name in images:
            img_path = os.path.join(emotion_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                y.append(idx)
            except Exception as e:
                continue
    
    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y)
    
    print(f"\n✅ Loaded {len(X)} images")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y

def build_model():
    """
    MobileNetV2-based model - memory efficient and stable
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
   
    base_model.trainable = False
    
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(len(EMOTIONS), activation='softmax')
    ])
    
    return model, base_model


def main():
   
    X, y = load_dataset()
    
    if len(X) == 0:
        print("❌ No data loaded. Please check your dataset!")
        return
    
   
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    
   
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.15,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
   
    print(f"\n🏗️  Building MobileNetV2 model...")
    model, base_model = build_model()
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print("\n🔥 Training phase 1: Frozen base model...")
    start_time = datetime.now()
    
    history_phase1 = model.fit(
        train_generator,
        epochs=25,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tune: unfreeze last layers
    print("\n🔥 Training phase 2: Fine-tuning...")
    base_model.trainable = True
    
    # Freeze all except last 20 layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR/10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_phase2 = model.fit(
        train_generator,
        epochs=EPOCHS-25,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    duration = datetime.now() - start_time
    print(f"\n✅ Training completed in {duration}")
    
    # Combine histories
    history = {
        'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
        'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
        'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
        'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
    }
    
    # Clear memory and reload best model
    del model
    tf.keras.backend.clear_session()
    
    print("\n📈 Loading best model for evaluation...")
    model = keras.models.load_model(MODEL_SAVE_PATH)
    
    # Evaluate
    print("Evaluating on validation set...")
    val_loss, val_acc = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0)
    
    print(f"\n🎯 Best Model Performance:")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Plot training history
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    plt.axvline(x=25, color='r', linestyle='--', alpha=0.5, label='Fine-tuning starts')
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.axvline(x=25, color='r', linestyle='--', alpha=0.5, label='Fine-tuning starts')
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH, dpi=150)
    print(f"\n📊 Training history saved to {HISTORY_PLOT_PATH}")
    
    return model, history

if __name__ == "__main__":
    try:
        model, history = main()
        print("\n🎉 Training completed successfully!")
        print("\n💡 Model Details:")
        print("   - Architecture: MobileNetV2 (memory-efficient)")
        print("   - Expected accuracy: 70-85% (depends on data quality)")
        print("   - Model saved to: emotion_model_mobilenet.h5")
        print("\n📝 Next Steps:")
        print("   1. Check training_history_mobilenet.png for learning curves")
        print("   2. Update your live detection script to use 'emotion_model_mobilenet.h5'")
        print("   3. If accuracy is low, consider collecting more diverse training data")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise