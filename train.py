import os, json
import tensorflow as tf
from src.config import IMG_SIZE, BATCH_SIZE, EPOCHS, MODEL_DIR
from src.model_builder import build_model

def train_model():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "data/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "data/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    
    class_names = train_ds.class_names
    model = build_model(num_classes=len(class_names))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"{MODEL_DIR}/best_model.h5", save_best_only=True, monitor="val_accuracy")
    
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=EPOCHS, callbacks=[checkpoint])
    
    with open(f"{MODEL_DIR}/class_names.json", "w") as f:
        json.dump(class_names, f)
    
    print("✅ Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()