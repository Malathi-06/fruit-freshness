import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.utils import load_img, img_to_array

def predict(image_path):
    model = tf.keras.models.load_model("models/best_model.h5")
    with open("models/class_names.json") as f:
        class_names = json.load(f)

    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)

    preds = model.predict(img_array)
    label = class_names[np.argmax(preds)]
    conf = np.max(preds)
    print(f"Prediction: {label} ({conf*100:.2f}%)")

if __name__ == "__main__":
    predict(r"data/test/rottenbanana/vertical_flip_Screen Shot 2018-06-12 at 9.25.41 PM.png")