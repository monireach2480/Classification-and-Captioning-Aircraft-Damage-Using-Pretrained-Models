# Aircraft Damage Classification and Captioning

Final project for classifying and captioning aircraft damage using pretrained deep learning models.

## Table of Contents
- [Introduction](#introduction)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Part 1: Aircraft Damage Classification](#part-1-aircraft-damage-classification)
  - [Part 2: Image Captioning with BLIP](#part-2-image-captioning-with-blip)
- [Results](#results)
- [License](#license)

## Introduction

Aircraft damage detection is essential for maintaining the safety and longevity of aircraft. Traditional manual inspection methods are time-consuming and prone to human error. This project aims to automate the classification of aircraft damage into two categories: **"dent"** and **"crack"** using feature extraction with a pre-trained VGG16 model. Additionally, a pre-trained Transformer model (BLIP) is used to generate captions and summaries for the images.

## Objectives

- Use the VGG16 model for image classification
- Prepare and preprocess image data for machine learning tasks
- Evaluate the model's performance using appropriate metrics
- Visualize model predictions on test data
- Implement a custom Keras layer
- Generate captions and summaries for images using the BLIP pretrained model

## Dataset

The project uses the Aircraft dataset, originally sourced from [Roboflow Aircraft Dataset](https://roboflow.com) (License: CC BY 4.0).

The dataset is organized into `train`, `valid`, and `test` directories, each containing `dent` and `crack` subdirectories with images.

## Project Structure
```
aircraft_damage_dataset_v1/
├── train/
│   ├── dent/
│   └── crack/
├── valid/
│   ├── dent/
│   └── crack/
└── test/
    ├── dent/
    └── crack/
```

## Installation

### Required Libraries

Install the required libraries using pip:
```bash
pip install --upgrade ml-dtypes
pip install pandas==2.2.3
pip install tensorflow==2.17.1
pip install pillow==11.1.0
pip install matplotlib==3.9.2
pip install transformers==4.38.2
pip install torch keras
```

### Environment Setup

1. **Suppress TensorFlow warnings** (optional):
```python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

2. **Import necessary libraries**:
```python
import warnings
warnings.filterwarnings('ignore')
import zipfile
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import random
import tarfile
import urllib.request
import shutil
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
```

3. **Set seed for reproducibility**:
```python
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
```

4. **Download and extract dataset**:
```python
batch_size = 32
n_epochs = 5
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"
tar_filename = "aircraft_damage_dataset_v1.tar"
extracted_folder = "aircraft_damage_dataset_v1"

urllib.request.urlretrieve(url, tar_filename)

if os.path.exists(extracted_folder):
    shutil.rmtree(extracted_folder)

with tarfile.open(tar_filename, "r") as tar_ref:
    tar_ref.extractall()

extract_path = "aircraft_damage_dataset_v1"
train_dir = os.path.join(extract_path, 'train')
test_dir = os.path.join(extract_path, 'test')
valid_dir = os.path.join(extract_path, 'valid')
```

## Usage

### Part 1: Aircraft Damage Classification

#### 1. Data Preprocessing

Create ImageDataGenerator instances:
```python
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
```

Create data generators:
```python
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    seed=seed_value,
    class_mode='binary',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    seed=seed_value,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    seed=seed_value,
    class_mode='binary',
    shuffle=False
)
```

#### 2. Model Definition

Load and configure VGG16:
```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

output = base_model.layers[-1].output
output = keras.layers.Flatten()(output)
base_model = Model(base_model.input, output)

for layer in base_model.layers:
    layer.trainable = False
```

Build classifier:
```python
model = Sequential()
model.add(base_model)
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
```

#### 3. Model Training
```python
history = model.fit(train_generator, epochs=n_epochs, validation_data=valid_generator)
train_history = model.history.history
```

#### 4. Visualizing Training Results

Plot training and validation loss:
```python
plt.title("Training Loss")
plt.ylabel("Loss")
plt.xlabel('Epoch')
plt.plot(train_history['loss'])
plt.show()

plt.title("Validation Loss")
plt.ylabel("Loss")
plt.xlabel('Epoch')
plt.plot(train_history['val_loss'])
plt.show()
```

Plot training and validation accuracy:
```python
plt.title("Training Accuracy")
plt.ylabel("Accuracy")
plt.xlabel('Epochs')
plt.plot(train_history['accuracy'])
plt.show()

plt.title("Validation Accuracy")
plt.ylabel("Accuracy")
plt.xlabel('Epochs')
plt.plot(train_history['val_accuracy'])
plt.show()
```

#### 5. Model Evaluation
```python
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
```

#### 6. Visualizing Predictions

Helper functions:
```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

def plot_image_with_title(image, model, true_label, predicted_label, class_names):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    true_label_name = class_names[true_label]
    pred_label_name = class_names[predicted_label]
    plt.title(f"True: {true_label_name}\nPred: {pred_label_name}")
    plt.axis('off')
    plt.show()

def test_model_on_image(test_generator, model, index_to_plot=0):
    test_images, test_labels = next(test_generator)
    predictions = model.predict(test_images)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    class_indices = test_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    image_to_plot = test_images[index_to_plot]
    true_label = test_labels[index_to_plot]
    predicted_label = predicted_classes[index_to_plot]
    plot_image_with_title(image=image_to_plot, model=model, true_label=true_label, predicted_label=predicted_label, class_names=class_names)
```

Visualize predictions:
```python
test_model_on_image(test_generator, model, 1)
```

### Part 2: Image Captioning with BLIP

#### 1. Loading BLIP Model
```python
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
```

#### 2. Custom Keras Layer
```python
class BlipCaptionSummaryLayer(tf.keras.layers.Layer):
    def __init__(self, processor, model, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.model = model

    def call(self, image_path, task):
        return tf.py_function(self.process_image, [image_path, task], tf.string)

    def process_image(self, image_path, task):
        try:
            image_path_str = image_path.numpy().decode("utf-8")
            image = Image.open(image_path_str).convert("RGB")
            if task.numpy().decode("utf-8") == "caption":
                prompt = "This is a picture of"
            else:
                prompt = "This is a detailed photo showing"
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            output = self.model.generate(**inputs)
            result = self.processor.decode(output[0], skip_special_tokens=True)
            return result
        except Exception as e:
            print(f"Error: {e}")
            return "Error processing image"
```

#### 3. Helper Function
```python
def generate_text(image_path, task):
    blip_layer = BlipCaptionSummaryLayer(processor, model)
    return blip_layer(image_path, task)
```

#### 4. Generating Captions and Summaries

Example usage:
```python
image_path = tf.constant("aircraft_damage_dataset_v1/test/dent/144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg")

caption = generate_text(image_path, tf.constant("caption"))
print("Caption:", caption.numpy().decode("utf-8"))

summary = generate_text(image_path, tf.constant("summary"))
print("Summary:", summary.numpy().decode("utf-8"))
```

Additional example:
```python
image_url = "aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg"
img = plt.imread(image_url)
plt.imshow(img)
plt.axis('off')
plt.show()

image_path = tf.constant("aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg")

caption = generate_text(image_path, tf.constant("caption"))
print("Caption:", caption.numpy().decode("utf-8"))

summary = generate_text(image_path, tf.constant("summary"))
print("Summary:", summary.numpy().decode("utf-8"))
```

## Results

- **Classification Model**: Achieved a test accuracy of **84.38%**
- **BLIP Model**: Successfully generates captions and summaries for aircraft damage images

## License

This project uses the Aircraft dataset from Roboflow, licensed under CC BY 4.0.