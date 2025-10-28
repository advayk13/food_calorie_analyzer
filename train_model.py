# type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import shutil
import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 
#  Dataset folder and balancing
# 
dataset_dir = 'Dataset'  # Original dataset
LIMIT_PER_CLASS = 300  # Max images per class

balanced_dataset_dir = 'BalancedDataset'
os.makedirs(balanced_dataset_dir, exist_ok=True)

for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    selected_images = random.sample(images, min(len(images), LIMIT_PER_CLASS))

    os.makedirs(os.path.join(balanced_dataset_dir, class_name), exist_ok=True)
    for img_name in selected_images:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(balanced_dataset_dir, class_name, img_name)
        if not os.path.exists(dst):
            shutil.copy(os.path.abspath(src), dst)

print("✅ Dataset balanced and copied to", balanced_dataset_dir)

# 
#  Data generators with augmentation
# 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    balanced_dataset_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    balanced_dataset_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# -
#  Compute class weights (fixed for new scikit-learn)
# 
labels = train_generator.classes

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

# 
#  Build improved CNN model
# 
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 
#  Callbacks: EarlyStopping + ModelCheckpoint
# 
checkpoint = ModelCheckpoint(
    "best_cnn_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1
)
early_stop = EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
)

# 
#  Train the model
# 
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    class_weight=class_weights_dict,
    callbacks=[checkpoint, early_stop]
)

print("✅ Training complete. Best model saved as 'best_cnn_model.h5'")
