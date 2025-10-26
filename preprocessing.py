import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def encode_labels(styles_df):
    """
    Encode style names to numerical labels.
    """
    le = LabelEncoder()
    styles_df['label'] = le.fit_transform(styles_df['style'])
    return styles_df, le

def load_and_preprocess_images(image_folder, image_size=(224, 224)):
    """
    Load images from a folder and preprocess (resize and normalize).
    """
    images = []
    labels = []
    
    for class_name in os.listdir(image_folder):
        class_folder = os.path.join(image_folder, class_name)
        if os.path.isdir(class_folder):
            for img_file in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    img = img / 255.0  # Normalize pixels to [0,1]
                    images.append(img)
                    labels.append(class_name)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split into train and validation sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
