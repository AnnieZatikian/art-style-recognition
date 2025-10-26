import os
import pickle
from src.data_loader import load_artists_data, load_art_styles_data
from src.preprocessing import load_and_preprocess_images, split_data
from src.model import create_cnn_model
from sklearn.preprocessing import LabelEncoder

def main():
    # Paths
    artists_csv_path = 'data/artists.csv'
    art_styles_csv_path = 'data/art_style.csv'
    image_folder = 'data/images'  # Folder where images are stored

    # Load data
    print("[INFO] Loading CSV files...")
    artists_df = load_artists_data(artists_csv_path)
    styles_df = load_art_styles_data(art_styles_csv_path)

    # Load and preprocess images
    print("[INFO] Loading and preprocessing images...")
    X, y_labels = load_and_preprocess_images(image_folder)

    # Encode labels from image folder names
    print("[INFO] Encoding labels from folder names...")
    label_encoder = LabelEncoder()
    label_encoder.fit(y_labels)
    y = label_encoder.transform(y_labels)

    # Split data
    print("[INFO] Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = split_data(X, y)

    # Create model
    print("[INFO] Creating model...")
    num_classes = len(label_encoder.classes_)
    model = create_cnn_model(num_classes=num_classes)

    # Train model
    print("[INFO] Training model...")
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=10,
                        batch_size=32)

    # Save model
    print("[INFO] Saving model...")
    model.save('models/art_style_classifier.h5')

    # Save label encoder
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    print("[INFO] Training complete and model saved!")

if __name__ == "__main__":
    main()
