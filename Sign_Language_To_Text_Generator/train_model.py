# train_model.py
# Usage: python train_model.py --csv landmarks.csv --model asl_model.h5

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['label']).values.astype(np.float32)
    y = df['label'].values
    return X, y

def build_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--model", type=str, default="asl_model.h5")
    parser.add_argument("--encoder", type=str, default="label_encoder.joblib")
    args = parser.parse_args()

    X, y_text = load_data(args.csv)
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    joblib.dump(le, args.encoder)
    print("Classes:", le.classes_)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.12, random_state=42, stratify=y)

    model = build_model(X.shape[1], len(le.classes_))
    checkpoint = callbacks.ModelCheckpoint(args.model, save_best_only=True, monitor='val_accuracy', mode='max')
    early = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=100, batch_size=32, callbacks=[checkpoint, early])

    print("Training finished. Model saved to", args.model)
