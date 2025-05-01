#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

def train_model(features_path, model_save_path='model'):
    # Let's teach our computer friend to spot bad things in our firewall!
    # First, let's get our learning cards ready
    df = pd.read_csv(features_path)

    # Split our cards into two piles: what we want to look at and what answer we expect
    X = df.drop('label', axis=1)
    y = df['label']

    # Let's make two groups: one pile to learn from and one pile to test if we learned well
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Make all our numbers the same size, like making sure all our toys fit in the same box
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save our toy box size so we can use it again later
    os.makedirs(model_save_path, exist_ok=True)
    joblib.dump(scaler, f'{model_save_path}/scaler.pkl')

    # Build our learning brain - it's like stacking building blocks!
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),  # Sometimes we take away some blocks to make our brain stronger
        Dense(32, activation='relu'),
        Dropout(0.2),  # Taking away more blocks!
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Tell our brain how to learn
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # If our brain isn't getting better at learning, we'll stop and take a break
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Time to learn! Like practicing a new game over and over
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # Let's see how well our brain learned!
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    print(f'\nTest accuracy: {accuracy:.4f}')

    # Save our smart brain for later
    model.save(f'/home/student/labs/bonus/firewall_and_web/model/detection_model.keras')

    # Draw pretty pictures to see how well we learned
    plt.figure(figsize=(12, 4))

    # First picture shows how good we got at the game
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Second picture shows how many mistakes we made
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    
    # Save our results in a special notebook
    with open(f'{model_save_path}/training_results.txt', 'w') as f:
        f.write("Our Computer Friend's Learning Report!\n")
        f.write("=====================================\n\n")
        f.write(f"How good we got at the game: {accuracy:.4f}\n")
        f.write(f"How many mistakes we made: {loss:.4f}\n\n")
        f.write("Training History:\n")
        f.write("----------------\n")
        f.write("Final Practice Score: {:.4f}\n".format(history.history['accuracy'][-1]))
        f.write("Final Test Score: {:.4f}\n".format(history.history['val_accuracy'][-1]))
        f.write("Final Practice Mistakes: {:.4f}\n".format(history.history['loss'][-1]))
        f.write("Final Test Mistakes: {:.4f}\n".format(history.history['val_loss'][-1]))

    return model, scaler

if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    model, scaler = train_model('/home/student/labs/bonus/firewall_and_web/logs/firewall_features.csv')
    print("Model training done, model saved to 'model/detection_model'")