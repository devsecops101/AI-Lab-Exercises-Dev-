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

def train_model(features_path='logs/web_features.csv', model_save_path='model'):
    """
    We're going to teach our computer friend how to spot when someone is being naughty on our website!
    Like teaching a puppy to know the difference between good visitors and bad visitors.
    """
    # First, let's read our big book of website visitors
    df = pd.read_csv(features_path)

    # We need to split our book into two parts:
    # One part tells us about how visitors behave (like what they do on our website)
    # The other part tells us if they were good or bad visitors
    X = df.drop('label', axis=1)
    y = df['label']

    # Now let's make two piles of stories:
    # One pile to learn from, like reading bedtime stories
    # And one pile to test if we learned the lesson well!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # We need to make all our numbers the same size
    # Like making sure all our crayons are the same length!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Let's save our crayon measurements for later
    os.makedirs(model_save_path, exist_ok=True)
    joblib.dump(scaler, f'{model_save_path}/scaler.pkl')

    # Now we're building our computer's thinking hat!
    # It's like stacking blocks to make a tower of smartness
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),  # Sometimes we close our eyes to think harder!
        Dense(32, activation='relu'),
        Dropout(0.2),  # Close our eyes again to remember better!
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Teaching our computer friend how to learn
    # Like explaining the rules of a fun game
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # If our computer friend gets tired of learning,
    # We'll let them take a little nap
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Time to practice! Like playing the same game many times
    # until we get really good at it
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # Let's see how many gold stars our computer friend earned!
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    print(f'\nOur computer friend got {accuracy:.4f} stars out of 1 star!')

    # Save our computer friend's new smart brain
    # Like taking a picture of our favorite drawing
    model.save(f'{model_save_path}/detection_model.keras')

    # Time to draw pictures of how well we learned!
    plt.figure(figsize=(12, 4))

    # First picture shows how many right answers we got
    # Like counting how many times we caught the ball
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('How many right answers we got')
    plt.ylabel('Right answers')
    plt.xlabel('Times we practiced')
    plt.legend(['Practice time', 'Test time'], loc='upper left')

    # Second picture shows how many oopsies we made
    # Like counting how many times we dropped the ball
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('How many oopsies we made')
    plt.ylabel('Oopsies')
    plt.xlabel('Times we practiced')
    plt.legend(['Practice time', 'Test time'], loc='upper left')

    plt.tight_layout()
    
    # Write down how well we did in our special diary
    with open(f'{model_save_path}/training_results.txt', 'w') as f:
        f.write("Our Computer Friend's Report Card!\n")
        f.write("============================\n\n")
        f.write(f"Gold Stars Earned: {accuracy:.4f} out of 1 star\n")
        f.write(f"Oopsies Made: {loss:.4f}\n\n")
        f.write("How We Did During Practice:\n")
        f.write("-------------------------\n")
        f.write("Final Practice Score: {:.4f}\n".format(history.history['accuracy'][-1]))
        f.write("Final Test Score: {:.4f}\n".format(history.history['val_accuracy'][-1]))
        f.write("Practice Oopsies: {:.4f}\n".format(history.history['loss'][-1]))
        f.write("Test Oopsies: {:.4f}\n".format(history.history['val_loss'][-1]))

    return model, scaler

if __name__ == "__main__":
    print("Time to teach our computer friend a new game!")
    print("We're using TensorFlow version:", tf.__version__)
    
    # Let's start learning!
    model, scaler = train_model()
    print("Yay! Our computer friend learned how to spot naughty visitors! All the results are saved in the 'model' toy box!")