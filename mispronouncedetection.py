import tkinter as tk
from tkinter import messagebox
from pydub import AudioSegment
import speech_recognition as sr
from langdetect import detect
from gtts import gTTS
import os
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to train machine learning models
def train_models(X, y):
    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training Random Forest Classifier
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    
    # Training Support Vector Machine Classifier
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    
    # Evaluating models
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
    
    messagebox.showinfo("Training Complete", f"Random Forest Accuracy: {rf_accuracy}\nSVM Accuracy: {svm_accuracy}")

# Function to play correction message
def play_correction_message(correction):
    correction_audio = AudioSegment.from_file("correction.mp3")
    correction_audio = correction_audio.set_frame_rate(22050)  # Set frame rate to match correction.mp3
    correction_audio = correction_audio.set_channels(1)  # Set channels to mono
    correction_audio = correction_audio + correction
    correction_audio.export("temp.mp3", format="mp3")
    temp_audio = AudioSegment.from_file("temp.mp3")
    temp_audio.play()
    os.remove("temp.mp3")

# Function to start voice communication and detect mispronunciation
def start_communication(X, y):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            # Increase timeout if necessary
            audio = recognizer.listen(source, timeout=10)  # 10 seconds timeout
        except sr.WaitTimeoutError:
            messagebox.showerror("Error", "Listening timed out. Please try again.")
            return

    try:
        print("Processing...")
        recognized_text = recognizer.recognize_google(audio)
        print(f"Recognized Text: {recognized_text}")  # Debugging line
        language = detect(recognized_text)
        
        if language == "en":
            if "mispronunciation" in recognized_text.lower():
                correction_message = "Please say 'mispronunciation' correctly."
            else:
                correction_message = "Your pronunciation is correct."
        else:
            correction_message = "Sorry, I can only correct mispronunciations in English."

        tts = gTTS(text=correction_message, lang='en', slow=False)
        tts.save("correction.mp3")
        play_correction_message()

    except sr.UnknownValueError:
        messagebox.showerror("Error", "Sorry, I could not understand what you said.")
    except sr.RequestError:
        messagebox.showerror("Error", "Could not request results. Please check your internet connection.")
    except Exception as e:
        print(f"An error occurred: {e}")  # Debugging line

# Function to greet and start communication
def greet_and_start_communication():
    current_time = datetime.datetime.now().strftime("%H:%M")
    if int(current_time.split(":")[0]) < 12:
        message = "Good morning! Let's start the communication. Please click the 'Start Communication' button."
    elif int(current_time.split(":")[0]) < 18:
        message = "Good afternoon! Let's start the communication. Please click the 'Start Communication' button."
    else:
        message = "Good evening! Let's start the communication. Please click the 'Start Communication' button."

    # Generate the greeting message using gTTS
    tts = gTTS(text=message, lang='en', slow=False)
    tts.save("greeting.mp3")

    # Play the greeting message using pydub
    greeting_audio = AudioSegment.from_file("greeting.mp3")
    greeting_audio.play()

    # Remove the greeting.mp3 file after playing
    os.remove("greeting.mp3")

# GUI setup
root = tk.Tk()
root.title("Mispronunciation Detection")
root.geometry("400x200")
root.configure(bg='purple')

label = tk.Label(root, text="Let's start!!", font=('Arial', 16), bg='purple', fg='white')
label.pack(pady=20)

# Dummy data for demonstration
# Replace X and y with your actual dataset
X = np.random.rand(100, 10)
y = np.random.randint(2, size=100)

train_button = tk.Button(root, text="Train Models", command=lambda: train_models(X, y), font=('Arial', 14), bg='white', fg='purple')
train_button.pack(pady=5)

start_button = tk.Button(root, text="Start Communication", command=lambda: start_communication(X, y), font=('Arial', 14), bg='white', fg='purple')
start_button.pack(pady=5)

# Play the greeting message as soon as the GUI is displayed
greet_and_start_communication()

root.mainloop()
