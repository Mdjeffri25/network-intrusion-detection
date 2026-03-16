# network-intrusion-detection
Deep Learning based Network Intrusion Detection System (NIDS) that analyzes network traffic features and classifies connections as normal or malicious using a neural network model. The system includes a Streamlit dashboard for real-time monitoring and prediction visualization.
Network Intrusion Detection System using Deep Learning

This project implements a Network Intrusion Detection System (NIDS) using Deep Learning techniques to identify malicious activities in network traffic. The system analyzes multiple network connection features and classifies traffic as normal or anomalous.

The project also provides an interactive Streamlit dashboard where users can input network parameters and view prediction results with confidence scores.

Key Features

Deep Learning based intrusion detection

Real-time prediction interface using Streamlit

Analysis of 41 network traffic features

Detection of abnormal network behavior

Visualization of prediction probabilities

Historical prediction tracking

Technologies Used

Python

TensorFlow / Keras

Streamlit

Scikit-learn

NumPy

Pandas

Plotly

Model Architecture

The system uses a Deep Neural Network with the following structure:

Input Layer: 41 features

Hidden Layer 1: 128 neurons (ReLU + Dropout)

Hidden Layer 2: 64 neurons (ReLU + Dropout)

Hidden Layer 3: 32 neurons (ReLU)

Output Layer: Softmax classification

The model is trained to detect normal traffic and intrusion attempts.

Dataset

The model is trained using the KDD Cup 99 network intrusion detection dataset, which contains labeled network traffic data including multiple types of cyber attacks.

Project Structure
frontend.py              → Streamlit application interface
dl_model.h5              → Trained deep learning model
scaler.pkl               → Feature scaling model
label_encoders.pkl       → Encoders for categorical features
target_encoder.pkl       → Target label encoder
requirements.txt         → Required Python libraries
How to Run the Project

Install dependencies

pip install -r requirements.txt

Run the Streamlit application

streamlit run frontend.py

Open browser

http://localhost:8501
Output

The system predicts whether network traffic is:

Normal

Intrusion / Attack

and displays the prediction with confidence percentage and probability distribution.

Future Improvements

Real-time packet capture integration

Advanced deep learning models

Cloud-based deployment

Integration with enterprise security monitoring systems
