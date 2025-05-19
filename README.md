# Deep Learning Models for Collaborative Filtering

This project is an implementation of two deep neural network (DNN) models — **CNN-based** and **RNN-based** — for collaborative filtering, evaluated on the **MovieLens** dataset. It is based on a survey of various deep learning architectures for recommender systems.

## 📚 Overview

The goal of this assignment is to explore and compare different deep neural network architectures applied to collaborative filtering (CF) tasks in recommendation systems. This project implements:

- A **Convolutional Neural Network (CNN)** model
- A **Recurrent Neural Network (RNN)** model

Both models are trained and evaluated on the MovieLens dataset using **RMSE** (Root Mean Square Error) and **MAE** (Mean Absolute Error) as evaluation metrics.

## 🗃 Dataset

- **Dataset Used:** [MovieLens 100k](https://grouplens.org/datasets/movielens/)
- **Preprocessing:**  
  - User and Movie IDs are label encoded into integer indices.
  - Only the relevant columns `UserID`, `MovieID`, and `Rating` are used.
  - The dataset is split into 80% training and 20% testing.

## 🧠 Models

### 1. CNN Model for Collaborative Filtering
- Learns user and item embeddings.
- Concatenates embeddings and applies a 1D convolution layer.
- Followed by dense layers to predict the rating.

### 2. RNN Model for Collaborative Filtering
- Learns user and item embeddings.
- Treats the embeddings as a sequence.
- Uses a SimpleRNN layer followed by dense layers.

## 📈 Evaluation Metrics

- **Root Mean Square Error (RMSE)**
- **Mean Absolute Error (MAE)**

These metrics are computed on the test set after training.

## 📊 Results

| Model | RMSE   | MAE    |
|-------|--------|--------|
| CNN   | ~0.879 | ~0.671 |
| RNN   | ~0.884 | ~0.671 |

> *Note:* Exact values may vary depending on the random initialization and training environment.

## 📦 Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Simba256/deep-cf-models.git
   cd deep-cf-models
