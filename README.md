# PRODIGY_ML_04
# ✋ Hand Gesture Recognition using CNN (LeapGestRecog Dataset)

This is Task 4 of my virtual internship at **Prodigy InfoTech** under the Machine Learning track.  
In this project, I implemented a **Convolutional Neural Network (CNN)** to classify hand gestures using the **LeapGestRecog** dataset, which was imported directly into Google Colab using **kagglehub** without downloading it locally.

---

## 🧠 Task Summary

- **Goal**: Build a deep learning model to recognize different hand gestures from image data.
- **Dataset**: [LeapGestRecog Dataset - Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- **Import Method**: Downloaded directly in Google Colab using `kagglehub.dataset_download("gti-upm/leapgestrecog")`
- **Model**: Compact CNN with data augmentation
- **Evaluation Metrics**: Accuracy, Classification Report, Confusion Matrix

---

## 📌 Features Used

- Input image size: **128×128×3**
- Data augmentation: random flip, rotation, and zoom
- Convolutional layers with ReLU activation
- MaxPooling layers for spatial downsampling
- Dropout for regularization
- Dense layer with softmax activation for multi-class classification

---

## 📈 Workflow

1. **Dataset Import**  
   - Used `kagglehub` in Google Colab to download the dataset directly from Kaggle.
   - Automatically scanned all subfolders for image files and inferred labels from folder names.
2. **Data Preparation**  
   - Stratified split into train, validation, and test sets.
   - Preprocessing pipeline using `tf.data` for efficient loading and batching.
3. **Model Training**  
   - Built and trained a compact CNN with data augmentation.
   - Early stopping and learning rate reduction callbacks to optimize training.
4. **Model Evaluation**  
   - Generated classification report and confusion matrix.
   - Saved the trained model (`.keras`) and class label map (`.csv`).

---

## 🖼️ Sample Output

- Model accuracy on the test set.
- Classification report with precision, recall, and F1-score for each gesture class.
- Confusion matrix visualizing model performance across classes.

---

## 🚀 Technologies Used

- Python
- TensorFlow / Keras
- scikit-learn
- kagglehub
- NumPy, Pandas, Matplotlib, Seaborn

---
## Results

<img width="496" height="712" alt="image" src="https://github.com/user-attachments/assets/4802a11f-eddc-47f7-b186-3cce3601ac96" />

---

## 📂 Files

- `task4_hand_gesture_recognition.ipynb` — full Google Colab notebook
- `README.md` — project documentation

---
