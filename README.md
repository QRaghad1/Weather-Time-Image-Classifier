#  Travel Image Classification (Weather & Time of Day)

##  Overview

This project focuses on **multi-task image classification** using travel images.
The goal is to predict:

*  **Weather Condition** → (Cloudy, NotClear, Rainy, Snowy, Sunny)
*  **Time of Day** → (Morning, Afternoon, Evening)

The system compares **classical machine learning models** and **deep learning approaches** to understand performance limits and challenges.

---

##  Models Implemented

### 1. K-Nearest Neighbors (KNN)

* Baseline model
* Tested with different K values (K=1, K=3)

### 2. Support Vector Machine (SVM)

* Kernel: RBF
* Tuned parameter: `C = {0.1, 1, 10, 100}`
* Multi-output classification using `MultiOutputClassifier`

### 3. Convolutional Neural Network (CNN)

* Based on **MobileNetV2 (Transfer Learning)**
* Frozen pretrained layers
* Custom classification head:

  * Dense (512)
  * Dropout (0.4)
  * Two outputs (weather + time)

---

##  Dataset

* Total Images: **904**
* Weather Classes: **5**
* Time Classes: **3**
* Image naming format:

  ```
  Weather_Time_index.jpg
  Example: Sunny_Morning_001.jpg
  ```

---

##  Data Preprocessing

### ✔ Steps:

1. Clean CSV file (keep relevant columns only)
2. Validate labels:

   * Weather ∈ {Sunny, Rainy, Cloudy, Snowy, NotClear}
   * Time ∈ {Morning, Afternoon, Evening}
3. Download and store images locally
4. Resize images:

   * SVM → 64×64
   * CNN → 224×224
5. Normalize pixel values (0 → 1)
6. Encode labels using `LabelEncoder`

---

## Exploratory Data Analysis (EDA)

###  Class Distribution

* Severe imbalance in weather:

  * Sunny: **55%**
  * Rainy: **2.7%**
* Time is more balanced:

  * Afternoon: **46.9%**

###  PCA Insights

* Weather classes → **high overlap (hard to separate)**
* Time classes → **better separability**

  * Evening is easiest to classify

---

##  Evaluation Metrics

We used:

* Accuracy
* Precision
* Recall
* F1-Score
* Macro & Weighted Averages

---

##  Results Summary

| Model      | Weather Accuracy | Time Accuracy | F1 Score |
| ---------- | ---------------- | ------------- | -------- |
| KNN (K=1)  | 59.1%            | 64.1%         | ~0.54    |
| SVM (C=10) | 59.1%            | **63.5%**     | ~0.58    |
| CNN        | 59–61%           | 56–61%        | ~0.56    |

---

##  Key Findings

* All models plateau around **~60% accuracy**
* Severe class imbalance limits performance
* Rainy class almost never detected (F1 ≈ 0.00)

###  Important Insight:

> The main limitation is **data quality**, not model complexity.

---

##  Challenges

* Class imbalance (Sunny vs Rainy ≈ 14:1)
* Visual similarity between classes
* Weak separation in feature space
* Limited dataset size

---

## Attempted Improvements

| Method            | Result                 |
| ----------------- | ---------------------- |
| Data Augmentation | Increased noise        |
| Fine-Tuning CNN   | Overfitting            |
| Sky Segmentation  | Removed useful context |

---

##  Final Recommendation

 **Best Model: SVM (C = 10)**

* Balanced performance
* Faster training
* Much lower computational cost than CNN

---

##  How to Run

### 1. Install dependencies


pip install numpy opencv-python scikit-learn tensorflow matplotlib seaborn


### 2. Dataset

Place images inside:


travel_images/


### 3. Run models

* SVM:


python svm_model.py


* CNN:


python cnn_model.py


* EDA:

python eda_analysis.py


---

## Conclusion

This project demonstrates that:

* More complex models ≠ better performance
* Data quality and balance are critical
* Classical ML (SVM) can outperform deep learning on small datasets

---

##  Author

* Raghad 
---
