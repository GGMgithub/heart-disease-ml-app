# Heart Disease Prediction using Machine Learning

---

## 1. Problem Statement
Heart disease is one of the leading causes of mortality worldwide. Early detection using patient clinical attributes can assist healthcare professionals in making informed decisions and initiating preventive care.

The goal of this project is to **predict the presence of heart disease** using supervised machine learning techniques based on patient health parameters.  
This system is intended as a **decision-support tool** and **not a replacement for professional medical diagnosis**.

---

## 2. Dataset Description
The dataset used in this project is the **Heart Disease UCI Dataset**, sourced from [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).

- **Number of instances:** 920  
- **Number of features:** 15
- **Target variable:** `num`  
  - `0` → No heart disease  
  - `1` → Presence of heart disease (all values > 0 converted to 1)

The dataset aggregates patient records from multiple medical centers and contains a mix of numerical and categorical clinical attributes.

### Key Features

| Feature | Description |
|---------|-------------|
| age | Age of the patient |
| sex | Gender (Male/Female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (Yes/No) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0–3) |
| thal | Thalassemia (normal, fixed defect, reversible defect) |

---

## 3. Data Preprocessing Pipeline
To ensure consistency between training and inference, the following preprocessing steps were applied:

1. **Column Removal:** `id` and `dataset` columns were dropped to prevent data leakage and source bias.  
2. **Missing Value Handling:**  
   - Numerical features → Median imputation  
   - Categorical features → Mode imputation  
3. **One-Hot Encoding** for categorical variables.  
4. **Feature Scaling** using **StandardScaler**.  
5. **Binary Classification:** `num > 0` converted to class `1`.  
6. **Feature Alignment:** During inference, uploaded datasets are aligned to the trained model feature set using saved `feature_names.pkl`.

---

## 4. Machine Learning Models Implemented
The following six classification models were implemented using the same dataset and preprocessing pipeline:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

---

## 5. Model Evaluation Metrics
Each model was evaluated on a held-out test dataset using the following metrics:

- **Accuracy** – Overall correctness  
- **AUC Score** – Ability to discriminate between classes  
- **Precision** – True positives / predicted positives  
- **Recall** – True positives / actual positives  
- **F1 Score** – Harmonic mean of precision and recall  
- **Matthews Correlation Coefficient (MCC)** – Robust evaluation in case of class imbalance

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8424 | 0.9032 | 0.8411 | 0.8824 | 0.8612 | 0.6801 |
| Decision Tree | 0.7935 | 0.7886 | 0.8019 | 0.8333 | 0.8173 | 0.5806 |
| KNN | 0.8533 | 0.9077 | 0.8505 | 0.8922 | 0.8708 | 0.7023 |
| Naive Bayes | 0.8424 | 0.8869 | 0.8544 | 0.8627 | 0.8585 | 0.6807 |
| Random Forest | 0.8696 | 0.9211 | 0.8482 | 0.9314 | 0.8879 | 0.7374 |
| XGBoost | 0.8533 | 0.8870 | 0.8378 | 0.9118 | 0.8732 | 0.7033 |
---

## 6. Observations on Model Performance

| Model | Observation |
|-------|-------------|
| Logistic Regression | Provides a strong baseline with good balance between precision and recall. High AUC indicates good discrimination ability. |
| Decision Tree | Simple and interpretable but lower accuracy and MCC indicate some overfitting and weaker generalization. |
| KNN | Performs well after feature scaling, with balanced precision and recall. Effective for distance-based classification. |
| Naive Bayes | Stable performance despite independence assumptions; maintains good precision and recall. |
| Random Forest | Best overall model with highest recall, F1 score, and MCC. Ensemble learning helps capture complex patterns and reduces overfitting. |
| XGBoost | Competitive with Random Forest; slightly lower MCC but high recall shows it captures true positives effectively. |

## Key Insights

1. **Ensemble Models Lead Performance:**  
   - Random Forest achieves the highest overall metrics, particularly **MCC (0.7374)**, **Recall (0.9314)**, and **F1 Score (0.8879)**, making it the most reliable model for detecting heart disease.  
   - XGBoost also performs competitively, with strong recall (**0.9118**) but slightly lower MCC (**0.7033**) compared to Random Forest.

2. **Distance-Based Model Benefits from Scaling:**  
   - KNN performs well after feature scaling, achieving balanced precision and recall (**0.8505** and **0.8922**) with good overall accuracy (**0.8533**).

3. **Simple Models Provide Strong Baselines:**  
   - Logistic Regression and Naive Bayes deliver consistent results with good AUC scores (**0.9032** and **0.8869**) and stable MCC values (~0.68), highlighting their utility for quick and interpretable models.

4. **Single Tree Models Have Limitations:**  
   - Decision Tree has the lowest MCC (**0.5806**) and accuracy (**0.7935**), indicating susceptibility to overfitting and lower generalization compared to ensemble methods.

5. **Feature Preprocessing Matters:**  
   - Proper handling of missing values, one-hot encoding, and scaling is crucial for models like KNN, Logistic Regression, and Naive Bayes to perform optimally.

6. **Recall vs Precision Trade-offs:**  
   - Ensemble models prioritize recall (true positive detection) without significantly sacrificing precision, which is important in healthcare applications where missing positive cases can be critical.

---

## 7. Streamlit Web Application
An interactive **Streamlit web application** was developed to demonstrate real-world usage of the trained models.

### Features
- Upload CSV test dataset  
- Download sample test dataset  
- Sidebar-based model selection  
- Automatic preprocessing aligned with training pipeline  
- KPI-style metric cards for intuitive interpretation  
- Confusion matrix visualization  
- ROC curve visualization  
- Tab-based layout:
  - **Model Performance**  
  - **Benchmarking** (dynamic comparison using session state)  
  - **Data Exploration** (numerical stats, correlation heatmap, categorical counts, target distribution)  

The application ensures **feature consistency** by aligning uploaded data with the trained model feature set.

---

## 8. Project Structure

```text
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- train_and_save_models.py
│
│-- data/
│   ├── heart_disease_test_sample_500.csv
│   └── heart_disease_uci.csv
│
│-- model/
│   ├── logistic_regression_model.pkl
│   ├── logistic_regression_model.py
│   ├── decision_tree_model.pkl
│   ├── decision_tree_model.py
│   ├── knn_model.pkl
│   ├── knn_model.py
│   ├── naive_bayes_model.pkl
│   ├── naive_bayes_model.py
│   ├── random_forest_model.pkl
│   ├── random_forest_model.py
│   ├── xgboost_model.pkl
│   ├── xgboost_model.py
│   └── scaler.pkl

```
This structure separates **data, model training, and deployment components** to ensure clarity, reproducibility, and maintainability

---

## 9. Deployment
The Streamlit application is deployed using **Streamlit Community Cloud** and is accessible via a public link.  
All dependencies are managed using `requirements.txt` to ensure smooth deployment.

---

## 10. Links
🔗 **Live Streamlit App:** *https://heart-disease-ml-app-etrbva2rt8vttjqxq7h2pq.streamlit.app/*  
🔗 **GitHub Repository:** *https://github.com/GGMgithub/heart-disease-ml-app/tree/main*  

---

## 11. Assignment Details
- **Student Name:** Goutham G M
- **Program:** M.Tech (AIML)  
- **Course:** Machine Learning  
- **Institution:** BITS Pilani  

---
