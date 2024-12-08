# **Breast Cancer Prediction Using Machine Learning**

This project aims to predict breast cancer diagnosis (malignant or benign) using machine learning techniques. It leverages Breast Cancer dataset to build and evaluate predictive models, demonstrating the potential of data science in medical diagnostics.

---

## **Table of Contents**
- [Dataset](#dataset)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Approach](#approach)
- [Results](#results)
- [Future Enhancements](#future-enhancements)


---

## **Dataset**

- **Number of Instances**: 570  
- **Number of Features**: 31 (numeric) + 1 target variable (diagnosis: 'M' for malignant, 'B' for benign)  
- **Missing Values**: None  

---

## **Features**

- **Data Preprocessing**: Cleaned data, handled missing values, and scaled features.  
- **Exploratory Data Analysis**: Visualized relationships between features and diagnosis using heatmaps and histograms.  
- **Modeling**: Built and evaluated models using Logistic Regression, Support Vector Machines (SVM), and Decision Trees.  
- **Evaluation Metrics**: Assessed performance using accuracy, precision, recall, F1-score, and confusion matrix.  

---

## **Technologies Used**

- **Programming Language**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Environment**: Jupyter Notebook  

---

## **Approach**

1. **Data Preprocessing**:
   - Dropped unnecessary columns (`id` and `Unnamed: 32`).  
   - Encoded the target variable (`M` -> 1, `B` -> 0).   

2. **Exploratory Data Analysis**:
   - Explored correlations between features and diagnosis.  
   - Visualized distributions of key features for malignant and benign cases.  

3. **Model Building**:
   - Implemented Logistic Regression, SVM, and Decision Tree models.  
   - Compared performance metrics for each model.  

4. **Model Evaluation**:
   - Calculated accuracy, precision, recall, F1-score, and confusion matrix for each model.  
   - Logistic Regression achieved the best accuracy (97%).  

---

## **Results**

- **Logistic Regression**:
  - Accuracy: 97%  
  - Precision: 98%  
  - Recall: 94%  
  - F1-Score: 96%  

- **SVM**:
  - Accuracy: 96%  
  - Precision: 97%  
  - Recall: 94%  
  - F1-Score: 95%  

- **Decision Tree**:
  - Accuracy: 94%  
  - Precision: 89%  
  - Recall: 94%  
  - F1-Score: 91%
 
  
---

## **Future Enhancements**

- Integrate additional datasets for broader applicability.
- Implement advanced models like Random Forests and Gradient Boosting.
- Deploy the model as a web application using Flask or Streamlit.
- Optimize feature selection to further improve accuracy and reduce computation time.


