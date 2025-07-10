
# Titanic-ml-pipeline – ML Classification Project

This project predicts the survival of passengers aboard the Titanic using various machine learning models. The focus is on end-to-end data preprocessing, feature engineering, model comparison, and evaluation.

---

## Problem Statement

The Titanic dataset contains demographic and travel information of passengers. The goal is to build a model that can accurately predict survival based on these variables.

---

## Key Features

- Cleaned and prepared the dataset by:
  - Dropping irrelevant columns: `Name`, `Ticket`, `Cabin`, `PassengerId`
  - Imputing missing `Age` values with median
  - Filling missing `Embarked` values with mode
- Encoded categorical variables:
  - Converted `Sex` into binary (`Gender`)
  - Applied one-hot encoding on `Embarked` with prefix
- Engineered new features to capture social and economic context:
  - `FamilySize`: total people per ticket
  - `IsAlone`: binary flag for passengers traveling alone
  - `Child`, `Senior`: age-based segmentation
  - `Fare_Per_Person`: normalized fare by family size
- These features helped isolate meaningful patterns in survival rates (e.g., solo travelers had lower survival, women and children had higher survival).

---

## Technologies Used

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost

---

## Models Compared

- Logistic Regression  
- Random Forest  
- XGBoost

All models were trained using an 80/20 train-test split. The best accuracy was achieved with Random Forest (~82%). Evaluation metrics included:

- ROC-AUC Curve
- Confusion Matrix
- Precision, Recall, F1-Score
- Kaggle public score: **0.73684**

---

## Next Steps / Improvements

- Incorporate SHAP or LIME for model explainability
- Apply GridSearchCV for better hyperparameter tuning
- Explore ensemble methods like VotingClassifier or Stacking
- Create a user-friendly interface with Streamlit to test predictions

---

## Notes

The project was developed as part of personal machine learning practice and university coursework. It demonstrates structured EDA, practical feature engineering, and clear model evaluation steps. The code is modular and well-commented for educational purposes.

This project aims to predict passenger survival from the Titanic dataset using machine learning techniques.  
It includes full exploratory data analysis (EDA), feature engineering, multiple model comparisons, and performance evaluation.

---

## Problem Statement

Can we accurately predict which passengers survived the Titanic disaster based on their personal and ticket information?

---

## Key Features

- Cleaned and prepared Titanic dataset  
- Feature engineering (`IsAlone`, `Fare_Per_Person`, `Title`, etc.)  
- Models used:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost  
- Accuracy: ~82% (on local test set)  
- Kaggle Score: 0.73684

---

## Technologies Used

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost


## Next Steps / Improvements

- Add SHAP or LIME explainability tools  
- Implement cross-validation and hyperparameter tuning  
- Deploy as a web app with Streamlit or Flask  
- Build ensemble models (Voting or Stacking)

---

