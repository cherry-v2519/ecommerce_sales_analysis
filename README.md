# E-commerce Sales Data Analysis and Insurance Cost Prediction

## 📌 About the Project

This repository contains two practical data projects: E-commerce Sales Data Analysis and Insurance Cost Prediction using Machine Learning.

The e-commerce project focuses on cleaning and analyzing real-world transaction data to understand revenue trends, customer behavior, product performance, and country-wise sales.

The insurance project explores the Machine Learning workflow, including data preprocessing, model training, evaluation, hyperparameter tuning, prediction error analysis, model explainability, and deployment using Streamlit.

Through these projects, I gained hands-on experience with Python, data analysis, data visualization, Machine Learning, and GitHub.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SHAP
* Streamlit
* Joblib
* Jupyter Notebook
* Git and GitHub

---

# 📊 Project 1: E-commerce Sales Data Analysis

## 📌 Project Overview

This project analyzes the UCI Online Retail Dataset to understand sales performance, revenue trends, customer behavior, product performance, and country-wise sales.

The analysis includes data cleaning, exploratory data analysis (EDA), customer analysis, statistical visualization, and business insights.

## 📂 Dataset

### UCI Online Retail Dataset

The dataset contains transaction records from an online retail store, including:

* Invoice number
* Product or stock code
* Product description
* Quantity
* Invoice date
* Unit price
* Customer ID
* Country

## 🔍 E-commerce Analysis Performed

### Day 1 – Data Exploration, Cleaning and Visualization

* Explored the dataset and its structure.
* Analyzed missing values and duplicates.
* Handled invalid transactions.
* Calculated revenue.
* Analyzed product-wise revenue.
* Studied monthly revenue trends.
* Analyzed country-wise revenue.
* Visualized transaction revenue distribution.
* Explored the relationship between quantity and revenue.

### Day 2 – Customer and Order Analysis

* Analyzed unique orders and customers.
* Calculated Average Order Value (AOV).
* Identified top customers by revenue.
* Identified customers with the highest number of orders.
* Performed customer-level revenue analysis.
* Analyzed customer-level Average Order Value.

### Day 3 – Seaborn and Statistical Visualization

* Visualized revenue distributions using Seaborn.
* Created boxplots to analyze outliers.
* Visualized product revenue.
* Performed correlation analysis.
* Created quantity-versus-revenue scatter plots.
* Practiced statistical data visualization.

### Day 4 – Final Analysis and Business Insights

* Identified top products by quantity sold.
* Analyzed monthly revenue growth.
* Identified the highest-revenue month.
* Analyzed monthly order trends.
* Identified top countries by quantity sold.
* Analyzed top customers by quantity purchased.
* Calculated revenue per unit.

## 📈 E-commerce Visualizations

The project includes visualizations such as:

* Top 10 Products by Revenue
* Monthly Revenue Trend
* Top 10 Countries by Revenue
* Revenue Distribution
* Revenue Boxplot
* Top 10 Customers by Revenue
* Top 10 Customers by Number of Orders
* Top 10 Customers by Average Order Value
* Top 10 Products by Quantity Sold
* Monthly Revenue Growth
* Top 10 Countries by Quantity Sold
* Top 10 Customers by Quantity Purchased
* Quantity vs Revenue Scatter Plot
* Correlation Heatmap

---

# 🤖 Project 2: Insurance Cost Prediction – Machine Learning

## 📌 Project Overview

This project uses an insurance dataset to predict medical insurance charges based on customer information.

The project follows a practical Machine Learning workflow:

**Data Preparation → Model Training → Evaluation → Hyperparameter Tuning → Error Analysis → Explainability → Deployment**

## 📊 Insurance Dataset

The dataset contains information such as:

* Age
* Sex
* BMI
* Number of children
* Smoker status
* Region
* Insurance charges

The dataset was preprocessed before applying Machine Learning models.

**Note:** The `insurance.csv` dataset is excluded from the repository using `.gitignore`.

## 🔍 Machine Learning Analysis Performed

### Day 5 – Data Preparation

* Performed Exploratory Data Analysis.
* Cleaned and inspected the dataset.
* Encoded categorical features.
* Created additional features.
* Applied feature scaling.
* Prepared the data for Machine Learning.

### Day 6 – Linear Regression

* Split the dataset into training and testing sets.
* Built a Linear Regression model.
* Predicted insurance charges.
* Evaluated performance using MAE, MSE, RMSE, and R².
* Visualized actual versus predicted values.
* Saved the trained Linear Regression model.

### Day 7 – Model Evaluation and Error Analysis

* Compared training and testing performance.
* Examined possible overfitting and underfitting.
* Calculated residuals.
* Created residual plots.
* Studied prediction errors and model behavior.

### Day 8 – Understanding Linear Regression Coefficients

* Analyzed Linear Regression coefficients.
* Examined positive and negative coefficients.
* Studied how coefficients relate to model predictions.
* Visualized coefficient-based feature analysis.

### Day 9 – Random Forest Regression

* Built a Random Forest Regression model.
* Compared Random Forest with Linear Regression.
* Evaluated models using R², MAE, and RMSE.
* Analyzed built-in feature importance.
* Saved the trained Random Forest model.

### Day 10 – Logistic Regression Classification

* Created a classification task using smoker status as the target.
* Built a Logistic Regression model.
* Used stratified train-test splitting.
* Evaluated performance using accuracy, confusion matrix, and classification report.
* Analyzed Logistic Regression coefficients.
* Saved the classification model.

### Day 11 – Cross-Validation and Model Comparison

* Applied 5-fold cross-validation.
* Compared Linear Regression and Random Forest.
* Calculated mean cross-validation scores.
* Analyzed score variation using standard deviation.
* Visualized model performance.

### Day 12 – Hyperparameter Tuning

* Used GridSearchCV to tune the Random Forest model.
* Tested different combinations of model hyperparameters.
* Evaluated the tuned model.
* Compared the tuned model with previous models.
* Saved the tuned Random Forest model.

### Day 13 – Model Deployment Basics

* Loaded a trained model using Joblib.
* Used the saved model to make predictions on new input data.
* Created a reusable prediction function.
* Checked that input features matched the model's expected structure.
* Learned the basic deployment workflow:

**Input → Loaded Model → Prediction**

### Day 14 – Streamlit Machine Learning Application

* Built an interactive Streamlit application.
* Loaded the trained Random Forest model.
* Saved and loaded the preprocessing scaler.
* Created input fields for customer details.
* Applied preprocessing consistent with model training.
* Generated estimated insurance charges.
* Tested the application with different inputs.

### Day 15 – Improving the Streamlit Application

* Improved the application layout.
* Added a sidebar with model information.
* Organized input fields into columns.
* Added a reset inputs option.
* Improved prediction result display.
* Added BMI category information.
* Improved result messages and user experience.

### Day 16 – Model Explainability with SHAP

* Introduced Machine Learning model explainability.
* Used SHAP with the Random Forest model.
* Calculated SHAP values for the test dataset.
* Created SHAP summary plots.
* Generated SHAP feature importance rankings.
* Examined individual predictions.
* Identified `isSmoker`, `age`, and `bmi` among the most influential features in the analysis.
* Studied how positive and negative SHAP values affect predictions.

### Day 17 – Prediction Error Analysis

* Generated predictions using the tuned Random Forest model.
* Calculated prediction errors and absolute errors.
* Computed MAE and RMSE.
* Visualized the error distribution.
* Analyzed the largest and smallest prediction errors.
* Compared actual and predicted values.
* Calculated the percentage of predictions within ₹5,000 of the actual charges.

### Day 18 – Model Performance Comparison

* Compared Linear Regression, Random Forest, and Tuned Random Forest Regression.
* Evaluated models using R², MAE, and RMSE.
* Visualized model performance.
* Compared different evaluation metrics to understand model behavior.
* Identified the best-performing model based on the actual evaluation results.
* Saved the comparison results in `model_comparison.csv`.

### Day 19 – Feature Importance Comparison

* Analyzed feature importance from the tuned Random Forest model.
* Compared built-in Random Forest feature importance with SHAP feature importance.
* Identified features that ranked highly in both methods.
* Visualized the feature importance rankings.
* Saved the results in `feature_importance_comparison.csv`.

### Day 20 – Final Model Review and Project Wrap-Up

* Reviewed model performance and evaluation metrics.
* Reviewed feature importance and SHAP analysis.
* Tested the Streamlit application.
* Reviewed project files and outputs.
* Consolidated the key learnings from the Machine Learning workflow.
* Updated project documentation for GitHub.

## 🧠 Machine Learning Models Used

* Linear Regression
* Random Forest Regression
* Tuned Random Forest Regression
* Logistic Regression Classification

## 📊 Model Evaluation Metrics

The regression models were evaluated using:

* **R² Score:** Measures how well the model explains variation in the target.
* **MAE:** Measures the average absolute prediction error.
* **MSE:** Measures the average squared prediction error.
* **RMSE:** Measures prediction error while giving greater weight to larger errors.

Model comparison results are saved in `model_comparison.csv`.

## 🔎 Model Explainability

SHAP was used to understand how input features influence the tuned Random Forest model's predictions.

Feature importance from Random Forest was also compared with SHAP importance to examine whether the two methods highlighted similar features.

The comparison results are saved in `feature_importance_comparison.csv`.

Feature importance describes the model's learned behavior; it does not establish that a feature causes a particular insurance charge.

## 🚀 Model Deployment

The insurance cost prediction model was deployed through a Streamlit application.

Users can enter information such as:

* Age
* Sex
* BMI
* Number of children
* Smoker status
* Region

The application applies the required preprocessing and generates an estimated insurance charge.

The prediction is a model estimate and is not an actual insurance quote.

## 📂 Project Structure

```text
matplot/
│
├── ecommerce_sales_analysis.ipynb
├── insurance_data_analysis.ipynb
│
├── Insurance_Linear_Regression.pkl
├── Insurance_Random_Forest.pkl
├── Insurance_Random_Forest_Tuned.pkl
├── Insurance_Logistic_Regression.pkl
├── insurance_scaler.pkl
│
├── model_comparison.csv
├── feature_importance_comparison.csv
│
├── app.py
├── README.md
├── .gitignore
│
└── online+retail/
    └── Online Retail.xlsx
```

*This is an illustrative structure. Keep only files and folders that actually exist in your repository and that you intend to track with Git. The insurance dataset is excluded using `.gitignore`.*

## 📁 Dataset Information

### E-commerce Dataset

The project uses the UCI Online Retail Dataset for sales analysis.

### Insurance Dataset

The project uses an `insurance.csv` dataset for insurance cost prediction.

**Note:** The insurance dataset is excluded from the GitHub repository using `.gitignore`.

## 🎯 Key Learning Outcomes

Through these projects, I gained hands-on experience with:

* Python programming
* Data cleaning and preprocessing
* Exploratory Data Analysis
* Data visualization
* Customer and sales analysis
* Feature engineering
* Categorical encoding
* Feature scaling
* Regression and classification
* Linear Regression
* Random Forest
* Logistic Regression
* Model evaluation
* Cross-validation
* Hyperparameter tuning with GridSearchCV
* Prediction error analysis
* Feature importance analysis
* SHAP-based model explainability
* Building Machine Learning prediction functions
* Deploying a Machine Learning model using Streamlit
* Git and GitHub project management

## 👩‍💻 Author

**Charanya Prasanna Punati**
