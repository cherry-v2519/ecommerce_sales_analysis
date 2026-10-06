# E-commerce Sales Data Analysis

## 📌 About the Project

This project analyzes real-world e-commerce transaction data to understand sales performance, revenue trends, customer behavior, products, and country-wise sales.

The project focuses on cleaning raw data, performing exploratory data analysis (EDA), analyzing customer and order behavior, and creating visualizations to identify useful patterns and insights.

The project was later extended into a Machine Learning workflow using an insurance dataset, covering model training, evaluation, hyperparameter tuning, deployment, and model explainability.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SHAP
- Streamlit
- Joblib
- Jupyter Notebook

## 📊 Dataset

### UCI Online Retail Dataset

The dataset contains transaction records from an online retail store, including:

- Invoice number
- Product/Stock code
- Product description
- Quantity
- Invoice date
- Unit price
- Customer ID
- Country

## 🔍 E-commerce Analysis Performed

### Day 1 – Data Exploration, Cleaning and Visualization

- Dataset exploration
- Missing-value analysis
- Duplicate detection and removal
- Handling invalid transactions
- Revenue calculation
- Product-wise revenue analysis
- Monthly revenue analysis
- Country-wise revenue analysis
- Transaction revenue distribution
- Quantity vs Revenue analysis

### Day 2 – Customer and Order Analysis

- Unique order analysis
- Unique customer analysis
- Average Order Value (AOV)
- Top customers by revenue
- Customers with the highest number of orders
- Customer-level revenue analysis
- Customer-level AOV analysis

### Day 3 – Seaborn and Statistical Visualization

- Revenue distribution using Seaborn
- Revenue boxplot and outlier analysis
- Product revenue visualization
- Correlation analysis
- Quantity vs Revenue scatter plot
- Statistical data visualization using Seaborn

### Day 4 – Final Analysis and Business Insights

- Top products by quantity sold
- Monthly revenue growth analysis
- Highest-revenue month
- Monthly order analysis
- Top countries by quantity sold
- Top customers by quantity purchased
- Revenue per unit calculation

## 📈 E-commerce Visualizations

The project includes:

- Top 10 Products by Revenue
- Monthly Revenue Trend
- Top 10 Countries by Revenue
- Revenue Distribution
- Revenue Boxplot
- Top 10 Customers by Revenue
- Top 10 Customers by Number of Orders
- Top 10 Customers by Average Order Value
- Top 10 Products by Quantity Sold
- Monthly Revenue Growth
- Top 10 Countries by Quantity Sold
- Top 10 Customers by Quantity Purchased
- Quantity vs Revenue Scatter Plot
- Correlation Heatmap

# 🤖 Insurance Cost Prediction – Machine Learning

After completing the e-commerce analysis, the project was extended into a Machine Learning workflow using an insurance dataset.

The goal was to understand the complete Machine Learning lifecycle:

**Data Preparation → Model Training → Evaluation → Tuning → Deployment → Explainability**

## 📊 Insurance Dataset

The insurance dataset contains information such as:

- Age
- Sex
- BMI
- Number of children
- Smoker status
- Region
- Insurance charges

The dataset was preprocessed before applying Machine Learning models.

> Note: The `insurance.csv` dataset is excluded from the repository using `.gitignore`.

## 🔍 Machine Learning Analysis

### Day 5 – Data Preparation

- Exploratory Data Analysis
- Data cleaning
- Categorical feature encoding
- Feature engineering
- Feature scaling
- Data preprocessing

### Day 6 – Linear Regression

- Train-test split
- Linear Regression
- Insurance charge prediction
- Model evaluation using:
  - MAE
  - MSE
  - RMSE
  - R² Score
- Actual vs Predicted visualization
- Saved trained Linear Regression model

### Day 7 – Model Evaluation and Error Analysis

- Compared training and testing performance
- Analyzed possible overfitting and underfitting
- Calculated residuals
- Created residual plots
- Studied prediction errors
- Learned how residuals help evaluate model behavior

### Day 8 – Understanding Feature Importance

- Analyzed Linear Regression coefficients
- Compared positive and negative coefficients
- Studied the relationship between features and predictions
- Created coefficient-based feature analysis
- Visualized Linear Regression coefficients

### Day 9 – Random Forest Regression

- Built a Random Forest Regression model
- Compared Random Forest with Linear Regression
- Evaluated models using:
  - R² Score
  - MAE
  - RMSE
- Analyzed Random Forest feature importance
- Saved the Random Forest model

### Day 10 – Logistic Regression Classification

- Converted the insurance problem into a classification task
- Used smoker status as the classification target
- Built a Logistic Regression model
- Used stratified train-test splitting
- Evaluated the model using:
  - Accuracy
  - Confusion Matrix
  - Classification Report
- Analyzed Logistic Regression coefficients
- Saved the classification model

### Day 11 – Cross-Validation and Model Comparison

- Applied 5-fold cross-validation
- Compared Linear Regression and Random Forest
- Calculated mean cross-validation scores
- Analyzed score variation using standard deviation
- Visualized model performance

### Day 12 – Hyperparameter Tuning

- Used GridSearchCV for Random Forest
- Tested different combinations of:
  - Number of estimators
  - Maximum depth
  - Minimum samples split
- Selected the best hyperparameters
- Evaluated the tuned Random Forest model
- Saved the tuned model

### Day 13 – Model Deployment Basics

- Loaded the trained model using Joblib
- Used the saved model to make new predictions
- Created a reusable prediction function
- Verified model input features
- Learned the basic Machine Learning deployment workflow:

**Input → Loaded Model → Prediction**

### Day 14 – Streamlit ML Prediction App

- Built an interactive Streamlit application
- Loaded the trained Random Forest model
- Saved and loaded the preprocessing scaler
- Created input fields for customer details
- Applied the same preprocessing used during model training
- Generated insurance charge predictions
- Tested the application with different customer inputs

### Day 15 – Improving the Streamlit App

- Improved the application layout
- Added a sidebar with model information
- Added two-column input sections
- Added a reset inputs option
- Improved the prediction result display
- Added BMI category information
- Added prediction range messages
- Tested the application with different customer profiles
- Improved the overall user experience

### Day 16 – Model Explainability with SHAP

- Introduced Machine Learning model explainability
- Installed and imported SHAP
- Created a SHAP TreeExplainer for the Random Forest model
- Calculated SHAP values for the test dataset
- Created SHAP summary plots
- Created SHAP feature importance ranking
- Analyzed individual model predictions using SHAP
- Identified `isSmoker`, `age`, and `bmi` as the most influential features
- Learned how positive and negative SHAP values affect predictions

### Key Learning from SHAP

SHAP helped me understand not only what the Machine Learning model predicts, but also which features influence those predictions and in which direction.

## 🧠 Machine Learning Models Used

- Linear Regression
- Random Forest Regression
- Logistic Regression
- Tuned Random Forest Regression

## 🚀 Model Deployment

The trained Random Forest model was deployed using **Streamlit**.

The application allows users to enter:

- Age
- Sex
- BMI
- Number of children
- Smoker status
- Region

The application then applies the required preprocessing and generates an estimated insurance charge.

### Day 17 – Prediction Error Analysis

- Generated predictions using the tuned Random Forest model
- Calculated prediction errors
- Computed MAE and RMSE
- Compared prediction accuracy
- Visualized error distribution
- Analyzed largest and smallest prediction errors
- Compared actual vs predicted values
- Calculated the percentage of predictions within ₹5,000 error

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
├── app.py
├── README.md
├── .gitignore
│
├── online+retail/
│   └── Online Retail.xlsx
│
└── online+retail.zip
```

## 📁 Dataset Files

### E-commerce Dataset

The project uses the **UCI Online Retail Dataset** for e-commerce sales analysis.

### Insurance Dataset

The project uses an `insurance.csv` dataset for insurance cost prediction.

> Note: The `insurance.csv` file is excluded from the GitHub repository using `.gitignore`.

## 🎯 Key Learning Outcomes

Through this project, I gained hands-on experience with:

- Data cleaning
- Exploratory Data Analysis
- Data visualization
- Customer and sales analysis
- Feature engineering
- Feature scaling
- Categorical encoding
- Regression
- Classification
- Model evaluation
- Error analysis
- Cross-validation
- Hyperparameter tuning
- Random Forest
- Model deployment
- Streamlit
- Model explainability
- SHAP
- Git and GitHub

## 👩‍💻 Author

**Charanya Prasanna Punati**