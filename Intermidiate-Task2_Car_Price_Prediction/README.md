# Car Selling Price Prediction

A machine learning project that predicts the resale (selling) price of a used car based on features like showroom price, age, kilometers driven, fuel type, seller type, transmission, and number of previous owners.



Project Overview

Used car pricing is often inconsistent and subjective, relying on manual negotiation and guesswork. This project builds a regression model that estimates a fair selling price for a used car based on historical listing data, helping both sellers set realistic prices and buyers evaluate whether an asking price is fair.

**Type:** Supervised Machine Learning — Regression
**Goal:** Predict `Selling_Price` (in lakhs INR) for a used car



Dataset Description

The dataset contains **301 used car listings** with the following original features:

| Column | Description |
|---|---|
| `Car_Name` | Model name of the car |
| `Year` | Year of purchase |
| `Selling_Price` | Price the owner wants to sell at (in lakhs) — **target variable** |
| `Present_Price` | Current showroom price of the new model (in lakhs) |
| `Kms_Driven` | Total kilometers driven |
| `Fuel_Type` | Petrol / Diesel / CNG |
| `Seller_Type` | Dealer / Individual |
| `Transmission` | Manual / Automatic |
| `Owner` | Number of previous owners |



Technologies Used

- **Python 3.11**
- **Pandas** & **NumPy** — data manipulation
- **Matplotlib** & **Seaborn** — data visualization
- **Scikit-learn** — model training and evaluation
- **Joblib** — model serialization
- **VS Code** — development environment



Methodology

### 1. Exploratory Data Analysis (EDA)
- Statistical summary, missing value checks, duplicate detection
- Distribution analysis of `Selling_Price` (found to be right-skewed)
- Correlation heatmap between numeric features
- Category-wise price comparison (Fuel Type, Transmission)

### 2. Data Cleaning & Preprocessing
- Removed 2 duplicate rows
- Removed an extreme outlier (500,000 km — physically implausible)
- Engineered `Car_Age` feature from `Year` (more meaningful than raw year)
- Dropped `Car_Name` (98 unique values — too high cardinality for a 300-row dataset)
- One-hot encoded categorical features (`Fuel_Type`, `Seller_Type`, `Transmission`) with `drop_first=True` to avoid the dummy variable trap

### 3. Model Training
Trained and compared three regression models:
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

Data was split 80/20 (train/test) with `random_state=42` for reproducibility.

### 4. Model Evaluation
Evaluated using:
- **MAE (Mean Absolute Error)** — average prediction error in lakhs
- **RMSE (Root Mean Squared Error)** — penalizes large errors more heavily
- **R² Score** — proportion of price variance explained by the model

---

Results

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| **Linear Regression** ✅ | **1.376** | **2.414** | **0.785** |
| Decision Tree | 1.771 | 3.306 | 0.597 |
| Random Forest | 1.554 | 3.561 | 0.532 |

Winner: Linear Regression

Why Linear Regression outperformed tree-based models:
- The dataset is relatively small (~240 training rows), which causes tree-based models (Decision Tree, Random Forest) to overfit the training data and generalize poorly.
- `Selling_Price` has a strong, largely linear relationship with `Present_Price` — used car resale value tends to be a fairly consistent proportion of original price minus depreciation, a pattern linear models capture very efficiently.
- **Key takeaway:** More complex models aren't automatically better. With small, mostly-linear datasets, simpler models can outperform ensemble methods.

### Model Limitations
- Performs best on **mainstream, mid-range cars** (well-represented in training data)
- Less reliable for **very cheap or very expensive vehicles** (SUVs, luxury cars) due to sparse representation in the dataset
- CNG-fueled cars are underrepresented (only 2 samples) — predictions for CNG vehicles should be treated with caution
