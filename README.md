# 🏡 Housing Price Prediction

A Machine Learning project that predicts house prices based on property features such as area, number of rooms, construction year, location, furnishing, and property type.

## 📌 About the Project

The goal of this project is to build a machine learning model that estimates house prices from property information. The project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and a Streamlit web application for making predictions.

## ✨ Features

- Data cleaning and missing-value handling
- Exploratory Data Analysis (EDA)
- Data visualization
- Linear Regression and Random Forest model comparison
- House price prediction using a trained model
- Interactive Streamlit web application
- Model evaluation using MAE, RMSE, and R² score

## 🛠️ Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Web Application:** Streamlit
- **Model Storage:** Joblib
- **Version Control:** Git and GitHub

## 📂 Project Structure

```text
Housing/
├── data/
│   ├── house_price.csv
│   └── processed/
│       └── house_price_cleaned.csv
├── models/
│   ├── best_house_price_model.joblib
│   ├── linear_regression.joblib
│   ├── random_forest.joblib
│   └── model_comparison.csv
├── outputs/
│   ├── area_vs_price.png
│   ├── correlation_heatmap.png
│   ├── price_by_property_type.png
│   └── price_distribution.png
├── app.py
├── eda.py
├── preprocess.py
├── train_model.py
├── visualize.py
├── .gitignore
└── README.md
```

## 📊 Dataset

The dataset contains **1,124 rows and 9 columns**, including:

| Feature | Description |
|---|---|
| `Area_SqFt` | Property area in square feet |
| `Rooms` | Number of rooms |
| `Build_Year` | Year the property was built |
| `Location` | Property location |
| `Street_Type` | Type of street |
| `Furnishing` | Furnishing status |
| `Property_Type` | Type of property |
| `Has_Pool` | Whether the property has a pool |
| `Price` | Target variable to predict |

## ⚙️ Machine Learning Models

Two regression models were trained and evaluated:

- **Linear Regression**
- **Random Forest Regressor**

### Model Evaluation

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 20,089.17 | 34,258.79 | 0.9445 |
| Random Forest | 34,151.67 | 43,630.23 | 0.9099 |

Linear Regression was selected as the best model based on its lower Mean Absolute Error (MAE) on the test set.

*Note: These results are based on the current train-test split and may vary if the dataset or training configuration changes.*

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/GiftyMerin/Housing_Prediction.git
cd Housing_Prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

Your browser should open the interactive house price prediction application.

## 📈 Exploratory Data Analysis

The project generates visualizations to explore:

- House price distribution
- Relationship between property area and price
- Price differences by property type
- Correlations between numerical features

The generated charts are stored in the `outputs/` directory.

## 🔍 Future Improvements

- Add more relevant property features
- Experiment with additional regression algorithms
- Improve model performance through hyperparameter tuning
- Deploy the application online
- Add more interactive visualizations

## 👩‍💻 Author

**Gifty Merin**

GitHub: [@GiftyMerin](https://github.com/GiftyMerin)

## 📄 License

No license has been specified for this project yet.

---

⭐ If you find this project useful, feel free to explore the code and learn from the implementation.
