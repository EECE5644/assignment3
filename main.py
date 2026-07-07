import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ==================== Data Loading ====================
data = pd.read_csv(r"./car_details.csv")
data = data.convert_dtypes()
data["Fuel Tank Capacity"] = data["Fuel Tank Capacity"].round().astype("Int64")
# data.info()
# print(data.describe())

# Columns: Make,Model,Price,Year,Kilometer,Fuel Type,Transmission,Location,Color,Owner,Seller Type,Engine,Max Power,Max Torque,Drivetrain,Length,Width,Height,Seating Capacity,Fuel Tank Capacity


# ==================== Data Preprocessing ====================

# ---------- Remove useless columns
data.drop(columns=["Location", "Color", "Length", "Width", "Height"], inplace=True)
# data.info()


# ---------- Numerical extraction
# NOTE: Extract and convert these values to integer numbers.
todo_cols = ["Engine", "Max Power", "Max Torque"]
for col in todo_cols: data[col] = data[col].str.extract(r"^([\d.]+)")[0].astype(float).round().astype("Int64")

# data[todo_cols].info()
# print(data[todo_cols].head())


# ---------- Detect and handle missing values
missing_cols = data.columns[data.isnull().any()]
# print(missing_cols)  # ['Engine', 'Max Power', 'Max Torque', 'Drivetrain', 'Seating Capacity', 'Fuel Tank Capacity']

todo_cols = ["Engine", "Max Power", "Max Torque", "Seating Capacity", "Fuel Tank Capacity"]

# NOTE: Fill fallback: Model -> Make -> global median
for col in todo_cols:
    data[col] = data[col].fillna(data.groupby("Model")[col].transform("median").round())
    data[col] = data[col].fillna(data.groupby("Make")[col].transform("median").round())
    data[col] = data[col].fillna(round(data[col].median()))

def _mode_or_na(s: pd.Series):
    m = s.mode()
    return m.iat[0] if not m.empty else pd.NA

# Drivetrain is categorical -> same Model -> Make -> global hierarchy, but mode instead of median
data["Drivetrain"] = data["Drivetrain"].fillna(data.groupby("Model")["Drivetrain"].transform(_mode_or_na))
data["Drivetrain"] = data["Drivetrain"].fillna(data.groupby("Make")["Drivetrain"].transform(_mode_or_na))
data["Drivetrain"] = data["Drivetrain"].fillna(data["Drivetrain"].mode().iat[0])

# data.info()
# print(data.nunique())

# ---------- Binary encoding
binary_cols = data.columns[data.nunique() == 2]
# print(binary_cols)  # Transmission
data["Transmission"] = data["Transmission"].map({"Manual": 0, "Automatic": 1})


# ---------- Dummy encoding
todo_cols = data.select_dtypes(include=["string"]).columns
# print(todo_cols)  # ['Make', 'Model', 'Fuel Type', 'Owner', 'Seller Type', 'Drivetrain']

# TEST: Drop `Model` column.
data.drop(columns=["Model"], inplace=True)

# CASE: Change the rare categories to "Other" for Make, Fuel Type, Owner, Seller Type, Drivetrain
def replace_rare_categories(col: pd.Series, threshold: int = 10) -> pd.Series:
    counts = col.value_counts()
    rare_categories = counts[counts < threshold].index.tolist()
    return col.replace(rare_categories, "Other")


data["Make"] = replace_rare_categories(data["Make"])
data["Fuel Type"] = replace_rare_categories(data["Fuel Type"])

# for col in todo_cols: print(data[col].value_counts())

# CASE: Owner has a natural order (fewer owners = more valuable).
owner_order = {
    "UnRegistered Car": 0,
    "First": 1,
    "Second": 2,
    "Third": 3,
    "Fourth": 4,
    "4 or More": 5,
}
data["Owner"] = data["Owner"].map(owner_order)

data = pd.get_dummies(data, drop_first=True)
bool_cols = data.select_dtypes(include=["boolean"]).columns
data[bool_cols] = data[bool_cols].astype("Int64")
# data.info()


# ---------- Segmentation
features = data.drop(columns="Price")
# NOTE: Log-transform the target variable to reduce skewness and improve model performance.
target = np.log1p(data["Price"])
# features.info()


# ==================== Model Training ====================

# ---------- Dataset splitting and model initialization
model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.8, random_state=818)


# ---------- Model training
model.fit(X_train, y_train)


# ==================== Model Evaluation ====================
# NOTE: target is log1p(Price), so invert predictions with expm1 back to real
# currency before scoring -> MAE/RMSE below are in actual price units.
y_test_pred = np.expm1(model.predict(X_test))
y_test_actual = np.expm1(y_test)

r2 = r2_score(y_test_actual, y_test_pred)
mae = mean_absolute_error(y_test_actual, y_test_pred)
mse = mean_squared_error(y_test_actual, y_test_pred)
rmse = np.sqrt(mse)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")
