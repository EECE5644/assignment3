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

# ---------- Binary encoding
binary_cols = data.columns[data.nunique() == 2]
# print(binary_cols)  # Transmission
data["Transmission"] = data["Transmission"].map({"Manual": 0, "Automatic": 1}).astype("Int64")

# ---------- Ordinal encoding
# NOTE: Owner has a natural order (fewer owners = more valuable).
owner_order = {
    "UnRegistered Car": 0,
    "First": 1,
    "Second": 2,
    "Third": 3,
    "Fourth": 4,
    "4 or More": 5,
}
data["Owner"] = data["Owner"].map(owner_order).astype("Int64")

# ---------- Segmentation and train/test split
features = data.drop(columns="Price")
# NOTE: Log-transform the target variable to reduce skewness and improve model performance.
target = np.log1p(data["Price"])

X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.8, random_state=818)

# ---------- Detect and handle missing values (stats fit on the training split only)
# print(X_train.columns[X_train.isnull().any()])  # ['Engine', 'Max Power', 'Max Torque', 'Drivetrain', 'Seating Capacity', 'Fuel Tank Capacity']

todo_cols = ["Engine", "Max Power", "Max Torque", "Seating Capacity", "Fuel Tank Capacity"]

# NOTE: Fill fallback: Model -> Make -> global median, all computed from X_train only.
def fit_median_fill_stats(data: pd.DataFrame, cols: list) -> dict:
    return {
        col: {
            "by_model": data.groupby("Model")[col].median().round(),
            "by_make": data.groupby("Make")[col].median().round(),
            "global": round(data[col].median()),
        }
        for col in cols
    }


def apply_median_fill(data: pd.DataFrame, stats: dict) -> pd.DataFrame:
    for col, s in stats.items():
        data[col] = data[col].fillna(data["Model"].map(s["by_model"]))
        data[col] = data[col].fillna(data["Make"].map(s["by_make"]))
        data[col] = data[col].fillna(s["global"])
    return data


median_fill_stats = fit_median_fill_stats(X_train, todo_cols)
X_train = apply_median_fill(X_train, median_fill_stats)
X_test = apply_median_fill(X_test, median_fill_stats)


def mode_or_na(s: pd.Series):
    m = s.mode()
    return m.iat[0] if not m.empty else pd.NA


# Drivetrain is categorical -> same Model -> Make -> global hierarchy, but mode instead of median.
def fit_drivetrain_fill_stats(train: pd.DataFrame) -> dict:
    return {
        "by_model": train.groupby("Model")["Drivetrain"].agg(mode_or_na),
        "by_make": train.groupby("Make")["Drivetrain"].agg(mode_or_na),
        "global": train["Drivetrain"].mode().iat[0],
    }


def apply_drivetrain_fill(data: pd.DataFrame, stats: dict) -> pd.DataFrame:
    data = data.copy()
    data["Drivetrain"] = data["Drivetrain"].fillna(data["Model"].map(stats["by_model"]))
    data["Drivetrain"] = data["Drivetrain"].fillna(data["Make"].map(stats["by_make"]))
    data["Drivetrain"] = data["Drivetrain"].fillna(stats["global"])
    return data


drivetrain_fill_stats = fit_drivetrain_fill_stats(X_train)
X_train = apply_drivetrain_fill(X_train, drivetrain_fill_stats)
X_test = apply_drivetrain_fill(X_test, drivetrain_fill_stats)

# X_train.info()
# print(X_train.nunique())

# TEST: Drop `Model` column.
X_train.drop(columns=["Model"], inplace=True)
X_test.drop(columns=["Model"], inplace=True)

# ---------- Replace rare categories
# NOTE: Change the rare categories to "Other" for Make and Fuel Type.
def fit_rare_categories(col: pd.Series, threshold: int = 10) -> list:
    counts = col.value_counts()
    return counts[counts < threshold].index.tolist()


def apply_rare_categories(col: pd.Series, rare_categories: list) -> pd.Series:
    return col.replace(rare_categories, "Other")


for col in ["Make", "Fuel Type"]:
    rare_categories = fit_rare_categories(X_train[col])
    X_train[col] = apply_rare_categories(X_train[col], rare_categories)
    X_test[col] = apply_rare_categories(X_test[col], rare_categories)

# for col in ["Make", "Fuel Type", "Seller Type", "Drivetrain"]: print(X_train[col].value_counts())

# ---------- Dummy encoding
categorical_cols = X_train.select_dtypes(include=["string"]).columns.tolist()
# print(categorical_cols)  # ['Make', 'Fuel Type', 'Seller Type', 'Drivetrain']

X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True).reindex(columns=X_train.columns, fill_value=False)

bool_cols = X_train.select_dtypes(include=["bool"]).columns
X_train[bool_cols] = X_train[bool_cols].astype("Int64")
X_test[bool_cols] = X_test[bool_cols].astype("Int64")
# X_train.info()


# ==================== Model Training ====================
model = LinearRegression()
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
