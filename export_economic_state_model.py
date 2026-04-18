import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


WINDOW_SIZE = 4
MAX_WEEKLY_SHIFT = 0.20
MODEL_OUTPUT_PATH = "/Users/Jiadong.Chen/Documents/New project 2/rf_model_state_fair.pkl"
METADATA_OUTPUT_PATH = "/Users/Jiadong.Chen/Documents/New project 2/rf_model_state_fair_metadata.pkl"
MASTER_SHEET_PATH = "/Users/Jiadong.Chen/Desktop/MSEF/Master Sheet for MSEF.csv"
SUPPLY_PATH = "/Users/Jiadong.Chen/Desktop/MSEF/Corn supply measured in quantity produced_cleaned.csv"


def load_and_prepare_data():
    data = pd.read_csv(MASTER_SHEET_PATH)
    data.columns = data.columns.str.strip()
    data = data.dropna(subset=["Year", "Period"]).copy()

    supply = pd.read_csv(SUPPLY_PATH)
    supply.columns = supply.columns.str.strip()
    supply["Annual_Production"] = pd.to_numeric(
        supply["Value"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    numeric_cols = [
        "Year",
        "Weekly Price",
        "Supply",
        "Demand_Ethanol",
        "Demand_Livestock",
        "Seasonality",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Week_Num"] = pd.to_numeric(
        data["Period"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "Year",
            "Week_Num",
            "Weekly Price",
            "Supply",
            "Demand_Ethanol",
            "Demand_Livestock",
            "Seasonality",
        ]
    ).copy()

    data["Year"] = data["Year"].astype(int)
    data["Week_Num"] = data["Week_Num"].astype(int)
    data = data[data["Year"] <= 2025].sort_values(["Year", "Week_Num"]).reset_index(drop=True)

    annual_supply_map = dict(zip(supply["Year"], supply["Annual_Production"]))
    data["Annual_Supply"] = data["Year"].map(annual_supply_map)
    data["Harvest_Change_Pct"] = data.groupby("Year")["Supply"].diff().fillna(0).clip(lower=0)
    data["Weekly_Bushels_Produced"] = (
        data["Harvest_Change_Pct"] / 100.0
    ) * data["Annual_Supply"]
    data["Weekly_Bushels_Produced"] = data["Weekly_Bushels_Produced"].fillna(0)
    data["Is_Harvesting"] = (data["Weekly_Bushels_Produced"] > 0).astype(int)
    data["Cumulative_Harvest"] = data.groupby("Year")["Weekly_Bushels_Produced"].cumsum()

    data["Previous_Week_Price"] = data["Weekly Price"].shift(1)
    data["Moving_Avg_4"] = data["Previous_Week_Price"].rolling(window=WINDOW_SIZE).mean()
    data["Momentum_Deviation"] = data["Weekly Price"] - data["Moving_Avg_4"]

    data = data.dropna(subset=["Moving_Avg_4", "Momentum_Deviation"]).copy()
    return data


def main():
    data = load_and_prepare_data()

    feature_columns = [
        "Week_Num",
        "Seasonality",
        "Weekly_Bushels_Produced",
        "Cumulative_Harvest",
        "Is_Harvesting",
        "Demand_Ethanol",
        "Demand_Livestock",
    ]

    train_mask = (data["Year"] >= 2015) & (data["Year"] <= 2022)
    test_mask = (data["Year"] >= 2023) & (data["Year"] <= 2025)

    train_data = data.loc[train_mask].copy()
    test_data = data.loc[test_mask].copy()

    if train_data.empty or test_data.empty:
        raise ValueError("Training or testing set is empty. Check the year filters.")

    x_train = train_data[feature_columns]
    y_train = train_data["Momentum_Deviation"]
    x_test = test_data[feature_columns]
    y_test = test_data["Momentum_Deviation"]
    moving_avg_test = test_data["Moving_Avg_4"]
    actual_price_test = test_data["Weekly Price"]

    model = RandomForestRegressor(
        n_estimators=600,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    print("Training economics-centered Random Forest model...")
    model.fit(x_train, y_train)

    predicted_deviation = model.predict(x_test)
    predicted_deviation_capped = np.clip(predicted_deviation, -MAX_WEEKLY_SHIFT, MAX_WEEKLY_SHIFT)
    predicted_price = moving_avg_test + predicted_deviation_capped

    price_mae = mean_absolute_error(actual_price_test, predicted_price)
    price_r2 = r2_score(actual_price_test, predicted_price)
    deviation_mae = mean_absolute_error(y_test, predicted_deviation)
    baseline_mae = mean_absolute_error(actual_price_test, moving_avg_test)

    importances = pd.DataFrame(
        {
            "Variable": feature_columns,
            "Importance (%)": model.feature_importances_ * 100,
        }
    ).sort_values(by="Importance (%)", ascending=False)

    print("\n--- ECONOMIC MODEL EVALUATION ---")
    print("Training Years: 2015-2022")
    print("Testing Years: 2023-2025")
    print(f"Training Rows: {len(train_data)}")
    print(f"Testing Rows: {len(test_data)}")
    print(f"Price MAE: ${price_mae:.4f}")
    print(f"Price R-squared: {price_r2:.4f}")
    print(f"Deviation MAE: ${deviation_mae:.4f}")
    print(f"4-week momentum baseline MAE: ${baseline_mae:.4f}")

    print("\n--- WHAT CAUSED THE PRICE TO DRIFT FROM ITS 1-MONTH MOMENTUM? ---")
    print(importances.to_string(index=False))

    metadata = {
        "feature_columns": feature_columns,
        "window_size": WINDOW_SIZE,
        "max_weekly_shift": MAX_WEEKLY_SHIFT,
        "train_years": [2015, 2022],
        "test_years": [2023, 2025],
        "model_type": "RandomForestRegressor",
        "target": "Momentum_Deviation",
        "baseline": "4-week moving average of previous prices",
        "price_mae": price_mae,
        "price_r2": price_r2,
        "baseline_mae": baseline_mae,
        "feature_importance": importances.to_dict(orient="records"),
    }

    with open(MODEL_OUTPUT_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(METADATA_OUTPUT_PATH, "wb") as metadata_file:
        pickle.dump(metadata, metadata_file)

    print(f"\nSaved model to: {MODEL_OUTPUT_PATH}")
    print(f"Saved metadata to: {METADATA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
