"""Feature engineering shared by the training pipeline and the local scoring UI."""
import numpy as np
import pandas as pd

BRANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B10", "B11", "B12", "B13", "B14"]
REGIONS = ["R11", "R21", "R22", "R23", "R24", "R25", "R26", "R31", "R41", "R42", "R43",
           "R52", "R53", "R54", "R72", "R73", "R74", "R82", "R83", "R91", "R93", "R94"]
RAW_COLS = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]

# Bin edges for the GLM baseline (classic actuarial banding)
AGE_BINS = [0, 21, 26, 31, 41, 51, 61, 71, 120]
VEHAGE_BINS = [-1, 0, 2, 5, 10, 15, 100]
BM_BINS = [0, 50, 55, 60, 70, 80, 100, 120, 1000]
POWER_BINS = [0, 5, 6, 7, 8, 10, 100]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers the same way for training and what-if scoring."""
    d = df.copy()
    d["DrivAge"] = d["DrivAge"].clip(18, 90)
    d["VehAge"] = d["VehAge"].clip(0, 20)
    d["BonusMalus"] = d["BonusMalus"].clip(50, 150)
    d["VehPower"] = d["VehPower"].clip(4, 12)
    d["Density"] = d["Density"].clip(lower=1)
    return d


def gbm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered driver, vehicle and region features for the gradient-boosted models."""
    d = clean(df)
    X = pd.DataFrame(index=d.index)
    X["DrivAge"] = d["DrivAge"]
    X["VehAge"] = d["VehAge"]
    X["VehPower"] = d["VehPower"]
    X["BonusMalus"] = d["BonusMalus"]
    X["LogDensity"] = np.log(d["Density"])
    X["AreaCode"] = d["Area"].map(lambda a: ord(a) - ord("A")).astype(float)
    X["VehGas"] = (d["VehGas"].astype(str).str.contains("Diesel")).astype(float)
    X["VehBrand"] = pd.Categorical(d["VehBrand"], categories=BRANDS).codes.astype(float)
    X["Region"] = pd.Categorical(d["Region"], categories=REGIONS).codes.astype(float)
    # engineered
    X["YoungDriver"] = (d["DrivAge"] < 25).astype(float)
    X["SeniorDriver"] = (d["DrivAge"] >= 70).astype(float)
    X["NoClaimsHistory"] = (d["BonusMalus"] <= 50).astype(float)
    X["MalusDriver"] = (d["BonusMalus"] > 100).astype(float)
    X["NewVehicle"] = (d["VehAge"] <= 1).astype(float)
    X["PowerPerVehAge"] = d["VehPower"] / (d["VehAge"] + 1)
    X["BM_x_Age"] = d["BonusMalus"] * np.log(d["DrivAge"] - 16)
    X["Density_x_Age"] = np.log(d["Density"]) * (d["DrivAge"] < 30)
    X["DieselHighPower"] = ((X["VehGas"] == 1) & (d["VehPower"] >= 8)).astype(float)
    return X


GBM_CATEGORICAL = ["VehBrand", "Region"]


def glm_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot banded design matrix for the Poisson/Gamma GLM baselines."""
    d = clean(df)
    parts = {
        "age": pd.cut(d["DrivAge"], AGE_BINS, right=False).astype(str),
        "vage": pd.cut(d["VehAge"], VEHAGE_BINS).astype(str),
        "bm": pd.cut(d["BonusMalus"], BM_BINS, right=True).astype(str),
        "pow": pd.cut(d["VehPower"], POWER_BINS).astype(str),
        "area": d["Area"].astype(str),
        "gas": d["VehGas"].astype(str),
        "brand": d["VehBrand"].astype(str),
        "region": d["Region"].astype(str),
    }
    cat = pd.DataFrame(parts, index=d.index)
    X = pd.get_dummies(cat, dtype=float)
    X["LogDensity"] = np.log(d["Density"]) / 5.0
    return X


class Avg:
    """Seed-averaged ensemble of fitted regressors (same predict API)."""
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)
