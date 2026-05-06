import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#Standard column names
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

#File names and source labels
files = {
    "processed.cleveland.data": "cleveland",
    "processed.hungarian.data": "hungarian",
    "processed.switzerland.data": "switzerland",
    "processed.va.data": "va"
}

#1. Load and combine datasets
dfs = []

for file_name, source_name in files.items():
    df = pd.read_csv(
        file_name,
        header=None,
        names=columns,
        na_values="?"
    )
    df["source"] = source_name
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
raw_df = combined_df.copy()


# 2. Create binary target
combined_df["target"] = (combined_df["num"] > 0).astype(int)
combined_df = combined_df.drop(columns=["num"])


#3. Convert non-source columns to numeric
for col in combined_df.columns:
    if col != "source":
        combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")


# 4. Missing summary before cleaning
missing_summary_before = pd.DataFrame({
    "column": combined_df.columns,
    "missing_count_before": combined_df.isna().sum().values,
    "missing_percent_before": (combined_df.isna().mean() * 100).round(2).values
})


#5. Drop columns with too much missing data
columns_to_drop = ["ca", "thal"]
combined_df = combined_df.drop(columns=columns_to_drop)


#6. Fill missing values
#    Numeric continuous columns -> median
#   Categorical coded columns -> mode
continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_code_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "target"]

for col in continuous_cols:
    combined_df[col] = combined_df[col].fillna(combined_df[col].median())

for col in categorical_code_cols:
    combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])

#Ensure coded categorical columns are integers
for col in categorical_code_cols:
    combined_df[col] = combined_df[col].round().astype(int)



# 7. Map coded values to readable labels
sex_map = {
    0: "female",
    1: "male"
}

cp_map = {
    1: "typical angina",
    2: "atypical angina",
    3: "non-anginal pain",
    4: "asymptomatic"
}

fbs_map = {
    0: "fasting blood sugar <= 120 mg/dl",
    1: "fasting blood sugar > 120 mg/dl"
}

restecg_map = {
    0: "normal",
    1: "ST-T wave abnormality",
    2: "left ventricular hypertrophy"
}

exang_map = {
    0: "no",
    1: "yes"
}

slope_map = {
    1: "upsloping",
    2: "flat",
    3: "downsloping"
}

target_map = {
    0: "no heart disease",
    1: "heart disease"
}

combined_df["sex"] = combined_df["sex"].map(sex_map)
combined_df["cp"] = combined_df["cp"].map(cp_map)
combined_df["fbs"] = combined_df["fbs"].map(fbs_map)
combined_df["restecg"] = combined_df["restecg"].map(restecg_map)
combined_df["exang"] = combined_df["exang"].map(exang_map)
combined_df["slope"] = combined_df["slope"].map(slope_map)
combined_df["target"] = combined_df["target"].map(target_map)

#Save readable cleaned version
cleaned_df = combined_df.copy()


#8. One-hot encode categorical columns
categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "source"]
encoded_df = pd.get_dummies(cleaned_df, columns=categorical_cols, drop_first=True)

#Convert boolean dummy columns from True/False to 1/0
bool_cols = encoded_df.select_dtypes(include=["bool"]).columns
encoded_df[bool_cols] = encoded_df[bool_cols].astype(int)

# Convert target back to numeric for modeling
encoded_df["target"] = encoded_df["target"].map({
    "no heart disease": 0,
    "heart disease": 1
})



# 9. Separate features and target
X = encoded_df.drop(columns=["target"])
y = encoded_df["target"]



# 10. Train/evaluation/test split
#     Final split:
#     60% training
#     10% evaluation/validation
#     30% testing
#First split off the final 30% test set
X_train_eval, X_test, y_train_eval, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

#Split the remaining 70% into 60% training and 10% evaluation
X_train, X_eval, y_train, y_eval = train_test_split(
    X_train_eval,
    y_train_eval,
    test_size=1/7,
    random_state=42,
    stratify=y_train_eval
)


#11. Min-max normalization
#     Fit only on the 60% training data
#    Transform evaluation and test data using the same scaler
numeric_cols_for_scaling = ["age", "trestbps", "chol", "thalach", "oldpeak"]

scaler = MinMaxScaler()

X_train_scaled = X_train.copy()
X_eval_scaled = X_eval.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols_for_scaling] = scaler.fit_transform(
    X_train[numeric_cols_for_scaling]
)

X_eval_scaled[numeric_cols_for_scaling] = scaler.transform(
    X_eval[numeric_cols_for_scaling]
)

X_test_scaled[numeric_cols_for_scaling] = scaler.transform(
    X_test[numeric_cols_for_scaling]
)


#12. Recombine scaled X and y for export
train_final = X_train_scaled.copy()
train_final["target"] = y_train.values

eval_final = X_eval_scaled.copy()
eval_final["target"] = y_eval.values

test_final = X_test_scaled.copy()
test_final["target"] = y_test.values

train_bool_cols = train_final.select_dtypes(include=["bool"]).columns
eval_bool_cols = eval_final.select_dtypes(include=["bool"]).columns
test_bool_cols = test_final.select_dtypes(include=["bool"]).columns

train_final[train_bool_cols] = train_final[train_bool_cols].astype(int)
eval_final[eval_bool_cols] = eval_final[eval_bool_cols].astype(int)
test_final[test_bool_cols] = test_final[test_bool_cols].astype(int)


#Use the scaler fitted on the 60% training data to avoid data leakage
full_scaled_df = encoded_df.copy()
full_scaled_df[numeric_cols_for_scaling] = scaler.transform(
    encoded_df[numeric_cols_for_scaling]
)

full_bool_cols = full_scaled_df.select_dtypes(include=["bool"]).columns
full_scaled_df[full_bool_cols] = full_scaled_df[full_bool_cols].astype(int)



#13. Missing summary after cleaning
missing_summary_after = pd.DataFrame({
    "column": cleaned_df.columns,
    "missing_count_after": cleaned_df.isna().sum().values,
    "missing_percent_after": (cleaned_df.isna().mean() * 100).round(2).values
})



#14. Data dictionary
data_dictionary = pd.DataFrame({
    "column": cleaned_df.columns,
    "description": [
        "Age in years",
        "Patient sex",
        "Chest pain type",
        "Resting blood pressure",
        "Serum cholesterol",
        "Fasting blood sugar category",
        "Resting ECG result",
        "Maximum heart rate achieved",
        "Exercise induced angina",
        "ST depression induced by exercise",
        "Slope of peak exercise ST segment",
        "Dataset source hospital",
        "Binary target: no heart disease or heart disease"
    ]
})


#15. Export to Excel
output_file = "heart_disease_combined_cleaned.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    raw_df.to_excel(writer, sheet_name="Raw_Combined_Data", index=False)
    missing_summary_before.to_excel(writer, sheet_name="Missing_Before", index=False)
    cleaned_df.to_excel(writer, sheet_name="Cleaned_Data", index=False)
    encoded_df.to_excel(writer, sheet_name="Encoded_Data", index=False)
    train_final.to_excel(writer, sheet_name="Train_Scaled", index=False)
    eval_final.to_excel(writer, sheet_name="Eval_Scaled", index=False)
    test_final.to_excel(writer, sheet_name="Test_Scaled", index=False)
    full_scaled_df.to_excel(writer, sheet_name="Full_Scaled_View", index=False)
    missing_summary_after.to_excel(writer, sheet_name="Missing_After", index=False)
    data_dictionary.to_excel(writer, sheet_name="Data_Dictionary", index=False)

print("Done!")
print(f"Excel file created: {output_file}")
print(f"Total rows: {len(encoded_df)}")
print(f"Training rows: {len(train_final)}")
print(f"Evaluation rows: {len(eval_final)}")
print(f"Testing rows: {len(test_final)}")

print("Training target counts:")
print(y_train.value_counts())

print("Evaluation target counts:")
print(y_eval.value_counts())

print("Testing target counts:")
print(y_test.value_counts())