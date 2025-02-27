import pandas as pd

# Load the dataset
data = pd.read_csv("titanic4.csv")

# Select relevant columns
columns_to_keep = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "survived"]
data = data[columns_to_keep]

# Handle missing values
data["age"].fillna(data["age"].median(), inplace=True)
data["fare"].fillna(data["fare"].median(), inplace=True)
data["embarked"].fillna(data["embarked"].mode()[0], inplace=True)

# Encode categorical variables
data["sex"] = data["sex"].map({"male": 0, "female": 1})
data["embarked"] = data["embarked"].map({"C": 0, "Q": 1, "S": 2})

# Save the cleaned dataset to a new CSV file (optional)
data.to_csv("titanic_cleaned.csv", index=False)

# Display the cleaned dataset
print(data.head())