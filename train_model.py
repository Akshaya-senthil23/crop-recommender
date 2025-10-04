import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load dataset
data_path = os.path.join("data", "crop_recommendation.csv")
df = pd.read_csv(data_path)

# Features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model to models folder
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/crop_model.pkl", "wb"))

print(" Model trained and saved as models/crop_model.pkl")
