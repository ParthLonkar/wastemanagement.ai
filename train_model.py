import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

engine = create_engine("mysql+pymysql://root:parth@2004@localhost/database_db")


try:
    df = pd.read_sql("SELECT * FROM inventory", con=engine)
except Exception as e:
    print("❌ Failed to load data from database:", e)
    exit()

if df.empty:
    print("❌ No data found in 'inventory' table.")
    exit()

df.columns = df.columns.str.strip().str.lower()

if 'category' not in df.columns or 'is_waste' not in df.columns:
    print("❌ Required columns ('category', 'is_waste') not found.")
    exit()

le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

features = ['stock_level', 'avg_daily_sales', 'days_to_expiry', 'category_encoded']
target = 'is_waste'

if not all(col in df.columns for col in features):
    print("❌ One or more feature columns are missing from the table.")
    exit()

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")


os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/waste_model.pkl")
joblib.dump(le, "model/category_encoder.pkl")
print("✅ Model and label encoder saved successfully.")






