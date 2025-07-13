import pandas as pd
from sqlalchemy import create_engine
from datetime import date
import os
import joblib


engine = create_engine("mysql+pymysql://root:parth@2004@localhost/database_db")


try:
    df = pd.read_sql("SELECT * FROM inventory", con=engine)
except Exception as e:
    print("❌ Failed to load data:", e)
    exit()

if df.empty:
    print("❌ 'inventory' table is empty.")
    exit()


try:
    model = joblib.load("model/waste_model.pkl")
    le = joblib.load("model/category_encoder.pkl")
except Exception as e:
    print("❌ Model files not found:", e)
    exit()

df.columns = df.columns.str.strip().str.lower()
df['category_encoded'] = le.transform(df['category'])
X = df[['stock_level', 'avg_daily_sales', 'days_to_expiry', 'category_encoded']]

df['predicted_is_waste'] = model.predict(X)


inventory_total = df['stock_level'].sum()
predicted_waste_total = df[df['predicted_is_waste'] == 1]['stock_level'].sum()
waste_diversion_rate = 100 * (1 - predicted_waste_total / inventory_total)

co2_saved = predicted_waste_total * 5.6

expired = df[df['days_to_expiry'] <= 0]
expiring_soon = df[(df['days_to_expiry'] > 0) & (df['days_to_expiry'] <= 5)]


carbon_goal_kg = 10000
carbon_progress = (co2_saved / carbon_goal_kg) * 100

metrics = pd.DataFrame([{
    "date": date.today(),
    "total_inventory_units": inventory_total,
    "predicted_waste_units": predicted_waste_total,
    "waste_diversion_rate_%": round(waste_diversion_rate, 2),
    "estimated_CO2_saved_kg": round(co2_saved, 2),
    "expired_products": len(expired),
    "about_to_expire_products": len(expiring_soon),
    "carbon_footprint_goal_kg": carbon_goal_kg,
    "carbon_footprint_progress_%": round(carbon_progress, 2),
    "total_products": len(df),
    "predicted_waste_items": len(df[df['predicted_is_waste'] == 1])
}])

try:
    metrics.to_sql("sustainability_metrics", con=engine, if_exists='append', index=False)
    print("✅ Sustainability KPIs saved to 'sustainability_metrics' table.")
except Exception as e:
    print("❌ Failed to save KPIs:", e)

# ✅ Print KPIs
print("\n🌱 Sustainability KPIs:")
print(metrics.T)


