import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

url = "https://api.open-meteo.com/v1/forecast?latitude=-0.4917&longitude=117.1458&daily=temperature_2m_max&timezone=auto"
data = requests.get(url).json()

df = pd.DataFrame({
    "Tanggal": pd.to_datetime(data["daily"]["time"]),
    "Suhu": data["daily"]["temperature_2m_max"]
})

df = df.tail(7).reset_index(drop=True)

df["Suhu_t-1"] = df["Suhu"].shift(1)
df["Suhu_t-2"] = df["Suhu"].shift(2)
df["Suhu_t-3"] = df["Suhu"].shift(3)

df = df.dropna().reset_index(drop=True)

X = df[["Suhu_t-1", "Suhu_t-2", "Suhu_t-3"]]
y = df["Suhu"]

model = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X, y)

r2_train = model.score(X, y)
print("R² Score (seluruh data):", round(r2_train, 3))

suhu_terakhir = df.iloc[-1]
input_prediksi = pd.DataFrame([[suhu_terakhir["Suhu"],
                                df.iloc[-2]["Suhu"],
                                df.iloc[-3]["Suhu"]]],
                              columns=["Suhu_t-1", "Suhu_t-2", "Suhu_t-3"])

prediksi_suhu = model.predict(input_prediksi)[0]
tanggal_prediksi = df["Tanggal"].iloc[-1] + pd.Timedelta(days=1)

print(f"Suhu terakhir: {suhu_terakhir['Suhu']} °C")
print(f"Prediksi besok ({tanggal_prediksi.date()}): {round(prediksi_suhu, 2)} °C")

plt.figure(figsize=(12,6))
plt.plot(df["Tanggal"], df["Suhu"], marker='o', label="Data Historis")
plt.plot(
    [df["Tanggal"].iloc[-1], tanggal_prediksi],
    [df["Suhu"].iloc[-1], prediksi_suhu],
    marker='o', linestyle='--', color='red', label="Prediksi Besok"
)
plt.xticks(rotation=45)
plt.xlabel("Tanggal")
plt.ylabel("Suhu (°C)")
plt.title("Prediksi Suhu Harian Kota Samarinda")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()