"""
STEP 1: Generate a realistic sample dataset of college canteen sales.

Real canteen data is hard to get, so we simulate it using sensible rules
(e.g. cold coffee sells more when it is hot, tea/samosa sell more when it rains).
Run:  python generate_data.py
Output: data/canteen_sales.csv
"""
import os
import numpy as np
import pandas as pd

np.random.seed(42)  # same random numbers every time -> reproducible results

# Average units sold on a "normal" day for each item
BASE_DEMAND = {
    "Samosa": 120,
    "Veg Burger": 70,
    "Masala Dosa": 90,
    "Chole Bhature": 80,
    "Veg Biryani": 60,
    "Sandwich": 75,
    "Tea": 200,
    "Cold Coffee": 85,
}

# Indian public holidays (month, day) used as sample holidays
HOLIDAYS = [(1, 26), (3, 25), (8, 15), (10, 2), (10, 24), (11, 1), (12, 25)]

dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
rows = []

for date in dates:
    day_of_week = date.dayofweek          # Monday=0 ... Sunday=6
    month = date.month
    is_holiday = int((date.month, date.day) in HOLIDAYS)

    # Exam periods: roughly May and November every year
    is_exam = int(month in (5, 11))

    # Temperature (deg C): hot in Apr-Jun, cooler in Dec-Jan, small random change
    temp = 28 + 8 * np.sin((month - 3) / 12 * 2 * np.pi) + np.random.normal(0, 2)
    temp = round(float(temp), 1)

    # Rain: more likely in Jun-Sep (monsoon)
    rain_probability = 0.45 if month in (6, 7, 8, 9) else 0.08
    is_rainy = int(np.random.rand() < rain_probability)

    # How busy is the college today? (weekday factor)
    if day_of_week <= 4:
        crowd = 1.0       # Mon-Fri
    elif day_of_week == 5:
        crowd = 0.55      # Saturday
    else:
        crowd = 0.25      # Sunday
    if is_holiday:
        crowd *= 0.3

    for item, base in BASE_DEMAND.items():
        qty = base * crowd

        # Item specific rules
        if item == "Cold Coffee":
            qty *= 1 + (temp - 28) * 0.04      # hotter -> more
            if is_rainy:
                qty *= 0.8
        if item == "Tea":
            qty *= 1 - (temp - 28) * 0.02      # colder -> more
            if is_rainy:
                qty *= 1.3
            if is_exam:
                qty *= 1.2                     # students stay awake
        if item == "Samosa" and is_rainy:
            qty *= 1.3
        if item == "Veg Biryani" and day_of_week in (0, 4):
            qty *= 1.15                        # Monday & Friday specials
        if item == "Chole Bhature" and day_of_week == 2:
            qty *= 1.2                         # Wednesday special
        if is_exam and item not in ("Tea", "Cold Coffee"):
            qty *= 0.9                         # fewer students eat heavy meals

        qty *= np.random.normal(1, 0.08)       # random noise (about 8%)
        qty = max(0, int(round(qty)))

        rows.append([date, item, day_of_week, month, temp,
                     is_rainy, is_exam, is_holiday, qty])

df = pd.DataFrame(rows, columns=[
    "date", "food_item", "day_of_week", "month", "temperature",
    "is_rainy", "is_exam_period", "is_holiday", "quantity_sold",
])

os.makedirs("data", exist_ok=True)
df.to_csv("data/canteen_sales.csv", index=False)
print(f"Dataset created: {len(df)} rows saved to data/canteen_sales.csv")
print(df.head())
