"""
STEP 3: Streamlit web interface.
Run:  streamlit run app.py
"""
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Canteen Demand Prediction", page_icon="🍽️", layout="wide")


# ---------------- Load saved files (cached so they load only once) ----------------
@st.cache_resource
def load_model():
    return joblib.load("model/demand_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("data/canteen_sales.csv", parse_dates=["date"])


@st.cache_data
def load_metrics():
    with open("model/metrics.json") as f:
        return json.load(f)


try:
    model = load_model()
    data = load_data()
    metrics = load_metrics()
    test_results = pd.read_csv("model/test_results.csv")
except FileNotFoundError:
    st.error("Model files not found. Run 'python generate_data.py' and "
             "'python train_model.py' first.")
    st.stop()

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.title("🍽️ AI-Based College Canteen Food Demand Prediction System")
st.write("Predict how many units of a food item will be sold, so the canteen "
         "can prepare the right quantity and reduce food waste.")

tab1, tab2, tab3 = st.tabs(["🔮 Predict Demand", "📊 Sales Analysis", "🤖 Model Performance"])

# ======================= TAB 1: PREDICTION =======================
with tab1:
    st.subheader("Enter details")
    col1, col2 = st.columns(2)

    with col1:
        item = st.selectbox("Food item", sorted(data["food_item"].unique()))
        date = st.date_input("Date")
        temperature = st.slider("Expected temperature (°C)", 10, 45, 30)

    with col2:
        is_rainy = st.checkbox("Rainy day")
        is_exam = st.checkbox("Exam period")
        is_holiday = st.checkbox("Holiday")

    if st.button("Predict Demand", type="primary"):
        # Build one row with the same columns used during training
        input_df = pd.DataFrame([{
            "food_item": item,
            "day_of_week": date.weekday(),
            "month": date.month,
            "temperature": temperature,
            "is_rainy": int(is_rainy),
            "is_exam_period": int(is_exam),
            "is_holiday": int(is_holiday),
        }])

        predicted = max(0, int(round(model.predict(input_df)[0])))

        # Compare with the item's historical average to label the demand
        avg_sales = data.loc[data["food_item"] == item, "quantity_sold"].mean()
        ratio = predicted / avg_sales
        if ratio >= 1.15:
            level, icon = "HIGH", "🔴"
        elif ratio <= 0.85:
            level, icon = "LOW", "🟢"
        else:
            level, icon = "NORMAL", "🟡"

        # Prepare ~10% extra as a safety buffer
        suggested = int(round(predicted * 1.10))

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted demand", f"{predicted} units")
        m2.metric("Historical average", f"{avg_sales:.0f} units")
        m3.metric("Demand level", f"{icon} {level}")

        st.success(
            f"**Recommendation:** Prepare about **{suggested} units** of "
            f"**{item}** on {DAY_NAMES[date.weekday()]}, {date.strftime('%d %b %Y')} "
            f"(predicted demand + 10% safety buffer)."
        )
        if level == "HIGH":
            st.warning("Demand is higher than usual. Arrange extra raw material and staff.")
        elif level == "LOW":
            st.info("Demand is lower than usual. Prepare less to avoid food waste.")

        # Chart: predicted vs average vs recommended
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(["Historical avg", "Predicted", "Recommended"],
                      [avg_sales, predicted, suggested],
                      color=["#9ca3af", "#3b82f6", "#10b981"])
        ax.bar_label(bars, fmt="%.0f")
        ax.set_ylabel("Units")
        ax.set_title(f"{item}: demand comparison")
        st.pyplot(fig)

# ======================= TAB 2: SALES ANALYSIS =======================
with tab2:
    st.subheader("Historical sales analysis")
    a, b = st.columns(2)

    with a:
        fig, ax = plt.subplots(figsize=(6, 4))
        avg_by_item = data.groupby("food_item")["quantity_sold"].mean().sort_values()
        avg_by_item.plot(kind="barh", ax=ax, color="#3b82f6")
        ax.set_xlabel("Average units sold per day")
        ax.set_title("Average daily sales by item")
        st.pyplot(fig)

    with b:
        sel = st.selectbox("Choose item for day-wise analysis",
                           sorted(data["food_item"].unique()), key="analysis_item")
        sub = data[data["food_item"] == sel]
        by_day = sub.groupby("day_of_week")["quantity_sold"].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([d[:3] for d in DAY_NAMES], by_day.values, color="#10b981")
        ax.set_ylabel("Average units sold")
        ax.set_title(f"{sel}: average sales by day of week")
        st.pyplot(fig)

    # Monthly trend for the selected item
    monthly = sub.groupby(sub["date"].dt.to_period("M"))["quantity_sold"].sum()
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(monthly.index.astype(str), monthly.values, marker="o", color="#f59e0b")
    ax.set_title(f"{sel}: total monthly sales trend")
    ax.set_ylabel("Units sold")
    plt.xticks(rotation=60)
    st.pyplot(fig)

    with st.expander("View raw dataset (first 100 rows)"):
        st.dataframe(data.head(100))

# ======================= TAB 3: MODEL PERFORMANCE =======================
with tab3:
    st.subheader("Model evaluation")
    st.write(f"Best model selected: **{metrics['best_model']}**")

    st.table(pd.DataFrame(metrics["results"]).T.rename(columns={"R2": "R² Score"}))
    st.caption("MAE = average prediction error in units (lower is better). "
               "R² = how much of the variation the model explains (closer to 1 is better).")

    c, d = st.columns(2)
    with c:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(test_results["actual"], test_results["predicted"], alpha=0.3, s=10)
        lim = [0, max(test_results["actual"].max(), test_results["predicted"].max())]
        ax.plot(lim, lim, color="red", linestyle="--", label="Perfect prediction")
        ax.set_xlabel("Actual quantity")
        ax.set_ylabel("Predicted quantity")
        ax.set_title("Actual vs Predicted (test data)")
        ax.legend()
        st.pyplot(fig)

    with d:
        # Which input factors matter most? (only Random Forest has this)
        inner = model.named_steps["model"]
        if hasattr(inner, "feature_importances_"):
            names = model.named_steps["prep"].get_feature_names_out()
            names = [n.replace("food__food_item_", "").replace("remainder__", "") for n in names]
            imp = pd.Series(inner.feature_importances_, index=names).sort_values().tail(10)
            fig, ax = plt.subplots(figsize=(5, 5))
            imp.plot(kind="barh", ax=ax, color="#8b5cf6")
            ax.set_title("Top 10 important features")
            st.pyplot(fig)
        else:
            st.info("Feature importance is available only for tree-based models.")
