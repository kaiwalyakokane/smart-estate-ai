import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Estate AI",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# LOAD MODEL
# ==========================================

import joblib

saved_data = joblib.load("house_model.pkl")

model = saved_data["model"]
features = saved_data["features"]
cities = saved_data["cities"]
property_types = saved_data["property_types"]
locality_types = saved_data["locality_types"]
furnishing_options = saved_data["furnishing_options"]
metrics = saved_data["metrics"]

model = saved_data["model"]
features = saved_data["features"]
cities = saved_data["cities"]
property_types = saved_data["property_types"]
locality_types = saved_data["locality_types"]
furnishing_options = saved_data["furnishing_options"]
metrics = saved_data["metrics"]

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("train.csv")

# ==========================================
# PREPARE DATA
# ==========================================

data["Price_INR_Lakhs"] = pd.to_numeric(
    data["Price_INR_Lakhs"],
    errors="coerce"
)

valid_prices = data["Price_INR_Lakhs"].dropna()

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #102A43, #164E63);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    color: #D9EAF2;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

.price-card {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(135deg, #12372A, #176B52);
    text-align: center;
    margin: 15px 0 25px 0;
}

.price-label {
    font-size: 18px;
    color: #D1FAE5;
}

.price-value {
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin: 10px 0;
}

.price-caption {
    font-size: 14px;
    color: #D1FAE5;
}

div[data-testid="stMetric"] {
    background-color: #191F2A;
    border: 1px solid #303846;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #CBD5E1;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🏡 SMART ESTATE AI")

st.sidebar.caption("Machine Learning Real Estate Dashboard")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Price Predictor",
        "📊 Market Analytics",
        "🤖 Model Performance",
        "📁 Dataset Explorer",
        "ℹ️ About Project"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Technology Stack")
st.sidebar.write("""
🐍 Python

🧮 Pandas

🤖 Scikit-learn

🌐 Streamlit

📈 Matplotlib
""")

st.sidebar.info(
    "Educational project using synthetic property data. "
    "Predictions are not verified market valuations."
)

# ==========================================
# PAGE 1: PRICE PREDICTOR
# ==========================================

if page == "🏠 Price Predictor":

    st.markdown("""
    <div class="hero">
        <h1>🏡 Smart Estate AI</h1>
        <p>Intelligent House Price Prediction for Indian Cities</p>
        <p>Estimate property prices using machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Property Configuration</div>',
        unsafe_allow_html=True
    )

    with st.form("property_form"):

        left, right = st.columns(2)

        with left:

            city = st.selectbox("📍 City", cities)

            locality_type = st.selectbox(
                "🏙️ Locality Category",
                locality_types
            )

            property_type = st.selectbox(
                "🏠 Property Type",
                property_types
            )

            bhk = st.selectbox(
                "🛏️ Bedrooms (BHK)",
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                index=2
            )

            bathrooms = st.selectbox(
                "🚿 Bathrooms",
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                index=1
            )

            super_area = st.number_input(
                "📐 Super Area (sq. ft.)",
                min_value=100.0,
                max_value=20000.0,
                value=1200.0,
                step=50.0
            )

            carpet_area = st.number_input(
                "📏 Carpet Area (sq. ft.)",
                min_value=50.0,
                max_value=20000.0,
                value=900.0,
                step=50.0
            )

        with right:

            floor_number = st.number_input(
                "🏢 Floor Number",
                min_value=0,
                max_value=100,
                value=3
            )

            total_floors = st.number_input(
                "🏗️ Total Floors",
                min_value=1,
                max_value=100,
                value=10
            )

            age = st.number_input(
                "📅 Property Age (years)",
                min_value=0,
                max_value=150,
                value=5
            )

            furnishing = st.selectbox(
                "🛋️ Furnishing Status",
                furnishing_options
            )

            parking = st.selectbox(
                "🚘 Parking Available",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

            lift = st.selectbox(
                "🛗 Lift Available",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

            gated = st.selectbox(
                "🔐 Gated Community",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

            metro_distance = st.number_input(
                "🚇 Distance to Metro (km)",
                min_value=0.0,
                max_value=100.0,
                value=2.0,
                step=0.5
            )

            center_distance = st.number_input(
                "📍 Distance to City Center (km)",
                min_value=0.0,
                max_value=200.0,
                value=5.0,
                step=0.5
            )

        submitted = st.form_submit_button(
            "🔮 PREDICT HOUSE PRICE",
            use_container_width=True
        )

    if submitted:

        input_values = {
            "City": city,
            "Locality_Type": locality_type,
            "Property_Type": property_type,
            "BHK": bhk,
            "Bathrooms": bathrooms,
            "Super_Area_SqFt": super_area,
            "Carpet_Area_SqFt": carpet_area,
            "Floor_Number": floor_number,
            "Total_Floors": total_floors,
            "Age_of_Property": age,
            "Furnishing_Status": furnishing,
            "Parking": parking,
            "Lift_Available": lift,
            "Gated_Community": gated,
            "Distance_to_Metro_km": metro_distance,
            "Distance_to_City_Center_km": center_distance
        }

        input_df = pd.DataFrame([{
            feature: input_values[feature]
            for feature in features
        }])

        prediction_lakhs = float(model.predict(input_df)[0])

        st.session_state["prediction_lakhs"] = prediction_lakhs
        st.session_state["property_details"] = input_values

    if "prediction_lakhs" in st.session_state:

        price_lakhs = st.session_state["prediction_lakhs"]
        price_rupees = price_lakhs * 100000
        details = st.session_state["property_details"]

        st.divider()

        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">ESTIMATED PROPERTY VALUE</div>
            <div class="price-value">₹ {price_rupees:,.0f}</div>
            <div class="price-caption">
                ₹ {price_lakhs:,.2f} Lakhs | Random Forest Regression
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 Property Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("📍 City", details["City"])
        c2.metric("🛏️ Bedrooms", details["BHK"])
        c3.metric("🚿 Bathrooms", details["Bathrooms"])
        c4.metric(
            "📐 Super Area",
            f'{details["Super_Area_SqFt"]:,.0f} sq. ft.'
        )

        st.warning(
            "This is an educational estimate based on synthetic "
            "data, not a verified market valuation."
        )

# ==========================================
# PAGE 2: MARKET ANALYTICS
# ==========================================

elif page == "📊 Market Analytics":

    st.title("📊 Real Estate Analytics")

    st.caption(
        "Explore the generated property dataset across cities."
    )

    city_summary = (
        data.groupby("City")["Price_INR_Lakhs"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    city_summary.columns = [
        "City",
        "Average Price (Lakhs)",
        "Median Price (Lakhs)",
        "Property Records"
    ]

    a, b, c = st.columns(3)

    a.metric(
        "Cities",
        data["City"].nunique()
    )

    b.metric(
        "Property Records",
        f"{len(data):,}"
    )

    c.metric(
        "Average Sample Price",
        f"₹ {valid_prices.mean():,.2f} L"
    )

    st.divider()

    st.subheader("Average House Price by City")

    fig, ax = plt.subplots(figsize=(12, 5))

    plot_data = city_summary.sort_values(
        "Average Price (Lakhs)",
        ascending=False
    )

    plot_data.plot(
        x="City",
        y="Average Price (Lakhs)",
        kind="bar",
        ax=ax,
        legend=False
    )

    ax.set_ylabel("Average Price (Lakhs)")
    ax.set_xlabel("City")
    ax.tick_params(axis="x", rotation=35)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("City Comparison")

    st.dataframe(
        city_summary.sort_values(
            "Average Price (Lakhs)",
            ascending=False
        ).round(2),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Price Distribution")

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.hist(
        valid_prices,
        bins=40
    )

    ax.set_xlabel("Price (Lakhs)")
    ax.set_ylabel("Number of Properties")
    ax.set_title("Distribution of Sample Property Prices")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Area vs Property Price")

    scatter_data = data.dropna(
        subset=["Super_Area_SqFt", "Price_INR_Lakhs"]
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(
        scatter_data["Super_Area_SqFt"],
        scatter_data["Price_INR_Lakhs"],
        alpha=0.35
    )

    ax.set_xlabel("Super Area (sq. ft.)")
    ax.set_ylabel("Price (Lakhs)")
    ax.set_title("Relationship Between Area and Price")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ==========================================
# PAGE 3: MODEL PERFORMANCE
# ==========================================

elif page == "🤖 Model Performance":

    st.title("🤖 Machine Learning Performance")

    st.write(
        "Performance of the trained Random Forest model "
        "on the held-out test set."
    )

    a, b, c = st.columns(3)

    a.metric(
        "MAE",
        f'{metrics["mae_lakhs"]:.2f} L'
    )

    b.metric(
        "RMSE",
        f'{metrics["rmse_lakhs"]:.2f} L'
    )

    c.metric(
        "R² Score",
        f'{metrics["r2"]:.4f}'
    )

    st.divider()

    st.subheader("Understanding the Metrics")

    st.markdown("""
    **Mean Absolute Error (MAE)**

    The average absolute difference between predicted and actual
    prices. Lower values are better.

    **Root Mean Squared Error (RMSE)**

    Penalizes large prediction errors more strongly. Lower values
    are better.

    **R² Score**

    Measures how well the model explains price variation on the
    test set. A value closer to 1 generally indicates a better fit.
    """)

    st.info(
        "These metrics evaluate performance on synthetic data. "
        "They do not establish accuracy on real property transactions."
    )

# ==========================================
# PAGE 4: DATASET EXPLORER
# ==========================================

elif page == "📁 Dataset Explorer":

    st.title("📁 Dataset Explorer")

    st.write(
        "Inspect the training data used for this educational project."
    )

    city_filter = st.selectbox(
        "Filter by City",
        ["All Cities"] + cities
    )

    filtered_data = data.copy()

    if city_filter != "All Cities":

        filtered_data = filtered_data[
            filtered_data["City"] == city_filter
        ]

    st.metric(
        "Records Displayed",
        f"{len(filtered_data):,}"
    )

    st.dataframe(
        filtered_data.head(1000),
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Dataset",
        data=csv_data,
        file_name="house_price_data.csv",
        mime="text/csv"
    )

# ==========================================
# PAGE 5: ABOUT PROJECT
# ==========================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Smart Estate AI")

    st.markdown("""
    ### Project Objective

    Develop a machine learning application that estimates
    residential property prices based on property characteristics.

    ### Machine Learning Algorithm

    **Random Forest Regression**

    The algorithm combines multiple decision trees to generate
    a numerical prediction.

    ### Input Features

    - City and locality type
    - Property type
    - Bedrooms and bathrooms
    - Super area and carpet area
    - Property age and floor details
    - Furnishing and amenities
    - Distances to metro and city center

    ### Output

    Estimated property price in Indian rupees.

    ### Dataset

    The project uses a synthetic Indian urban property dataset
    for educational machine learning experiments.

    ### Limitations

    Synthetic prices may not reflect actual market conditions.
    A reliable real-world valuation system would require
    representative, verified property transaction data.
    """)

    st.success(
        "Smart Estate AI — House Price Prediction Mini Project"
    )

# ==========================================
# FOOTER
# ==========================================

st.sidebar.divider()

st.sidebar.caption(
    "Smart Estate AI | Educational ML Project"
)
