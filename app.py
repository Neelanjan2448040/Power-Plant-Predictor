import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Power Plant Energy Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- STYLING ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #0c2d5e;
        font-weight: bold;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #0c2d5e;
        border-radius:10px;
        border:0;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #008080;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# --- DATA LOADING AND PREPROCESSING ---
@st.cache_data
def load_and_prep_data():
    """Loads and splits the dataset for evaluation."""
    try:
        df = pd.read_excel("Folds5x2_pp.xlsx")
        X = df.drop(['PE'], axis=1)
        y = df['PE']
        # We only need the test set to evaluate the pre-trained model
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return df, X_test, y_test
    except FileNotFoundError:
        st.error("Could not find 'Folds5x2_pp.xlsx'. Please ensure it's in your GitHub repository.")
        return None, None, None

df, X_test, y_test = load_and_prep_data()

# --- MODEL LOADING CACHE ---
@st.cache_resource
def load_model_and_scaler():
    """Loads the pre-trained model and scaler from joblib files."""
    try:
        model = joblib.load("best_power_plant_model.joblib")
        scaler = joblib.load("scaler.joblib")
        return model, scaler
    except FileNotFoundError:
        st.error("Could not find 'best_power_plant_model.joblib' or 'scaler.joblib'. Please ensure they are in your GitHub repository.")
        return None, None

model, scaler = load_model_and_scaler()

# --- HOME PAGE ---
def home_page():
    st.title("⚡ Power Plant Energy Output Prediction")
    st.markdown("A Machine Learning project to predict the net hourly electrical energy output of a Combined Cycle Power Plant based on hourly average ambient variables.")
    
    st.header("Dataset Description")
    st.write("The dataset contains 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011), when the power plant was set to work with full load. Features consist of hourly average ambient variables Temperature (AT), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust Vacuum (V) to predict the net hourly electrical energy output (PE) of the plant.")
    if df is not None:
        st.dataframe(df.describe())

    st.header("Data Visualizations")
    tab1, tab2, tab3 = st.tabs(["Variable Distributions", "Correlation Heatmap", "Pairplot Relationships"])

    with tab1:
        st.subheader("Distribution of Each Variable")
        if df is not None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            sns.histplot(df['AT'], ax=axes[0, 0], kde=True, color='skyblue').set_title('Temperature (AT)')
            sns.histplot(df['V'], ax=axes[0, 1], kde=True, color='salmon').set_title('Exhaust Vacuum (V)')
            sns.histplot(df['AP'], ax=axes[1, 0], kde=True, color='lightgreen').set_title('Ambient Pressure (AP)')
            sns.histplot(df['RH'], ax=axes[1, 1], kde=True, color='gold').set_title('Relative Humidity (RH)')
            plt.tight_layout()
            st.pyplot(fig)

    with tab2:
        st.subheader("Correlation Between Variables")
        if df is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt='.2f', ax=ax)
            st.pyplot(fig)
            
    with tab3:
        st.subheader("Pairwise Relationships Between Variables")
        if df is not None:
            with st.spinner("Generating pairplot... This may take a moment."):
                pairplot_fig = sns.pairplot(df)
                st.pyplot(pairplot_fig)


# --- MODEL PERFORMANCE PAGE (MODIFIED FOR EFFICIENCY) ---
def model_performance_page():
    st.title("🤖 Best Model Performance")
    st.write("This page shows the performance of the best pre-trained model (Tuned XGBoost) on the test data.")
    
    if model is None or scaler is None or X_test is None or y_test is None:
        st.error("Model, scaler, or test data could not be loaded. The app cannot display performance.")
        return

    # Scale the test data and make predictions
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    # --- Display Results ---
    st.subheader("Results for: Best Model (Tuned XGBoost)")
    
    # Create and display the predictions table with the difference
    st.subheader("Predictions Table")
    predictions_df = pd.DataFrame({
        'Actual Energy Output': y_test,
        'Predicted Energy Output': y_pred
    }).round(2)
    predictions_df['Difference'] = (predictions_df['Actual Energy Output'] - predictions_df['Predicted Energy Output']).round(2)
    st.dataframe(predictions_df)
        
    # Display the plots below the table
    st.subheader("Performance Plots")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Actual vs. Predicted')
        ax1.grid(True)
        st.pyplot(fig1)
    
    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        residuals = y_test - y_pred
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.set_xlabel('Residuals (Actual - Predicted)')
        ax2.set_title('Residuals Distribution')
        ax2.grid(True)
        st.pyplot(fig2)


# --- PREDICTION PAGE ---
def prediction_page():
    st.title("💡 Make a Prediction")
    st.write("Enter the values for the following features to predict the energy output using the best model.")
    
    if model is None or scaler is None:
        st.error("Prediction model or scaler could not be loaded.")
        return

    col1, col2 = st.columns(2)
    with col1:
        at_val = st.number_input("Ambient Temperature (AT)", min_value=0.0, max_value=40.0, value=25.0, step=0.1)
        v_val = st.number_input("Exhaust Vacuum (V)", min_value=25.0, max_value=85.0, value=54.0, step=0.1)
    with col2:
        ap_val = st.number_input("Ambient Pressure (AP)", min_value=990.0, max_value=1040.0, value=1013.0, step=0.1)
        rh_val = st.number_input("Relative Humidity (RH)", min_value=25.0, max_value=100.0, value=73.0, step=0.1)

    if st.button("Predict Energy Output"):
        input_data = pd.DataFrame({'AT': [at_val], 'V': [v_val], 'AP': [ap_val], 'RH': [rh_val]})
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        predicted_value = prediction[0]

        st.subheader("Prediction Result")
        
        plot_col, _ = st.columns([0.6, 0.4])
        with plot_col:
            max_output = df['PE'].max()
            remainder = max_output - predicted_value if max_output > predicted_value else 0
            
            labels = [f'Predicted Output\n({predicted_value:.2f} MW)', 'Potential to Max']
            sizes = [predicted_value, remainder]
            colors = ['#0c2d5e', '#d3d3d3']
            explode = (0.1, 0)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                   shadow=True, startangle=90, colors=colors)
            ax.axis('equal')
            
            st.pyplot(fig)


# --- MAIN APP LOGIC ---
st.sidebar.title("Navigation")
page_options = ["Home", "Model Performance", "Prediction"]
page = st.sidebar.radio("Choose a page:", page_options)

if df is not None:
    if page == "Home":
        home_page()
    elif page == "Model Performance":
        model_performance_page()
    elif page == "Prediction":
        prediction_page()

