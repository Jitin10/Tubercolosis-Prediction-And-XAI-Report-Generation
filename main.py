import streamlit as st
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- MODIFIED CSS: Grey Background + White/Blue Sidebar Dropdown ---
css_styles = """
<style>

/* --- Main Background (MODIFIED) --- */
[data-testid="stAppViewContainer"] > .main {
    /* Changed to a soft grey matching the example image */
    background: #F0F2F5; 
    color: #0D1B2A;
    font-family: 'Poppins', sans-serif;
}

/* --- Sidebar Background --- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D47A1, #1565C0, #1E88E5);
    color: white;
    border-right: 2px solid #0D47A1;
}

/* --- Sidebar Text --- */
[data-testid="stSidebar"] * {
    color: black !important;
    font-weight: 500;
}
[data-testid="stSidebar"] label {
    color: #FFE082 !important;
    font-weight: 600;
}

/* --- “Choose a Feature” Dropdown (MODIFIED) --- */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #FFFFF;
    color: black !important; /* Dark blue text */
    border-radius: 8px !important;
    border: 2px solid #BBDEFB !important; /* Light blue border */
    font-weight: 600 !important;
    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.15) inset;
}

/* Remove inner white box effect */
[data-testid="stSidebar"] [data-baseweb="select"] > div > div {
    background: transparent !important;
}

/* Selected text (MODIFIED) */
[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #0D47A1 !important; /* Dark blue text */
}

/* Dropdown options menu (MODIFIED) */
[data-baseweb="popover"] {
    background: #FFFFFF !important; /* White background */
    border-radius: 8px !important;
    color: #0D47A1 !important; /* Dark blue text */
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}

/* Style for the options on hover (NEW) */
[data-baseweb="popover"] li:hover {
    background-color: #E3F2FD !important; /* Light blue hover */
    color: #0D47A1 !important;
}

/* --- Headings --- */
h1 {
    color: #0D47A1;
    font-weight: 700;
    text-align: center;
}
h2 {
    color: #1565C0;
    border-bottom: 3px solid #BBDEFB;
    padding-bottom: 6px;
}
h3, h4 {
    color: #1E88E5;
}

/* --- Buttons --- */
.stButton>button {
    background: linear-gradient(90deg, #1976D2, #42A5F5);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 4px 10px rgba(25, 118, 210, 0.3);
}
.stButton>button:hover {
    background: linear-gradient(90deg, #1565C0, #1E88E5);
    transform: scale(1.03);
}

/* --- Inputs --- */
[data-testid="stSelectbox"] > div, 
[data-testid="stNumberInput"] > div, 
[data-testid="stTextInput"] > div, 
[data-testid="stSlider"] {
    background-color: white !important;
    border-radius: 10px !important;
    padding: 8px;
    border: 1px solid #BBDEFB !important;
}

/* --- Report Cards --- */
.report-card {
    background-color: #FFFFFF;
    border-radius: 16px;
    padding: 25px;
    margin-top: 20px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
    border: 1px solid #E3F2FD;
}

/* --- DataFrames --- */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #BBDEFB;
    background-color: white;
}

/* --- Footer/Deploy Cleanup --- */
footer, .viewerBadge_container__1QSob, .stDeployButton {
    display: none !important;
}

</style>
"""

st.markdown(css_styles, unsafe_allow_html=True)

# --- End of CSS ---


# Define class names globally
class_names = ['Normal', 'Tuberculosis']

# ==============================================================================
# Model Training and Data Loading (Cached)
# ==============================================================================
# Use st.cache_resource to load and train the model only once
@st.cache_resource
def load_model_and_data():
    """
    Loads data, trains model, fits encoder, and pre-calculates averages.
    Returns all necessary objects for the app.
    """
    data_filename = 'tuberculosis_dataset.csv'
    try:
        # --- MODIFICATION 3: Use relative file path ---
        df_full = pd.read_csv(data_filename)
    except FileNotFoundError:
        # Update error message to match
        st.error(f"Error: '{data_filename}' not found. Please make sure it's in the same folder as this app.")
        st.stop()

    # --- 1. Preprocessing ---
    df = df_full.drop('Patient_ID', axis=1, errors='ignore')
    X = df.drop('Class', axis=1)
    y = df['Class']
    y_encoded = y.apply(lambda x: 1 if x == 'Tuberculosis' else 0)
    
    X_train = X.copy()
    y_train = y_encoded.copy()
    
    # --- 4. Encode Categorical Features ---
    categorical_cols = X.select_dtypes(include=['object']).columns
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
    
    # --- 5. Train the Model ---
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # --- 6. Get Feature Importances ---
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # --- MODIFICATION 1: Change from 3 to 5 ---
    top_features = feature_importance_df.head(5)['Feature'].tolist()

    # --- 7. Pre-calculate Averages for Top Features ---
    # We now get averages for ALL numerical features, but we'll still
    # pass the numerical_top_features list to the chart function.
    numerical_top_features = [f for f in top_features if f in X.select_dtypes(include=np.number).columns]
    
    # Get averages for all numerical features in top_features
    if numerical_top_features:
        averages_df = df_full.groupby('Class')[numerical_top_features].mean().reset_index()
    else:
        averages_df = pd.DataFrame(columns=['Class'] + numerical_top_features)

    return model, encoder, top_features, df_full, categorical_cols, X.columns, averages_df, numerical_top_features

# ==============================================================================
# Natural Language Generation (NLG) Function
# ==============================================================================
def generate_explanation_rf(patient_original_features, predicted_class_name, predicted_probability, top_features_list, averages_df, numerical_top_features):
    """
    Generates a human-readable explanation for a single prediction
    using GLOBAL feature importances.
    """
    try:
        # 1. Build the main prediction string
        final_report = f"**Prediction Report:**\n"
        final_report += f"The model predicts **{predicted_class_name}** with **{predicted_probability * 100:.1f}%** confidence.\n\n"
        
        # --- MODIFICATION 2: Make the list dynamic ---
        final_report += f"**Model's Key Factors:**\n"
        final_report += (f"This model *globally* finds these features most important for its decisions:\n")
        
        # Create a clean, numbered list of features
        top_features_clean = [f.replace("_", " ") for f in top_features_list]
        for i, feature in enumerate(top_features_clean):
            final_report += f"{i+1}. {feature}\n"
        
        final_report += "\n" # Add a space before the next section
        
        # 3. Build the "local data" string
        final_report += f"**Analysis of Key Factors for This Patient:**\n"
        final_report += "Here is a breakdown of this patient's data for those key factors:\n\n"
        
        explanation_parts = []
        for feature in top_features_list:
            value = patient_original_features[feature]
            feature_name_clean = feature.replace("_", " ")
            
            if isinstance(value, float):
                formatted_value = f"**{value:.2f}**"
            else:
                formatted_value = f"**{value}**"
            
            explanation_parts.append(f"- **{feature_name_clean}:** The patient's reported value is {formatted_value}.")
        
        final_report += "\n".join(explanation_parts) + "\n"
        
        # --- NEW: Risk Analysis Section ---
        final_report += "\n**Risk Analysis based on Key Factors:**\n"
        
        if not numerical_top_features: # Check if list is empty
            final_report += "No numerical key factors were identified for a direct risk comparison.\n"
        else:
            try:
                avg_tb = averages_df[averages_df['Class'] == 'Tuberculosis'].iloc[0]
                
                risk_statements = []
                # Only analyze numerical features for this comparison
                for feature in numerical_top_features:
                    patient_value = patient_original_features[feature]
                    tb_avg = avg_tb[feature]
                    feature_clean = feature.replace("_", " ")

                    # Define a simple threshold (e.g., 90% of the TB average)
                    # This logic assumes "higher is riskier" which is true for our modified features
                    risk_threshold = tb_avg * 0.9 

                    if patient_value >= risk_threshold:
                        risk_statements.append(f"- The patient's **{feature_clean}** (value: **{patient_value:.2f}**) is **near or above** the average for Tuberculosis patients (Avg: {tb_avg:.2f}), indicating a **potential risk factor**.")
                    else:
                        risk_statements.append(f"- The patient's **{feature_clean}** (value: **{patient_value:.2f}**) is **below** the average for Tuberculosis patients (Avg: {tb_avg:.2f}), indicating a **lower risk** in this area.")
                
                final_report += "\n".join(risk_statements) + "\n"
            except IndexError:
                final_report += "Could not retrieve average TB data for risk analysis.\n"
            except Exception as e:
                final_report += f"An error occurred during risk analysis: {e}\n"

        return final_report
    
    except Exception as e:
        return f"An error occurred during report generation: {e}"

# ==============================================================================
# MODIFIED: Chart Generation Function
# ==============================================================================
def create_comparison_charts(patient_features, averages_df, numerical_top_features):
    """
    Creates multiple Altair bar charts (one for each feature) and displays them vertically.
    """
    
    if not numerical_top_features:
        st.warning("No numerical key factors to display in charts.")
        return

    try:
        avg_normal = averages_df[averages_df['Class'] == 'Normal'].iloc[0]
        avg_tb = averages_df[averages_df['Class'] == 'Tuberculosis'].iloc[0]
    except IndexError:
        st.error("Error: Could not retrieve average data for chart generation.")
        return

    # Loop through each feature and create a *separate* chart
    for feature in numerical_top_features:
        feature_clean = feature.replace("_", " ")
        
        # 1. Create data for *this specific feature's* chart
        chart_data = [
            {'Type': 'Average (Normal)', 'Value': avg_normal[feature]},
            {'Type': 'Average (TB)', 'Value': avg_tb[feature]},
            {'Type': 'This Patient', 'Value': patient_features[feature]}
        ]
        
        chart_df = pd.DataFrame(chart_data)

        # 2. Create the Altair chart with an independent Y-axis
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X('Type:N', title=None, axis=None), # No x-axis labels, rely on legend
            y=alt.Y('Value:Q', title=f'Value'), # Independent Y-axis
            color=alt.Color('Type:N', 
                            legend=alt.Legend(title="Comparison"),
                            scale=alt.Scale(domain=['This Patient', 'Average (TB)', 'Average (Normal)'],
                                            range=['#0068C9', '#FF6B6B', '#6BCCB4']) # Custom colors
                           ),
            tooltip=['Type', 'Value']
        ).properties(
            title=f'Comparison for: {feature_clean}' # Chart title
        ).interactive()
        
        # 3. Display the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)

# ==============================================================================
# Streamlit Frontend
# ==============================================================================

# --- 1. Load Model and Data ---
model, encoder, top_features, df_full, cat_cols, feature_order, averages_df, num_top_features = load_model_and_data()

# --- 2. App Title ---
st.title("Tuberculosis Prediction & XAI Report Generator 🩺")
st.write("This app uses a Machine Learning model to predict Tuberculosis and explain its reasoning.")

# --- 3. Sidebar Navigation ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a feature", 
                                ["Predict for New Patient", "Find Patient by ID"])

# ==============================================================================
# FEATURE 1: Predict for New Patient
# ==============================================================================
if app_mode == "Predict for New Patient":
    st.header("New Patient Prediction")
    st.write("Enter the patient's details below to get a prediction.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", options=df_full['Gender'].unique())
        chest_pain = st.selectbox("Chest Pain", options=df_full['Chest_Pain'].unique())
        cough_severity = st.slider("Cough Severity", min_value=0, max_value=10, value=3)
        breathlessness = st.slider("Breathlessness", min_value=0, max_value=10, value=2)
        fatigue = st.slider("Fatigue", min_value=0, max_value=10, value=5)
    
    with col2:
        weight_loss = st.number_input("Weight Loss (in kg)", min_value=0.0, max_value=30.0, value=5.0, format="%.2f")
        fever = st.selectbox("Fever", options=df_full['Fever'].unique())
        night_sweats = st.selectbox("Night Sweats", options=df_full['Night_Sweats'].unique())
        sputum_prod = st.selectbox("Sputum Production", options=df_full['Sputum_Production'].unique())
        blood_in_sputum = st.selectbox("Blood in Sputum", options=df_full['Blood_in_Sputum'].unique())
        smoking = st.selectbox("Smoking History", options=df_full['Smoking_History'].unique())
        prev_tb = st.selectbox("Previous TB History", options=df_full['Previous_TB_History'].unique())

    # --- Prediction Logic ---
    if st.button("Predict Diagnosis"):
        
        patient_original_features = {
            'Age': age, 'Gender': gender, 'Chest_Pain': chest_pain,
            'Cough_Severity': cough_severity, 'Breathlessness': breathlessness,
            'Fatigue': fatigue, 'Weight_Loss': weight_loss, 'Fever': fever,
            'Night_Sweats': night_sweats, 'Sputum_Production': sputum_prod,
            'Blood_in_Sputum': blood_in_sputum, 'Smoking_History': smoking,
            'Previous_TB_History': prev_tb
        }
        
        X_new = pd.DataFrame([patient_original_features], columns=feature_order)
        X_new_encoded = X_new.copy()
        X_new_encoded[cat_cols] = encoder.transform(X_new[cat_cols])
        
        pred_encoded = model.predict(X_new_encoded)[0]
        pred_name = class_names[pred_encoded]
        prob = model.predict_proba(X_new_encoded)[0][pred_encoded]
        
        # --- Generate Report (Wrapped in a card) ---
        with st.container():
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            
            st.subheader("Prediction Complete")
            # --- MODIFICATION: Pass new arguments ---
            report = generate_explanation_rf(
                patient_original_features, pred_name, prob, top_features, averages_df, num_top_features
            )
            st.markdown(report)
            
            st.markdown("---")
            st.markdown("#### Key Factor Comparison Charts")
            st.write("These charts compare the patient's data to the average values for 'Normal' and 'Tuberculosis' patients. Each chart has its own scale.")
            create_comparison_charts(patient_original_features, averages_df, num_top_features)
            
            st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# FEATURE 2: Find Patient by ID
# ==============================================================================
elif app_mode == "Find Patient by ID":
    st.header("Patient Report Generator")
    
    patient_id_input = st.text_input("Enter Patient ID (e.g., PID000001)")
    
    if st.button("Generate Report"):
        if patient_id_input:
            
            # --- MODIFICATION 4: Add the .strip() fix ---
            sanitized_id = patient_id_input.strip()
            
            try:
                # Find patient using the sanitized ID
                patient_data = df_full[df_full['Patient_ID'] == sanitized_id].iloc[0]
                
                # Display the sanitized ID
                st.subheader(f"Report for {sanitized_id}")
                st.write("---")
                
                st.markdown("#### Patient's Original Data")
                st.dataframe(patient_data)
                
                patient_original_features = patient_data.drop(['Patient_ID', 'Class'])
                X_patient = pd.DataFrame([patient_original_features], columns=feature_order)
                X_patient_encoded = X_patient.copy()
                X_patient_encoded[cat_cols] = encoder.transform(X_patient[cat_cols])
                
                pred_encoded = model.predict(X_patient_encoded)[0]
                pred_name = class_names[pred_encoded]
                prob = model.predict_proba(X_patient_encoded)[0][pred_encoded]

                # --- Generate Report (Wrapped in a card) ---
                with st.container():
                    st.markdown('<div class="report-card">', unsafe_allow_html=True)
                    
                    st.write("---")
                    st.markdown("#### Prediction and Explanation")
                    # --- MODIFICATION: Pass new arguments ---
                    report = generate_explanation_rf(
                        patient_original_features, pred_name, prob, top_features, averages_df, num_top_features
                    )
                    st.markdown(report)
                    
                    st.markdown("---")
                    st.markdown("#### Key Factor Comparison Charts")
                    st.write("These charts compare the patient's data to the average values for 'Normal' and 'Tuberculosis' patients. Each chart has its own scale.")
                    create_comparison_charts(patient_original_features, averages_df, num_top_features)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

            except IndexError:
                # Use the sanitized ID in the error
                st.error(f"Patient ID '{sanitized_id}' not found. Please check the ID and try again.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a Patient ID.")

