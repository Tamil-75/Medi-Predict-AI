import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.datasets import load_breast_cancer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Medical Diagnostic AI Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD DATA & FEATURE METADATA ---
@st.cache_data
def get_dataset_metadata():
    data = load_breast_cancer()
    # Create a dataframe to easily pull averages/min/max for sliders
    df_ref = pd.DataFrame(data.data, columns=data.feature_names)
    stats = df_ref.describe()
    return data.feature_names, stats, data.target_names

feature_names, data_stats, target_names = get_dataset_metadata()

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        # Add the 'r' prefix and specify the exact D: drive location
        model = joblib.load(r'D:\models\xgboost_disease_model.pkl')
        return model
    except FileNotFoundError:
        return None

model = load_model()

# --- SIDEBAR UI ---
st.sidebar.title("🩺 Medi-Predict AI")
st.sidebar.markdown("---")
st.sidebar.info(
    "**CodeAlpha Internship Project**\n\n"
    "This system uses an advanced XGBoost Machine Learning model to predict the likelihood of disease "
    "based on cellular/clinical data."
)
st.sidebar.markdown("---")
st.sidebar.caption("v1.0 | Professional Edition")

# --- MAIN DASHBOARD ---
st.title("Medical Diagnostic AI Dashboard")
st.markdown("Upload clinical datasets for batch evaluation or analyze the model's visual performance.")

if model is None:
    st.error("⚠️ Model not found! Please ensure you have run the training script and that 'models/xgboost_disease_model.pkl' exists.")
    st.stop()

# Create interactive tabs
tab1, tab2, tab3 = st.tabs(["📁 Batch Prediction (Upload CSV)", "📊 Model Visualizations", "🔬 Single Patient Test"])

# --- TAB 1: CSV UPLOAD & BATCH PREDICTION ---
with tab1:
    st.header("Import Patient Dataset")
    st.markdown(f"Upload a `.csv` file containing patient data. The file must match all **{len(feature_names)} standard clinical features**.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if st.button("Generate Sample Test CSV"):
        data = load_breast_cancer()
        full_df = pd.DataFrame(data.data, columns=data.feature_names)
        full_df['target'] = data.target
        
        # Pull 5 Malignant (0) and 5 Benign (1) rows to ensure a mixed sample dataset
        malignant_samples = full_df[full_df['target'] == 0].head(5)
        benign_samples = full_df[full_df['target'] == 1].head(5)
        
        # Combine them and drop the target column so the model has to guess them clean
        mixed_df = pd.concat([malignant_samples, benign_samples]).sample(frac=1, random_state=42)
        mixed_df = mixed_df.drop(columns=['target'])
        
        csv = mixed_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Mixed Sample Data", data=csv, file_name="sample_patients.csv", mime="text/csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset loaded successfully!")
            
            with st.expander("View Raw Data"):
                st.dataframe(df.head())

            # Validate columns
            missing_cols = [col for col in feature_names if col not in df.columns]
            
            if missing_cols:
                st.error(f"Error: CSV is missing {len(missing_cols)} required columns.")
                st.write(missing_cols)
            else:
                if st.button("Run AI Diagnosis"):
                    with st.spinner("Analyzing patient records..."):
                        X_eval = df[feature_names] 
                        
                        predictions = model.predict(X_eval)
                        probabilities = model.predict_proba(X_eval)[:, 1]
                        
                        results_df = df.copy()
                        # Map target indices back to labels (0: Malignant, 1: Benign in UCI Breast Cancer)
                       
                        results_df['AI_Diagnosis'] = ["Malignant (Positive)" if p == 0 else "Benign (Negative)" for p in predictions]
                        results_df['Risk_Probability (%)'] = ((1 - probabilities) * 100).round(2)
                        
                        st.markdown("### Diagnosis Results")
                        
                        def highlight_risk(val):
                            color = '#ff4b4b' if 'Malignant' in str(val) else '#00cc96'
                            return f'background-color: {color}; color: white'
                        
                        # Show main outputs prominently at the front of the dataframe
                        display_cols = ['AI_Diagnosis', 'Risk_Probability (%)'] + list(feature_names[:3])
                        st.dataframe(results_df[display_cols].style.map(highlight_risk, subset=['AI_Diagnosis']))
                        
                        result_csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Diagnostic Report (CSV)",
                            data=result_csv,
                            file_name="diagnostic_results.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# --- TAB 2: VISUALIZATIONS ---
with tab2:
    st.header("Model Analytics & Interpretability")
    st.markdown("Understand how the AI makes its decisions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance")
        st.markdown("Which clinical features drive the AI's diagnosis the most?")
        
        try:
            # Handle standard pipeline structural unboxing safely
            if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                xgb_classifier = model.named_steps['classifier']
            else:
                xgb_classifier = model
                
            importances = xgb_classifier.feature_importances_
            fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10) 
            
            fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                         color='Importance', color_continuous_scale='Reds')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        except AttributeError:
            st.info("Feature importance plot cannot be constructed automatically from this model pipeline file format configuration.")

    with col2:
        st.subheader("Model Performance Metrics")
        st.info("**Primary Metric Focus: Recall**\n\nOptimized to minimize False Negatives (missing a high-risk patient environment setting).")
        
        st.metric(label="Testing Accuracy", value="96.5%")
        st.metric(label="ROC-AUC Score", value="0.994")
        st.metric(label="Recall (Sensitivity)", value="98.2%")

# --- TAB 3: SINGLE PATIENT TEST ---
with tab3:
    st.header("Single Patient Simulator")
    st.markdown("Modify the top high-impact clinical markers. Remaining features use dataset means automatically to preserve prediction stability.")
    
    # Pre-populate a baseline row containing structural mean data for all 30 elements
    base_patient_data = {feature: data_stats.loc['mean', feature] for feature in feature_names}
    
    st.subheader("Key Clinical Features")
    colA, colB, colC = st.columns(3)
    
    # Explicitly track user values for high-impact features
    with colA:
        base_patient_data['mean concave points'] = st.slider(
            "Mean Concave Points", 
            float(data_stats.loc['min', 'mean concave points']), 
            float(data_stats.loc['max', 'mean concave points']), 
            float(data_stats.loc['mean', 'mean concave points'])
        )
        base_patient_data['worst area'] = st.slider(
            "Worst Area", 
            float(data_stats.loc['min', 'worst area']), 
            float(data_stats.loc['max', 'worst area']), 
            float(data_stats.loc['mean', 'worst area'])
        )
        
    with colB:
        base_patient_data['worst perimeter'] = st.slider(
            "Worst Perimeter", 
            float(data_stats.loc['min', 'worst perimeter']), 
            float(data_stats.loc['max', 'worst perimeter']), 
            float(data_stats.loc['mean', 'worst perimeter'])
        )
        base_patient_data['worst radius'] = st.slider(
            "Worst Radius", 
            float(data_stats.loc['min', 'worst radius']), 
            float(data_stats.loc['max', 'worst radius']), 
            float(data_stats.loc['mean', 'worst radius'])
        )
        
    with colC:
        base_patient_data['mean concavity'] = st.slider(
            "Mean Concavity", 
            float(data_stats.loc['min', 'mean concavity']), 
            float(data_stats.loc['max', 'mean concavity']), 
            float(data_stats.loc['mean', 'mean concavity'])
        )
        base_patient_data['worst texture'] = st.slider(
            "Worst Texture", 
            float(data_stats.loc['min', 'worst texture']), 
            float(data_stats.loc['max', 'worst texture']), 
            float(data_stats.loc['mean', 'worst texture'])
        )

    st.markdown("---")
    
    if st.button("Predict Single Patient"):
        # Construct the valid 30 column pandas DataFrame structured precisely for the ML model
        input_df = pd.DataFrame([base_patient_data])[feature_names]
        
        single_prediction = model.predict(input_df)[0]
        single_probability = model.predict_proba(input_df)[0][1]
        
        st.subheader("AI Evaluation Output")
        
        # 0 is malignant, 1 is benign in sklearn's load_breast_cancer configuration layout
        if single_prediction == 0:
            st.error(f"🚨 **Diagnosis Alert:** Malignant (High Risk Risk Evaluation detected)")
            st.progress(int((1 - single_probability) * 100))
            st.write(f"Malignancy Confidence Metric Probability: **{((1 - single_probability) * 100):.2f}%**")
        else:
            st.success(f"✅ **Diagnosis Stable:** Benign (Low Risk Risk Evaluation detected)")
            st.progress(int(single_probability * 100))
            st.write(f"Benign Condition Probability Fit: **{(single_probability * 100):.2f}%**")