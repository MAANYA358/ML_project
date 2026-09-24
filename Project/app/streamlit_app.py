import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="💰",
    layout="centered"
)
@st.cache_resource
def load_model():
    model = joblib.load('models/insurance_model.pkl')
    columns = joblib.load('models/model_columns.pkl')
    return model, columns

model, model_columns = load_model()
st.title("💰 Medical Insurance Cost Predictor")
st.markdown("Estimate your medical insurance expenses based on personal health and demographic details.")

with st.expander("ℹ️ About this project"):
    st.write("""
    This app predicts medical insurance costs using a **Multiple Linear Regression** model 
    trained on real-world insurance data. It considers factors like age, BMI, smoking status, 
    number of children, and region to estimate expected annual insurance charges.

    **Tech stack:** Python, scikit-learn, pandas, Streamlit  
    **Model:** Linear Regression  
    **Dataset:** 1,338 records with age, sex, BMI, children, smoker status, and region
    """)
    st.subheader("Enter your details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=64, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=55.0, value=25.0, step=0.1)
    children = st.slider("Number of Children", min_value=0, max_value=5, value=0)

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

st.markdown("---")

if st.button("🔮 Predict Insurance Cost", use_container_width=True):
    
#data frame
    input_dict = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if sex == 'male' else 0,
        'smoker_yes': 1 if smoker == 'yes' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0,
    }
    
    input_df = pd.DataFrame([input_dict])

    input_df = input_df[model_columns]
    

    prediction = model.predict(input_df)[0]
    
#result
    st.success(f"### Estimated Insurance Cost: ${prediction:,.2f}")

st.markdown("---")
st.caption("Built with Streamlit · Model: Linear Regression · Dataset: Medical Insurance Cost dataset")