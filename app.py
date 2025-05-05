# streamlit_app.py
import streamlit as st

# **set_page_config harus baris pertama setelah import streamlit**
st.set_page_config(page_title='Dropout Risk Predictor', layout='wide')

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load trained model pipeline
# @st.cache(allow_output_mutation=True)
def load_model(path='./model/best_model.joblib'):
    return joblib.load(path)

model = load_model()

le = LabelEncoder()
le.classes_ = np.array(['Dropout', 'Enrolled', 'Graduate'])

st.title('📊 Student Dropout Risk Predictor')
st.markdown(
    """
    Aplikasi ini memprediksi risiko mahasiswa **Dropout**, **Graduate**, atau **Enrolled**
    berdasarkan data akademik dan demografis.
    """
)

# Sidebar: choose input mode
st.sidebar.header('Input Options')
input_mode = st.sidebar.selectbox('Pilih mode input:', ['Upload CSV', 'Manual Entry'])

# Define base features
feature_cols = [
    'Marital_status','Application_mode','Application_order','Course',
    'Daytime_evening_attendance','Previous_qualification','Previous_qualification_grade',
    'Nacionality','Mothers_qualification','Fathers_qualification',
    'Mothers_occupation','Fathers_occupation','Admission_grade','Displaced',
    'Educational_special_needs','Debtor','Tuition_fees_up_to_date','Gender',
    'Scholarship_holder','Age_at_enrollment','International',
    'Curricular_units_1st_sem_credited','Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations','Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade','Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited','Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations','Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade','Curricular_units_2nd_sem_without_evaluations',
    'Unemployment_rate','Inflation_rate','GDP'
]
# Engineered features
engineered_cols = ['avg_sem_grade','total_units_approved']
all_features = feature_cols + engineered_cols

# helper to color labels
def color_label(val):
    color = 'red' if val=='Dropout' else ('green' if val=='Graduate' else 'blue')
    return f"<span style='color:{color}; font-weight:bold'>{val}</span>"

if input_mode == 'Upload CSV':
    uploaded_file = st.sidebar.file_uploader('Upload file CSV', type='csv')
    if uploaded_file:
        data = pd.read_csv(uploaded_file, sep=';')
        # compute engineered features
        data['avg_sem_grade'] = (data['Curricular_units_1st_sem_grade'] + data['Curricular_units_2nd_sem_grade']) / 2
        data['total_units_approved'] = data['Curricular_units_1st_sem_approved'] + data['Curricular_units_2nd_sem_approved']
        st.subheader('Data Preview')
        st.dataframe(data.head())
        X = data[all_features]
        preds_int = model.predict(X)
        preds_label = le.inverse_transform(preds_int)
        data['Predicted_Status'] = preds_label
        st.subheader('Prediction Results')
        styled = data[['Status','Predicted_Status']].style.format({'Predicted_Status': color_label}, escape=False)
        st.dataframe(styled)
else:
    st.sidebar.subheader('Manual Input')
    # Collect inputs
    def user_input():
        d = {}
        for col in feature_cols:
            # numeric vs categorical detection
            if col in ['Previous_qualification_grade','Admission_grade',
                       'Curricular_units_1st_sem_grade','Curricular_units_2nd_sem_grade',
                       'Unemployment_rate','Inflation_rate','GDP']:
                d[col] = st.sidebar.number_input(col, value=0.0)
            else:
                d[col] = st.sidebar.number_input(col, value=0)
        return pd.DataFrame([d])
    input_df = user_input()
    # compute engineered features
    input_df['avg_sem_grade'] = (input_df['Curricular_units_1st_sem_grade'] + input_df['Curricular_units_2nd_sem_grade']) / 2
    input_df['total_units_approved'] = input_df['Curricular_units_1st_sem_approved'] + input_df['Curricular_units_2nd_sem_approved']
    st.subheader('Input Data')
    st.write(input_df)
    # predict using all features
    preds_int = model.predict(input_df[all_features])
    preds_label = le.inverse_transform(preds_int)
    st.subheader('Prediction')
    st.markdown(color_label(preds_label[0]), unsafe_allow_html=True)

# EDA Section
st.markdown('---')
col1, col2 = st.columns(2)
with col1:
    st.subheader('Class Balance')
    df_full = pd.read_csv('./data.csv', sep=';')
    counts = df_full['Status'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_ylabel('Count')
    st.pyplot(fig)
with col2:
    st.subheader('Correlation Matrix')
    num_df = df_full.select_dtypes(include=['int64','float64'])
    corr = num_df.corr()
    fig2, ax2 = plt.subplots(figsize=(5,4))
    sns.heatmap(corr, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

st.markdown('---')
st.caption('Prototype Streamlit untuk memonitor risiko dropout mahasiswa secara interaktif.')
