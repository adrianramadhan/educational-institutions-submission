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
@st.cache(allow_output_mutation=True)
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

# Mapping dicts for dropdown labels
mappings = {
    'Marital_status': {1:'Single',2:'Married',3:'Widower',4:'Divorced',5:'Union',6:'Separated'},
    
    'Application_mode': {
        1:'1st-general', 2:'Ordinance612',5:'Special-Azores',7:'Other-courses',
        10:'Ordinance854',15:'International',16:'Special-Madeira',17:'2nd-general',
        18:'3rd-general',26:'Ordinance533-b2',27:'Ordinance533-b3',39:'Over23',
        42:'Transfer',43:'Change-course',44:'Tech-diploma',51:'Change-institution',
        53:'Short-cycle',57:'Change-international'
    },
    
    'Course': {
        33: 'Biofuel-Tech', 171: 'Animation-Multimedia', 8014: 'Social-Service-Evening',
        9003: 'Agronomy',9070: 'Communication-Design',9085: 'Vet-Nursing',
        9119: 'Informatics-Eng',9130: 'Equinculture',9147: 'Management',
        9238: 'Social-Service',9254: 'Tourism',9500: 'Nursing',
        9556: 'Oral-Hygiene',9670: 'Advertising-Marketing',9773: 'Journalism-Comms',
        9853: 'Basic-Ed',9991: 'Management-Evening'
    },
    
    'Daytime_evening_attendance': {1:'Daytime',0:'Evening'},
    
    'Previous_qualification': {
        1:'Secondary',2:"Bachelor",3:'Degree',4:"Master",5:'Doctorate',
        6:'Frequency',9:'12th-incomplete',10:'11th-incomplete',12:'Other-11th',
        14:'10th',15:'10th-incomplete',19:'Basic-3rd',38:'Basic-2nd',
        39:'Tech-spec',40:'Higher-1st',42:'Professional',43:'Master-2nd'
    },
    
    'Nacionality': {
        1: 'PT',2: 'DE',6: 'ES',11: 'IT',13: 'NL',14: 'UK',17: 'LT',21: 'AO',
        22: 'CV',24: 'GN',25: 'MZ',26: 'ST',32: 'TR',41: 'BR',62: 'RO',
        100: 'MD',101: 'MX',103: 'UA',105: 'RU',108: 'CU',109: 'CO'
    },
    
    'Mothers_qualification': {
        1: 'Sec-Ed',2:'Higher-Bachelor',3:'Higher-Degree',4:'Higher-Master',
        5:'Higher-PhD',6:'Freq-Higher',9:'12th-incomplete',10:'11th-incomplete',
        12:'Other-11th',14:'10th',18:'Commerce',19:'Basic-3rd',22:'Tech-course',
        26:'7th',27:'HS-2nd-cycle',29:'9th-incomplete',30:'8th',34:'Unknown',
        35:'Illiterate',36:'Basic-read',37:'Basic-1st',38:'Basic-2nd',
        39:'Tech-spec',40:'Higher-1st',41:'Specialized',42:'Prof-course',
        43:'Master',44:'PhD'
    },
    
    'Fathers_qualification': {
        1: 'Sec-Ed',2:'Higher-Bachelor',3:'Higher-Degree',4:'Higher-Master',
        5:'Higher-PhD',6:'Freq-Higher',9:'12th-incomplete',10:'11th-incomplete',
        12:'Other-11th',13:'HS-2nd',14:'10th',18:'Commerce',19:'Basic-3rd',
        20:'HS-Complementary',22:'Tech-course',25:'HS-Incomplete',26:'7th',
        27:'HS-2nd-cycle',29:'9th-incomplete',30:'8th',31:'Admin-Commerce',
        33:'Accounting',34:'Unknown',35:'Illiterate',36:'Basic-read',
        37:'Basic-1st',38:'Basic-2nd',39:'Tech-spec',40:'Higher-1st',
        41:'Specialized',42:'Prof-course',43:'Master',44:'PhD'
    },
    
    'Mothers_occupation': {
        0: 'Student',1: 'Reps-Legislative',2: 'Science-Specialist',3: 'Mid-Tech',
        4: 'Admin',5: 'Service-Security',6: 'Agri-Forestry',7: 'Industry',
        8: 'Machine-Ops',9: 'Unskilled',10: 'Military',90: 'Other',99: 'Blank',
        122: 'Health',123: 'Teacher',125: 'ICT-Specialist',131: 'Eng-Tech',
        132: 'Health-Tech',134: 'Legal-Social',141: 'Office',143: 'Finance-Ops',
        144: 'Admin-Support',151: 'Service',152: 'Sales',153: 'Care-Worker',
        171: 'Construction',173: 'Craftsman',175: 'Manufacturing',
        191: 'Cleaner',192: 'Agri-Unskilled',193: 'Industry-Unskilled',
        194: 'Kitchen'
    },
    
    'Fathers_occupation': {
        0: 'Student',1: 'Reps-Legislative',2: 'Science-Specialist',3: 'Mid-Tech',
        4: 'Admin',5: 'Service-Security',6: 'Agri-Forestry',7: 'Industry',
        8: 'Machine-Ops',9: 'Unskilled',10: 'Military',90: 'Other',99: 'Blank',
        101: 'Military-Officer',102: 'Military-Sergeant',103: 'Military-Other',
        112: 'Admin-Director',114: 'Hospitality-Director',121: 'STEM-Specialist',
        122: 'Health',123: 'Teacher',124: 'Finance-Specialist',131: 'Eng-Tech',
        132: 'Health-Tech',134: 'Legal-Social',135: 'ICT-Tech',141: 'Office',
        143: 'Finance-Ops',144: 'Admin-Support',151: 'Service',152: 'Sales',
        153: 'Care-Worker',154: 'Security',161: 'Commercial-Farmer',
        163: 'Subsistence-Farmer',171: 'Construction',172: 'Metal-Worker',
        174: 'Electrician',175: 'Manufacturing',181: 'Plant-Ops',
        182: 'Assembler',183: 'Driver',192: 'Agri-Unskilled',
        193: 'Industry-Unskilled',194: 'Kitchen'
    },
    
    'Displaced': {0:'No',1:'Yes'},
    'Educational_special_needs': {0:'No',1:'Yes'},
    'Debtor': {0:'No',1:'Yes'},
    'Tuition_fees_up_to_date': {0:'No',1:'Yes'},
    'Gender': {0:'Female',1:'Male'},
    'Scholarship_holder': {0:'No',1:'Yes'},
    'International': {0:'No',1:'Yes'}
}
# Function to render dropdown or numeric
def render_input():
    inputs = {}
    for col in feature_cols:
        if col in mappings and mappings[col]:
            options = list(mappings[col].keys())
            labels = [mappings[col][k] for k in options]
            sel = st.sidebar.selectbox(col, options, format_func=lambda x: mappings[col][x])
            inputs[col] = sel
        elif col in ['Previous_qualification_grade','Admission_grade',
                     'Curricular_units_1st_sem_grade','Curricular_units_2nd_sem_grade',
                     'Unemployment_rate','Inflation_rate','GDP']:
            inputs[col] = st.sidebar.number_input(col, value=0.0)
        else:
            inputs[col] = st.sidebar.number_input(col, value=0)
    return pd.DataFrame([inputs])

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
        styled = data[['Status','Predicted_Status']].style.format({'Predicted_Status': lambda v: f"**{v}**"}, escape=False)
        st.dataframe(styled)
else:
    st.sidebar.subheader('Manual Entry')
    input_df = render_input()
    # compute engineered
    input_df['avg_sem_grade'] = (input_df['Curricular_units_1st_sem_grade'] + input_df['Curricular_units_2nd_sem_grade']) / 2
    input_df['total_units_approved'] = input_df['Curricular_units_1st_sem_approved'] + input_df['Curricular_units_2nd_sem_approved']
    st.subheader('Input Data')
    st.write(input_df)
    preds_int = model.predict(input_df[all_features])
    preds_label = le.inverse_transform(preds_int)
    st.subheader('Prediction')
    # color-coded label
    color = 'red' if preds_label[0]=='Dropout' else ('green' if preds_label[0]=='Graduate' else 'blue')
    st.markdown(f"<span style='color:{color}; font-size:24px; font-weight:bold'>{preds_label[0]}</span>", unsafe_allow_html=True)

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
