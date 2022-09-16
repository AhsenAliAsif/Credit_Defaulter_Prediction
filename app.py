import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import logging




st.title('Credit Default Prediction Application')
st.text('Data Source:Kaggle and Project of ineuron.ai')
st.markdown('This application is used to predict if a customer will default on a loan or not')
df=pd.read_csv('UCI_Credit_Card.csv')
st.subheader('Initial Data sample')
st.write(df.head(10))
st.write('No of rows and columns in the dataset are respectively',df.shape)

eda=st.container()
model=st.container()
prediction=st.sidebar.header('Prediction Panel')

with eda:
    st.subheader('Exploratory Data Analysis')
    st.write('The following are the statistical description of the dataset')
    st.write(df.describe())
    st.write('The following is the correlation matrix of the dataset')
    st.write(df.corr())
    st.write('The following is the distribution of the target variable')
    st.write(df['default.payment.next.month'].value_counts())
    st.write('The following is the distribution of the target variable in probability')
    st.write(df['default.payment.next.month'].value_counts(normalize=True))


with model:
    st.subheader('Model Building')
    st.text('I used different models to predict the target variable and tested their accuracy')
    st.text('''  Most suitable ones are SVM and Gradient Descent Classifier  ''')
    st.subheader('SVM and Gradient Descent Classifier both showed Accuracy of 82% on test Data')

with prediction:
    limit=st.sidebar.number_input(label='Amount of given credit in dollars (includes individual and family/supplementary credit',min_value=0,max_value=1000000,format='%d')
    if limit==0:
        limit=1
    limit=np.log(limit)
    sex=st.sidebar.radio(label='Choose Sex Male or Female',options=['male','female'])
    sex_chooser=None
    if sex=='male':
        sex_chooser=1
    else:
        sex_chooser=0
    education_level=st.sidebar.radio(label='Choose Education level',options=['graduate school','university','high school','others'])
    education_chooser=None
    if education_level=='graduate school':
        education_chooser=1
    elif education_level=='university':
        education_chooser=2
    elif education_level=='high school':
        education_chooser=3
    else:   
        education_chooser=0
    marriage=st.sidebar.radio(label='Choose Marital Status',options=['married','single','others'])
    marriage_chooser=None
    if marriage=='married':
        marriage_chooser=1
    elif marriage=='single':
        marriage_chooser=2
    else:
        marriage_chooser=3
    age=st.sidebar.number_input(label='Age in years',min_value=20,max_value=60,format='%d')
    pay0=st.sidebar.radio(label='Repayment status in September, 2005',options=['pay duly','one month delay', 'two months delay','three months delay','four months delay','five months delay','six months delay','seven months delay','eight months delay','nine months delay and above'])
    pay0_chooser=None
    if pay0=='pay duly':
        pay0_chooser=0
    elif pay0=='one month delay':
        pay0_chooser=1
    elif pay0=='two months delay':
        pay0_chooser=2
    elif pay0=='three months delay':
        pay0_chooser=3
    elif pay0=='four months delay':
        pay0_chooser=4
    elif pay0=='five months delay':
        pay0_chooser=5
    elif pay0=='six months delay':
        pay0_chooser=6
    elif pay0=='seven months delay':
        pay0_chooser=7
    elif pay0=='eight months delay':
        pay0_chooser=8
    elif marriage=='nine months delay and above':
        pay0_chooser=9
    else:
        pay0_chooser=9
    pay2=st.sidebar.radio(label='Repayment status in August, 2005',options=['pay duly','one month delay', 'two months delay','three months delay','four months delay','five months delay','six months delay','seven months delay','eight months delay','nine months delay and above'])
    pay2_chooser=None
    if pay2=='pay duly':
        pay2_chooser=0
    elif pay2=='one month delay':
        pay2_chooser=1
    elif pay2=='two months delay':
        pay2_chooser=2
    elif pay2=='three months delay':
        pay2_chooser=3
    elif pay2=='four months delay':
        pay2_chooser=4
    elif pay2=='five months delay':
        pay2_chooser=5
    elif pay2=='six months delay':
        pay2_chooser=6
    elif pay2=='seven months delay':
        pay2_chooser=7
    elif pay2=='eight months delay':
        pay2_chooser=8
    elif marriage=='nine months delay and above':
        pay2_chooser=9
    else:
        pay2_chooser=9
    pay3=st.sidebar.radio(label='Repayment status in July, 2005',options=['pay duly','one month delay', 'two months delay','three months delay','four months delay','five months delay','six months delay','seven months delay','eight months delay','nine months delay and above'])
    pay3_chooser=None
    if pay3=='pay duly':
        pay3_chooser=0
    elif pay3=='one month delay':
        pay3_chooser=1
    elif pay3=='two months delay':
        pay3_chooser=2
    elif pay3=='three months delay':
        pay3_chooser=3
    elif pay3=='four months delay':
        pay3_chooser=4
    elif pay3=='five months delay':
        pay3_chooser=5
    elif pay3=='six months delay':
        pay3_chooser=6
    elif pay3=='seven months delay':
        pay3_chooser=7
    elif pay3=='eight months delay':
        pay3_chooser=8
    elif marriage=='nine months delay and above':
        pay3_chooser=9
    else:
        pay3_chooser=9
    pay4=st.sidebar.radio(label='Repayment status in June, 2005',options=['pay duly','one month delay', 'two months delay','three months delay','four months delay','five months delay','six months delay','seven months delay','eight months delay','nine months delay and above'])
    pay4_chooser=None
    if pay4=='pay duly':
        pay4_chooser=0
    elif pay4=='one month delay':
        pay4_chooser=1
    elif pay4=='two months delay':
        pay4_chooser=2
    elif pay4=='three months delay':
        pay4_chooser=3
    elif pay4=='four months delay':
        pay4_chooser=4
    elif pay4=='five months delay':
        pay4_chooser=5
    elif pay4=='six months delay':
        pay4_chooser=6
    elif pay4=='seven months delay':
        pay4_chooser=7
    elif pay4=='eight months delay':
        pay4_chooser=8
    elif marriage=='nine months delay and above':
        pay4_chooser=9
    else:
        pay4_chooser=9
    pay5=st.sidebar.radio(label='Repayment status in May, 2005',options=['pay duly','one month delay', 'two months delay','three months delay','four months delay','five months delay','six months delay','seven months delay','eight months delay','nine months delay and above'])
    pay5_chooser=None
    if pay5=='pay duly':
        pay5_chooser=0
    elif pay5=='one month delay':
        pay5_chooser=1
    elif pay5=='two months delay':
        pay5_chooser=2
    elif pay5=='three months delay':
        pay5_chooser=3
    elif pay5=='four months delay':
        pay5_chooser=4
    elif pay5=='five months delay':
        pay5_chooser=5
    elif pay5=='six months delay':
        pay5_chooser=6
    elif pay5=='seven months delay':
        pay5_chooser=7
    elif pay5=='eight months delay':
        pay5_chooser=8
    elif marriage=='nine months delay and above':
        pay5_chooser=9
    else:
        pay5_chooser=9
    pay6=st.sidebar.radio(label='Repayment status in April, 2005',options=['pay duly','one month delay', 'two months delay','three months delay','four months delay','five months delay','six months delay','seven months delay','eight months delay','nine months delay and above'],key=2)
    pay6_chooser=None
    if pay6=='pay duly':
        pay6_chooser=0
    elif pay6=='one month delay':
        pay6_chooser=1
    elif pay6=='two months delay':
        pay6_chooser=2
    elif pay6=='three months delay':
        pay6_chooser=3
    elif pay6=='four months delay':
        pay6_chooser=4
    elif pay6=='five months delay':
        pay6_chooser=5
    elif pay6=='six months delay':
        pay6_chooser=6
    elif pay6=='seven months delay':
        pay6_chooser=7
    elif pay6=='eight months delay':
        pay6_chooser=8
    elif marriage=='nine months delay and above':
        pay6_chooser=9
    else:
        pay6_chooser=9
    bill_amt1=st.sidebar.number_input(label=' Bill Amount  Statement  of month September',min_value=0,max_value=1000000,format='%d',value=0)
    bill_amt2=st.sidebar.number_input(label=' Bill Amount  Statement  of month August',min_value=0,max_value=1000000,format='%d',value=0)
    bill_amt3=st.sidebar.number_input(label=' Bill Amount  Statement  of month July',min_value=0,max_value=1000000,format='%d',value=0)
    bill_amt4=st.sidebar.number_input(label=' Bill Amount  Statement  of month June',min_value=0,max_value=1000000,format='%d',value=0)
    bill_amt5=st.sidebar.number_input(label=' Bill Amount  Statement  of month May',min_value=0,max_value=1000000,format='%d',value=0)
    bill_amt6=st.sidebar.number_input(label=' Bill Amount  Statement  of month April',min_value=0,max_value=1000000,format='%d',value=0)
    pay_amt1=st.sidebar.number_input(label=' Pay Amount  Statement  of month September',min_value=0,max_value=1000000,format='%d',value=0)
    pay_amt2=st.sidebar.number_input(label=' Pay Amount  Statement  of month August',min_value=0,max_value=1000000,format='%d',value=0)
    pay_amt3=st.sidebar.number_input(label=' Pay Amount  Statement  of month July',min_value=0,max_value=1000000,format='%d',value=0)
    pay_amt4=st.sidebar.number_input(label=' Pay Amount  Statement  of month June',min_value=0,max_value=1000000,format='%d',value=0)
    pay_amt5=st.sidebar.number_input(label=' Pay Amount  Statement  of month May',min_value=0,max_value=1000000,format='%d',value=0)
    pay_amt6=st.sidebar.number_input(label=' Pay Amount  Statement  of month April',min_value=0,max_value=1000000,format='%d',value=0)
    input_dict={'LIMIT_BAL':limit, 'SEX':sex_chooser, 'EDUCATION':education_chooser, 'MARRIAGE':marriage_chooser, 'AGE':age, 'PAY_0':pay0_chooser, 'PAY_2':pay2_chooser,'PAY_3':pay3_chooser, 'PAY_4':pay4_chooser, 'PAY_5':pay5_chooser, 'PAY_6':pay6_chooser,'BILL_AMT1':bill_amt1,'BILL_AMT2':bill_amt2,'BILL_AMT3':bill_amt3,'BILL_AMT4':bill_amt4,'BILL_AMT5':bill_amt5,'BILL_AMT6':bill_amt6,'PAY_AMT1':pay_amt1,'PAY_AMT2':pay_amt2,'PAY_AMT3':pay_amt3,'PAY_AMT4':pay_amt4,'PAY_AMT5':pay_amt5,'PAY_AMT6':pay_amt6}
    prediction_df=pd.DataFrame([input_dict])
    model=pickle.load(open('credict_defualter_model.pkl','rb'))
    def prediction_func():
        predicted=model.predict(prediction_df)
        prediction_prob=model.predict_proba(prediction_df)
        if predicted==0:
            st.write('Prediction is that the customer will not default')
            st.write('Probability of not defaulting is',prediction_prob[0][0])
        elif predicted==1:
            st.markdown(' # Prediction is that the customer will default')
            st.write(' Probability of defaulting is',prediction_prob[0][1])
    button=st.button('Get Prediction',on_click=prediction_func)

    
    


    
    
    
    
    
    




