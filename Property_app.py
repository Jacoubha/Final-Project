
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os




st.set_page_config(page_title='Property', layout='wide')

    
# Load Data
df = pd.read_csv('traning.csv')

tabs=st.tabs(["📊 Analysis Page", "🤖 ML Prediction"])


with tabs[0]:

    st.dataframe(df)
    # header
    st.header("Category")
    #figure Category 
    dx=df['Category'].value_counts().reset_index()
    fig=px.bar(dx, x='Category', y='count',title="Category",color='Category', 
             color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig,use_container_width=True)



    # header
    st.header("Usage")
    #figure Usage 
    fig=px.histogram(df,x='Usage',color='Usage',color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig,use_container_width=True)

    
    # header
    st.header("Year Built")
    #figure Year_Built 
    fig=px.histogram(df,x='Year_Built', color='Year_Built',
                     color_discrete_sequence=px.colors.qualitative.Antique )
    st.plotly_chart(fig,use_container_width=True)



    # header
    st.header("Price vs Fin_sqft")
    #checking relationship Price and Fin_sqft
    fig=px.scatter (df,x='Sale_price' , y='Lotsize',
                color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig,use_container_width=False)


    # header
    st.header("Avarage Residential vs Commercial")
    
    dx=df.groupby('Usage')['Sale_price'].mean().reset_index()
    fig=px.bar (dx,y='Sale_price' , x='Usage'  ,color='Usage',
            color_discrete_sequence=px.colors.qualitative.Pastel_r )
    st.plotly_chart(fig,use_container_width=True)
    


with tabs[1]:
    st.title("🤖 Property Price Prediction Model")
    pipeline_path = os.path.abspath("pipeline_Pre.h5")  
    pipeline_Pre = joblib.load(pipeline_path)
    pipeline = joblib.load('model.model')
    CondoProject = joblib.load('CondoProject.List')
    District = joblib.load('District.List')
    Stories = joblib.load('Stories.List')
    Year_Built= joblib.load('Year_Built.List')
    Nr_of_rms= joblib.load('Nr_of_rms.List')
    Fin_sqft= joblib.load('Fin_sqft.List')
    Units= joblib.load('Units.List')
    Bdrms= joblib.load('Bdrms.List')
    Fbath= joblib.load('Fbath.List')
    Hbath= joblib.load('Hbath.List')
    Lotsize= joblib.load('Lotsize.List')
    Usage= joblib.load('Usage.List')
    Category= joblib.load('Category.List')



    st.subheader("📋 Enter P Details")

    user_input = {
        'CondoProject': st.selectbox ("CondoProject", CondoProject),
        'District': st.selectbox("District",District),
        'Stories': st.selectbox("Stories",Stories),
        'Year_Built': st.selectbox("Year_Built", Year_Built),
        'Nr_of_rms': st.selectbox("Nr_of_rms", Nr_of_rms),
        'Fin_sqft': st.slider("Fin_sqft", min_value=140.0,max_value=1090307.0, value=5000.0, step=100.0 ),
         'Units': st.selectbox("Units", Units),
        'Bdrms': st.selectbox("Bdrms", Bdrms),
        'Fbath': st.selectbox("Fbath", Fbath),
        'Hbath': st.selectbox("Hbath", Hbath),
        'Lotsize': st.slider("Lotsize",   min_value=140.0,max_value=1090307.0, value=5000.0, step=100.0 ),
        'Usage': st.selectbox("Usage", Usage),
        'Category': st.selectbox("Category", Category),
    }


    if st.button("Predict Price"):
        input_df = pd.DataFrame([user_input])
        try:
            pred = pipeline.predict(input_df)[0]
            st.success("✅ Price Predict" if pred != 0 else "Price Unkown")
            print(pred)

            st.download_button("📥 Download This Prediction", 
                               data=input_df.to_csv(index=False), 
                               file_name="prediction_result.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
