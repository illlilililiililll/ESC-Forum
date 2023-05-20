#http://localhost:8501/
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet

img = Image.open('Assembly.png')
df = pd.read_csv('Supermart_Grocery_Sales.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'])

st.image(img, width=300)
category = list(set(df['Sub Category'].values.tolist()))
string = st.sidebar.selectbox(
    '확인하고 싶은 카테고리를 선택하세요',
    category
)

year_range = st.sidebar.slider( #tuple
    "Select a range of years",
    min_value=2015,
    max_value=2018,
    value=(2015, 2018),
    step=1
)
start = year_range[0]
end = year_range[1]

mask = (df['Sub Category'] == string) & (df['Order Date'].dt.year >= start) & (df['Order Date'].dt.year <= end)
input_df = df.loc[mask]

pred_y_str = st.sidebar.text_input("예측기간(년) : ")
if pred_y_str:
    try:
        pred_y = int(pred_y_str)
    except ValueError:
        st.write('정수 값을 입력해주세요')
else:
    pred_y = 1 # assign a default value to pred_y


m = Prophet()
input_df.rename(columns = {'Order Date' : 'ds'}, inplace = True)
input_df.rename(columns = {'Sales' : 'y'}, inplace = True)
m.fit(input_df)

future = m.make_future_dataframe(periods=pred_y*365)
forecast = m.predict(future)

st.pyplot(m.plot(forecast))