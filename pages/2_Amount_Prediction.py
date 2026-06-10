import pandas as pd 
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
st.title("Sales Amount Prediction")
st.image("https://www.kavout.com/wp-content/uploads/2024/11/20241110140934.jpg")
df=pd.read_csv("Amazon Sale Report.csv.gz", low_memory=False)
cols=['Category','Size','Qty','Fulfilment','B2B','Amount']
df=df[cols].dropna()
encoders={}
for col in ['Category','Size','Fulfilment','B2B']:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col].astype(str))
    encoders[col]=le 
X=df.drop("Amount", axis=1)
y=df["Amount"]
model=RandomForestRegressor()
model.fit(X,y)
st.subheader("Enter Product Details")
category=st.selectbox("Category",encoders['Category'].classes_)
size=st.selectbox("Size",encoders['Size'].classes_)
fulfilment=st.selectbox("Fulfilment",encoders['Fulfilment'].classes_)
qty = st.number_input("Quantity",min_value=1,step=1)
b2b=st.selectbox("B2B",encoders['B2B'].classes_)
if st.button("Predict Sales Amount"):
    input_data=pd.DataFrame({
        'Category':[encoders['Category'].transform([category])[0]],
        'Size':[encoders['Size'].transform([size])[0]],
        'Qty':[qty],
        'Fulfilment':[encoders['Fulfilment'].transform([fulfilment])[0]],
        'B2B':[encoders['B2B'].transform([b2b])[0]]
    })
    predicted_amount=model.predict(input_data)[0]
    st.success(f"Predicted Sales Amount: ₹{predicted_amount:,.2f}")
