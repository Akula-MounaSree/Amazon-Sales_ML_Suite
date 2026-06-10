import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
st.title("Customer Segmentation")
df=pd.read_csv("Amazon Sale Report.csv.gz", low_memory=False)
data=df[['Amount', 'Qty']].dropna()
kmeans=KMeans(n_clusters=3, random_state=42, n_init=10)
data['Cluster']=kmeans.fit_predict(data)
cluster_names={0:"Premium Customer",1:"Regular Customer", 2:"Budget Customer"}
data['Customer_Type']=data['Cluster'].map(cluster_names)
st.success("Customer Segmentation Completed")
#cluster summary
st.subheader("Cluster Summary")
summary=(data.groupby("Customer_Type").agg(
    {'Amount':'mean',
     'Qty':'mean'}
).round(2))
st.dataframe(summary)
#scatter plot
st.subheader(summary)
fig=px.scatter(data, x="Amount", y="Qty", color="Customer_Type", title="Customer Segmenatation using k-Means")
st.plotly_chart(fig, use_container_width=True)

#distribution
st.subheader("Segment Distribution")
count_df=(data['Customer_Type'].value_counts().reset_index())
count_df.columns=['Customer_Type', 'Count']
fig2=px.pie(count_df, names="Customer_Type", values='Count')
st.plotly_chart(fig2)
