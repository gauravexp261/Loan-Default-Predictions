import streamlit as st
from catboost import CatBoostRegressor,CatBoostClassifier, metrics, Pool, cv
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pickle  # Import pickle to load the model

# Load the machine learning model
model = pickle.load(open("model/model.pkl", "rb"))

def load_data():
    data = pd.read_csv("data/data.csv")
    return data

data = load_data()

def add_sidebar():
    st.sidebar.header("Loan Default Parameters")
    
    slider_labels = [
        ("Primary Term", "primary_term"),
        ("Encoded Payment 1", "encoded_payment_1"),
        ("Encoded Payment 7", "encoded_payment_7"),
        ("Final Term", "final_term"),
        ("Record Number", "record_number"),
        ("Encoded Payment 6", "encoded_payment_6"),
        ("Days Since Confirmed", "days_since_confirmed"),
        ("Days Since Opened", "days_since_opened"),
        ("Days Till Primary Close", "days_till_primary_close")
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict

def get_radar_chart(input_data):
    categories = ['primary_term', 'encoded_payment_1', 'encoded_payment_7', 'final_term', 
                  'record_number', 'encoded_payment_6', 
                  'days_since_confirmed', 'days_since_opened',
                  'days_till_primary_close']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data[category] for category in categories],
        theta=categories,
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig

def add_predictions(input_data):
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    prediction = model.predict(input_array)
    prediction_probabilities = model.predict_proba(input_array)
    
    st.subheader("Loan Default Predictions")
    st.write("The borrower has:")
    
    if prediction[0] == 0:
        st.write("<span class='predictions Defaulted'> Defaulted</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='predictions Paid'> Paid</span>", unsafe_allow_html=True)
        #st.write("paid loan on Time")
    
    st.write("Probability of Loan Default: ", prediction_probabilities[0][0])
    st.write("Probability of Loan Paid on Time: ", prediction_probabilities[0][1])
    
    #st.write("This application can aid financial professionals in their decision-making process, but it is not a replacement for professional financial advice.")

def main():
    st.set_page_config(page_title="Loan Default Predictions",
                      layout="wide", 
                      initial_sidebar_state="expanded")

    with open("assets/style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    input_dict = add_sidebar()

    # # Display radar chart
    # radar_chart = get_radar_chart(input_dict)
    # st.plotly_chart(radar_chart)

    # # Display predictions
    # add_predictions(input_dict)

    with st.container():
        st.title("Loan Default Predictions")
        st.write("This application can aid financial professionals in their decision-making process, but it is not a replacement for professional financial advice.")
        col1, col2 = st.columns([4, 1])
        
        with col1:
            radar_chart = get_radar_chart(input_dict)
            st.plotly_chart(radar_chart, use_container_width=True)
        
        with col2:
            add_predictions(input_dict)

if __name__ == '__main__':
    main()
