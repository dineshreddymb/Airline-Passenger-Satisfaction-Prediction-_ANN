import os, logging, warnings, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

sys.path.append(os.path.dirname(__file__))  # Fix for custom transformer on Streamlit Cloud

import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


# ------------------------------------------------------
# Custom LabelEncoder Transformer
# ------------------------------------------------------
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = self.encoders[col].transform(X[col].astype(str))
        return X


# ------------------------------------------------------
# Streamlit Page Settings
# ------------------------------------------------------
st.set_page_config(page_title="Airline Satisfaction", page_icon="✈️", layout="wide")


# ------------------------------------------------------
# BEAUTIFUL FLIGHT THEME UI (Background + Plane + Clouds)
# ------------------------------------------------------
st.markdown("""
<style>

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(#88ccee, #e6f7ff) !important;
    background-attachment: fixed;
    height: 100%;
    overflow-x: hidden;
    font-family: "Poppins", sans-serif;
}

/* CLOUDS */
.cloud {
    position: fixed;
    background: white;
    border-radius: 50px;
    opacity: 0.72;
    animation: floatCloud 35s linear infinite;
    z-index: -1;
}

@keyframes floatCloud {
    from { transform: translateX(-250px); }
    to   { transform: translateX(150%); }
}

/* AIRPLANE */
.plane {
    position: fixed;
    width: 140px;
    top: 22%;
    left: -200px;
    animation: flyPlane 18s linear infinite;
    z-index: -1;
}

@keyframes flyPlane {
    0%   { transform: translateX(-200px) rotate(0deg); opacity: 0.7; }
    50%  { transform: translateX(75vw) rotate(3deg); opacity: 1; }
    100% { transform: translateX(110vw) rotate(0deg); opacity: 0.7; }
}

/* TITLE */
.title3d {
    text-align:center;
    font-size:52px;
    font-weight:900;
    margin-bottom:25px;
    background: linear-gradient(90deg,#004aad,#00d4ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    text-shadow:0 0 18px rgba(0,0,0,0.25);
    animation: titleFloat 4s infinite ease-in-out;
}

@keyframes titleFloat {
    0% { transform:translateY(0px); }
    50% { transform:translateY(-12px); }
    100% { transform:translateY(0px); }
}

/* GLASS CARD */
.card {
    background: rgba(255,255,255,0.45);
    backdrop-filter: blur(12px);
    padding:20px;
    border-radius:14px;
    border:1px solid rgba(255,255,255,0.4);
    box-shadow:0 0 12px rgba(0,0,0,0.18);
    width:95%;
    margin:auto;
    margin-bottom:20px;
}

.section-title {
    font-size:20px;
    font-weight:700;
    color:#003366;
    margin-bottom:12px;
}

.label {
    color:#003366;
    font-weight:600;
}

/* Rating buttons */
.stRadio > div {
    display:flex !important;
    gap:12px !important;
}

/* Button */
.stButton > button {
    background:linear-gradient(90deg,#004aad,#00b7ff);
    color:white;
    padding:10px 28px;
    font-size:18px;
    border-radius:10px;
    font-weight:700;
    transition:0.4s ease;
}

.stButton > button:hover {
    transform:scale(1.05);
    box-shadow:0 8px 25px rgba(0,100,200,0.35);
}

</style>
""", unsafe_allow_html=True)


# CLOUDS
clouds = [
    (200, 10), (150, 30), (220, 55),
    (180, 70), (140, 85), (210, 40),
]
for (size, top) in clouds:
    st.markdown(
        f"<div class='cloud' style='width:{size}px;height:{size/2}px;top:{top}%;'></div>",
        unsafe_allow_html=True,
    )

# AIRPLANE
st.markdown("""
<img class="plane" 
     src="https://www.svgrepo.com/show/419539/airplane-plane.svg">
""", unsafe_allow_html=True)


# TITLE
st.markdown("""
<h1 class="title3d">
✈️ Airline Passenger Satisfaction Prediction
</h1>
""", unsafe_allow_html=True)


# ------------------------------------------------------
# Load ANN model + Preprocessor
# ------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("best_airline_ann_model.h5")
    pre = joblib.load("airline_preprocessor_pipeline.pkl")
    return model, pre

model, preprocessor = load_artifacts()


# ------------------------------------------------------
# Rating input (1–5)
# ------------------------------------------------------
def rating(label, key):
    st.markdown(f"<div class='label'>{label}</div>", unsafe_allow_html=True)
    return st.radio("", [1,2,3,4,5], horizontal=True, key=key, label_visibility="collapsed")


# ------------------------------------------------------
# FORM — ALL INPUT FEATURES
# ------------------------------------------------------
with st.form("airline_form"):

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>👤 Passenger Information</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Age = st.number_input("Age", 1, 120, 30)
        Class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
    with c2:
        CustomerType = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
        TravelType = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
        FlightDistance = st.number_input("Flight Distance", 0, 10000, 500)

    st.markdown("</div>", unsafe_allow_html=True)

    ##################################################

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⏱ Flight Timing</div>", unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    DepDelay = d1.number_input("Departure Delay (min)", 0, 3000, 0)
    ArrDelay = d2.number_input("Arrival Delay (min)", 0, 3000, 0)
    st.markdown("</div>", unsafe_allow_html=True)

    ##################################################

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⭐ Service Ratings (1–5)</div>", unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)

    with r1:
        Seat = rating("Seat comfort", "seat")
        Wifi = rating("Inflight wifi service", "wifi")
        Food = rating("Food and drink", "food")
        Clean = rating("Cleanliness", "clean")
        Boarding = rating("Online boarding", "boarding")

    with r2:
        Support = rating("Online support", "support")
        Entertainment = rating("Inflight entertainment", "enter")
        Checkin = rating("Checkin service", "checkin")
        Gate = rating("Gate location", "gate")
        TimeConv = rating("Departure/Arrival time convenient", "time")

    with r3:
        InflightService = rating("Inflight service", "inflight")
        Legroom = rating("Leg room service", "legroom")
        Onboard = rating("On-board service", "onboard")
        BookingEase = rating("Ease of Online booking", "booking")
        Baggage = rating("Baggage handling", "baggage")

    st.markdown("</div>", unsafe_allow_html=True)

    submit = st.form_submit_button("🔮 Predict Satisfaction")


# ------------------------------------------------------
# Prediction
# ------------------------------------------------------
if submit:

    df = pd.DataFrame({
        "Gender":[Gender],
        "Customer Type":[CustomerType],
        "Age":[Age],
        "Type of Travel":[TravelType],
        "Class":[Class],
        "Flight Distance":[FlightDistance],
        "Departure Delay in Minutes":[DepDelay],
        "Arrival Delay in Minutes":[ArrDelay],
        "Seat comfort":[Seat],
        "Departure/Arrival time convenient":[TimeConv],
        "Food and drink":[Food],
        "Gate location":[Gate],
        "Inflight wifi service":[Wifi],
        "Inflight entertainment":[Entertainment],
        "Online support":[Support],
        "Ease of Online booking":[BookingEase],
        "On-board service":[Onboard],
        "Leg room service":[Legroom],
        "Baggage handling":[Baggage],
        "Checkin service":[Checkin],
        "Cleanliness":[Clean],
        "Online boarding":[Boarding],
        "Inflight service":[InflightService],
    })

    X = preprocessor.transform(df).astype(float)
    prob = float(model.predict(X, verbose=0).squeeze())

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🎯 Prediction Result</div>", unsafe_allow_html=True)

    if prob >= 0.5:
        st.success(f"🙂 Passenger is likely SATISFIED (score = {prob:.3f})")
    else:
        st.error(f"☹️ Passenger is NOT satisfied (score = {prob:.3f})")

    st.markdown("</div>", unsafe_allow_html=True)
