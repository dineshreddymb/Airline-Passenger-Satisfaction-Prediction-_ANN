import os, logging, warnings, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

sys.path.append(os.path.dirname(__file__))  # Needed for custom class

import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


# ------------------------------------------------------
# Custom LabelEncoderTransformer
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
# Streamlit Page Setup
# ------------------------------------------------------
st.set_page_config(page_title="Airline Satisfaction", page_icon="✈️", layout="wide")


# ------------------------------------------------------
# Background + UI Styling
# ------------------------------------------------------
st.markdown("""
<style>

body {
    background: black !important;
    overflow: hidden;
    font-family: "Poppins", sans-serif;
}

/* Stars Animation */
@keyframes starPulse {
    0% { opacity: 0.2; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.3); }
    100% { opacity: 0.2; transform: scale(1); }
}
.star {
    position: fixed;
    background: white;
    border-radius: 50%;
    box-shadow: 0 0 6px rgba(255,255,255,0.8);
    animation: starPulse 3s infinite ease-in-out;
}

/* 3D Title */
@keyframes float3D {
    0%   { transform: translateY(0px) rotateX(0deg); }
    50%  { transform: translateY(-18px) rotateX(8deg); }
    100% { transform: translateY(0px) rotateX(0deg); }
}
@keyframes titlepop {
    0% { opacity:0; transform:scale(0.6); }
    100% { opacity:1; transform:scale(1); }
}
.title3d {
    text-align:center;
    font-size:52px;
    font-weight:900;
    margin-top:12px;
    margin-bottom:25px;
    background: linear-gradient(90deg,#7ee8fa,#eec0c6);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation: float3D 4s ease-in-out infinite, titlepop 1s ease;
    text-shadow: 0 0 25px rgba(255,255,255,0.3);
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    padding:18px;
    border-radius:14px;
    border:1px solid rgba(255,255,255,0.15);
    width:94%;
    margin:auto;
    margin-bottom:20px;
    box-shadow:0 0 12px rgba(255,255,255,0.15);
}

.section-title {
    font-size:20px;
    font-weight:700;
    color:#eafaff;
    margin-bottom:12px;
}

.label {
    color:#d9e7ff;
    font-weight:600;
}

/* Button */
.stButton > button {
    background:linear-gradient(90deg,#00aaff,#33ddff);
    color:white;
    padding:10px 25px;
    font-size:17px;
    border-radius:10px;
    font-weight:700;
    transition:0.3s ease;
}
.stButton > button:hover {
    transform:scale(1.04);
    box-shadow:0 8px 20px rgba(0,170,255,0.45);
}

</style>
""", unsafe_allow_html=True)


# STARFIELD
stars = [
    (2,8,12,0),(3,15,50,0.4),(2,22,80,1.1),(3,30,35,0.7),
    (2,40,10,0.2),(3,48,60,1.3),(2,55,85,0.5),(3,65,20,1.7),
    (2,75,45,0.9),(3,82,70,1.9),(2,5,65,0.8),(3,28,22,1.5),
]
for i,(size,l,t,d) in enumerate(stars):
    st.markdown(
        f"<div class='star' style='width:{size}px;height:{size}px;left:{l}%;top:{t}%;animation-delay:{d}s;'></div>",
        unsafe_allow_html=True,
    )


# TITLE
st.markdown("""
<h1 class="title3d">✈️ Airline Passenger Satisfaction Prediction</h1>
""", unsafe_allow_html=True)


# ------------------------------------------------------
# Load Model + Pipeline
# ------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("best_airline_ann_model.h5")
    pre = joblib.load("airline_preprocessor_pipeline.pkl")
    return model, pre

model, preprocessor = load_artifacts()


# ------------------------------------------------------
# Rating Slider (0–5)
# ------------------------------------------------------
def rating(label, key):
    st.markdown(f"<div class='label'>{label}</div>", unsafe_allow_html=True)
    return st.slider("", 0, 5, 3, key=key, label_visibility="collapsed")


# ------------------------------------------------------
# MAIN FORM
# ------------------------------------------------------
with st.form("airline_form"):

    # Passenger Info
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>👤 Passenger Information</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        Gender = st.selectbox("Gender", ["Male","Female"])
        Age = st.number_input("Age", 1, 120, 30)
        Class = st.selectbox("Class", ["Eco","Eco Plus","Business"])
    with c2:
        CustomerType = st.selectbox("Customer Type", ["Loyal Customer","disloyal Customer"])
        TravelType = st.selectbox("Type of Travel", ["Business travel","Personal Travel"])
        FlightDistance = st.number_input("Flight Distance", 0, 10000, 500)
    st.markdown("</div>", unsafe_allow_html=True)

    # Timing
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⏱ Flight Timing</div>", unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    DepDelay = d1.number_input("Departure Delay (min)", 0, 3000, 0)
    ArrDelay = d2.number_input("Arrival Delay (min)", 0, 3000, 0)
    st.markdown("</div>", unsafe_allow_html=True)

    # Ratings
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⭐ Service Ratings (0–5)</div>", unsafe_allow_html=True)

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
# ------------------------------------------------------
# ✨ PREMIUM ANIMATED PREDICTION RESULT
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

    # --- Animated CSS for Result ---
    st.markdown("""
    <style>
    .result-box {
        padding: 25px;
        border-radius: 15px;
        backdrop-filter: blur(12px);
        animation: popIn 0.8s ease both;
        margin-top: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 0 20px rgba(255,255,255,0.15);
    }

    @keyframes popIn {
        0% { transform: scale(0.6); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }

    .meter {
        width: 80%;
        height: 18px;
        border-radius: 10px;
        margin: auto;
        background: rgba(255,255,255,0.15);
        overflow: hidden;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .meter-fill {
        height: 100%;
        width: 0%;
        background: linear-gradient(90deg,#00eaff,#0077ff);
        border-radius: 10px;
        animation: fillAnim 1.5s ease forwards;
    }

    @keyframes fillAnim {
        from { width: 0%; }
        to { width: VAR_WIDTH%; }
    }

    .badge {
        font-size: 26px;
        font-weight: 900;
        background: linear-gradient(90deg,#00f2fe,#4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 25px rgba(0,200,255,0.4);
        animation: pulseBadge 2s infinite ease-in-out;
    }

    @keyframes pulseBadge {
        0% { transform: scale(1); }
        50% { transform: scale(1.12); }
        100% { transform: scale(1); }
    }
    </style>
    """.replace("VAR_WIDTH", str(int(prob * 100))), unsafe_allow_html=True)

    # --- RESULT BOX ---
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)

    # Badge + Progress Meter
    if prob >= 0.5:
        st.markdown("<div class='badge'>🙂 SATISFIED PASSENGER</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='badge'>☹️ NOT SATISFIED</div>", unsafe_allow_html=True)

    # Meter Bar
    st.markdown("""
        <div class='meter'>
            <div class='meter-fill'></div>
        </div>
    """, unsafe_allow_html=True)

    # Score (Glow Text)
    st.markdown(
        f"<h3 style='text-align:center; color:#a8f7ff; "
        f"text-shadow:0 0 10px #00eaff;'>Prediction Score: {prob:.3f}</h3>",
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


    X = preprocessor.transform(df).astype(float)
    prob = float(model.predict(X, verbose=0).squeeze())

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🎯 Prediction Result</div>", unsafe_allow_html=True)

    if prob >= 0.5:
        st.success(f"🙂 Passenger is likely SATISFIED (score = {prob:.3f})")
    else:
        st.error(f"☹️ Passenger is NOT satisfied (score = {prob:.3f})")

    st.markdown("</div>", unsafe_allow_html=True)
