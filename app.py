# app.py -- Airline Satisfaction (Airport Runway Theme - Daytime)
import os, sys, logging, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# ensure current directory is on path (helps when deploying custom transformers)
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Prayer lines (user preference)
# ---------------------------
st.write("Radhe Radhe 🙏 Jai Shri Radhe Krishna 🌸")
st.write("Narayana Akhila Guru Bhagavan Sharanam 🕉️")

# ---------------------------
# Custom LabelEncoderTransformer (must match training)
# ---------------------------
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

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Airline Satisfaction (Runway)", page_icon="✈️", layout="wide")

# ---------------------------
# Runway Theme CSS + HTML + Animations
# ---------------------------
st.markdown("""
<style>
/* base */
html, body, [data-testid="stAppViewContainer"] {
  background: linear-gradient(#cfe9ff, #f8fbff);
  font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial;
  overflow-x: hidden;
}

/* central container spacing */
.main-container {
  padding: 12px 20px;
}

/* runway area */
.runway-wrap {
  display:flex;
  justify-content:center;
  margin-bottom:18px;
}
.runway {
  position: relative;
  width: 92%;
  height: 220px;
  background: linear-gradient(#6b6b6b, #4f4f4f);
  border-radius: 8px;
  box-shadow: 0 12px 40px rgba(0,0,0,0.18);
  overflow: hidden;
  border: 2px solid rgba(0,0,0,0.12);
}

/* runway center stripe (repeated) */
.runway::before {
  content: "";
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  width: 60%;
  height: 100%;
  background-image: linear-gradient(180deg, rgba(255,255,255,0.92) 0 8px, rgba(255,255,255,0) 8px);
  background-size: 120px 24px; /* stripe spacing */
  opacity: 0.95;
}

/* side markings (threshold) */
.threshold-left, .threshold-right {
  position:absolute;
  top: 0;
  width: 6%;
  height: 100%;
  background: repeating-linear-gradient(
      to bottom,
      rgba(255,255,255,0.95) 0 24px,
      rgba(255,255,255,0.0) 24px 48px
  );
  opacity: 0.8;
}
.threshold-left { left: 2%; transform: skewX(-6deg); }
.threshold-right { right: 2%; transform: skewX(6deg); }

/* taxiway side (green grass) */
.grass-left, .grass-right {
  position:absolute;
  top:0;
  height:100%;
  width:10%;
  background: linear-gradient(#7ed957, #5fc33b);
  filter: saturate(0.95);
  opacity: 0.98;
}
.grass-left { left: -10%; border-radius: 0 8px 8px 0; }
.grass-right { right: -10%; border-radius: 8px 0 0 8px; }

/* runway lights - two sides, animated "glow" moving */
.runway-lights {
  position:absolute;
  bottom:10px;
  left:0; right:0;
  height:20px;
  pointer-events: none;
}
.light {
  position: absolute;
  width:8px; height:8px;
  border-radius:50%;
  background: rgba(255,230,120,0.95);
  box-shadow: 0 0 8px rgba(255,200,90,0.9);
  transform: translateY(0);
  opacity: 0.95;
  animation: pulse 2.2s infinite;
}
@keyframes pulse {
  0% { transform: translateY(0px); opacity: 0.6; }
  50% { transform: translateY(-3px); opacity: 1; }
  100% { transform: translateY(0px); opacity: 0.6; }
}

/* plane rolling animation */
.plane {
  position: absolute;
  bottom: 60px;
  left: -18%;
  width: 180px;
  transform: rotate(-2deg);
  animation: planeRoll 6s linear infinite;
  z-index: 5;
}
@keyframes planeRoll {
  0% { left: -22%; transform: translateY(0) scale(0.9) rotate(-2deg); opacity: 0.95; }
  50% { left: 60%; transform: translateY(-16px) scale(1.03) rotate(0deg); opacity: 1; }
  100% { left: 120%; transform: translateY(0) scale(0.95) rotate(2deg); opacity: 0.95; }
}

/* glass card */
.card {
  background: rgba(255,255,255,0.9);
  border-radius: 12px;
  padding: 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.08);
  border: 1px solid rgba(0,0,0,0.06);
}

/* titles */
.title {
  font-size:34px;
  font-weight:800;
  color:#03396c;
  margin: 8px 0 16px 0;
  text-align:center;
}

/* small label */
.small-label {
  color:#0b3b5a;
  font-weight:600;
  margin-bottom:8px;
}

/* radio alignment */
.stRadio > div { display:flex !important; gap:8px !important; }

/* submit */
.stButton > button {
  background: linear-gradient(90deg,#0066cc,#00a0ff);
  color:white;
  padding:10px 22px;
  font-size:16px;
  border-radius:10px;
  font-weight:700;
}
.stButton > button:hover { transform:scale(1.03); box-shadow: 0 10px 24px rgba(0,120,200,0.18); }

@media (max-width: 760px) {
  .plane { width: 120px; bottom: 46px; }
  .runway { height: 160px; }
}
</style>
""", unsafe_allow_html=True)

# runway + plane HTML block (daytime)
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="runway-wrap">', unsafe_allow_html=True)
st.markdown('''
  <div class="runway">
    <div class="grass-left"></div>
    <div class="grass-right"></div>
    <div class="threshold-left"></div>
    <div class="threshold-right"></div>

    <!-- runway lights (we place several lights using inline style) -->
    <div class="runway-lights">
''', unsafe_allow_html=True)

# place lights at intervals (left & right)
lights_html = ""
for i in range(8):
    left_pos = 6 + i*11  # percentage
    right_pos = 78 - i*11
    # left light
    lights_html += f'<div class="light" style="left:{left_pos}%;"></div>'
    # right light
    lights_html += f'<div class="light" style="left:{right_pos}%;"></div>'

st.markdown(lights_html, unsafe_allow_html=True)
st.markdown('''
    </div> <!-- runway-lights -->
    <!-- plane SVG (simple) -->
    <img class="plane" src="https://www.svgrepo.com/show/419539/airplane-plane.svg" alt="plane">
  </div>
''', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # close runway-wrap

# ---------------------------
# Load model & preprocessor (cached)
# ---------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = load_model("best_airline_ann_model.h5")
    except Exception as e:
        # helpful debug message if model fails to load
        st.error("Error loading Keras model: " + str(e))
        raise

    try:
        pre = joblib.load("airline_preprocessor_pipeline.pkl")
    except Exception as e:
        st.error("Error loading preprocessor pipeline: " + str(e))
        raise

    return model, pre

model, preprocessor = load_artifacts()

# ---------------------------
# Helper: rating (1-5)
# ---------------------------
def rating(label, key):
    st.markdown(f'<div class="small-label">{label}</div>', unsafe_allow_html=True)
    return st.radio("", [1,2,3,4,5], horizontal=True, key=key, label_visibility="collapsed")

# ---------------------------
# Form (all features)
# ---------------------------
with st.form("airline_form"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title">✈️ Airline Passenger Satisfaction Predictor</div>', unsafe_allow_html=True)

    st.markdown('<div style="display:flex;gap:20px">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        Gender = st.selectbox("Gender", ["Male","Female"])
        Age = st.number_input("Age", min_value=1, max_value=120, value=30)
        Class = st.selectbox("Class", ["Eco","Eco Plus","Business"])
    with c2:
        CustomerType = st.selectbox("Customer Type", ["Loyal Customer","disloyal Customer"])
        TravelType = st.selectbox("Type of Travel", ["Business travel","Personal Travel"])
        FlightDistance = st.number_input("Flight Distance", min_value=0, max_value=10000, value=500)
    st.markdown('</div>', unsafe_allow_html=True)

    # Flight times
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="small-label">⏱ Flight Timing</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    DepDelay = d1.number_input("Departure Delay (min)", min_value=0, max_value=3000, value=0)
    ArrDelay = d2.number_input("Arrival Delay (min)", min_value=0, max_value=3000, value=0)

    # Ratings grid
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="small-label">⭐ Service Ratings (1–5)</div>', unsafe_allow_html=True)
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

    st.markdown('</div>', unsafe_allow_html=True)  # close card

    submit = st.form_submit_button("🔮 Predict Satisfaction")

# ---------------------------
# Prediction logic
# ---------------------------
if submit:
    # prepare dataframe in the same column order your pipeline expects
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

    # transform and predict
    try:
        X = preprocessor.transform(df).astype(float)
    except Exception as e:
        st.error("Preprocessor transform failed: " + str(e))
        st.stop()

    try:
        prob = float(model.predict(X, verbose=0).squeeze())
    except Exception as e:
        st.error("Model prediction failed: " + str(e))
        st.stop()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="small-label">🎯 Prediction Result</div>', unsafe_allow_html=True)
    if prob >= 0.5:
        st.success(f"🙂 Passenger is likely SATISFIED — score = {prob:.3f}")
    else:
        st.error(f"☹️ Passenger is NOT satisfied — score = {prob:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Footer prayer lines
# ---------------------------
st.write("")
st.write("Radhe Radhe 🙏 Jai Shri Radhe Krishna 🌸")
st.write("Narayana Akhila Guru Bhagavan Sharanam 🕉️")
