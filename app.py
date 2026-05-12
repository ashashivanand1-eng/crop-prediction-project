import streamlit as st
import joblib
import matplotlib.pyplot as plt

# ----------- INITIALIZE SESSION STATE ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"

# ----------- BACKGROUND IMAGE + GLOBAL STYLING ----------
def set_background(imagefile):
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("https://i.imgur.com/3ZQ3Q0H.jpg")

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    label, .stMarkdown, .stTextInput, .stNumberInput {{
        font-weight: bold;
        color: black !important;
        font-size: 18px !important;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# ✅ Use short path to your image
set_background("static/bg.jpg")

# ----------- LOGIN FUNCTION ----------
def login():
    st.title("🌾 Smart Crop Prediction System")
    username = st.text_input("**Username**")
    password = st.text_input("**Password**", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.session_state.page = "crop"
            st.rerun()
        else:
            st.markdown("<div style='color:red;font-weight:bold;'>Invalid login</div>", unsafe_allow_html=True)

# ----------- LOGIN CHECK ----------
if not st.session_state.logged_in:
    login()
    st.stop()

# ----------- CROP PREDICTION PAGE ----------
if st.session_state.page == "crop":
    st.header("🌱 Crop Prediction")

    col1, col2 = st.columns([1, 1])

    with col1:
        temp = st.number_input("**Temperature**")
        humidity = st.number_input("**Humidity**")
        ph = st.number_input("**pH**")
        rainfall = st.number_input("**Rainfall**")

        if st.button("Predict Crop"):
            try:
                model = joblib.load("model_crop.pk")
                result = model.predict([[temp, humidity, ph, rainfall]])
                st.session_state.crop = result[0]

                st.markdown(
                    f"<div style='background-color:#ffffffcc;padding:10px;border-radius:5px;"
                    f"color:black;font-weight:bold;font-size:18px;'>Best Crop: {result[0]}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div style='color:black;font-weight:bold;'>Reason: thrives at {temp}°C, "
                    f"{humidity}% humidity, pH {ph}, and rainfall {rainfall} mm.</div>",
                    unsafe_allow_html=True
                )

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba([[temp, humidity, ph, rainfall]])[0]
                    crops = model.classes_
                    scores = probs
                    st.session_state.crop_graph = {"crops": crops, "scores": scores}

            except Exception as e:
                st.markdown(f"<div style='color:red;font-weight:bold;'>Error loading crop model: {e}</div>", unsafe_allow_html=True)

        if st.button("Go to Yield Prediction"):
            st.session_state.page = "yield"
            st.rerun()

    with col2:
        if "crop_graph" in st.session_state:
            fig, ax = plt.subplots()
            ax.bar(st.session_state.crop_graph["crops"], st.session_state.crop_graph["scores"], color="green")
            ax.set_ylabel("Prediction Probability", fontweight="bold")
            ax.set_xlabel("Crops", fontweight="bold")
            st.pyplot(fig)

# ----------- YIELD PREDICTION PAGE ----------
if st.session_state.page == "yield":
    st.header("🌾 Yield Prediction")

    crop = st.session_state.get("crop", "Unknown")
    st.markdown(f"<div style='color:black;font-weight:bold;'>Selected Crop: {crop}</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        area = st.number_input("**Area (hectares)**")
        fertilizer = st.number_input("**Fertilizer Used**")
        rainfall = st.number_input("**Rainfall**")

        if st.button("Predict Yield"):
            try:
                model = joblib.load("model_yield.pk")
                result = model.predict([[area, fertilizer, rainfall]])

                st.markdown(
                    f"<div style='background-color:#ffffffcc;padding:10px;border-radius:5px;"
                    f"color:black;font-weight:bold;font-size:18px;'>Predicted Yield: {result[0]} tons</div>",
                    unsafe_allow_html=True
                )

                st.session_state.yield_graph = {"x": [rainfall-20, rainfall, rainfall+20],
                                                "y": [result[0]-2, result[0], result[0]+2]}

            except Exception as e:
                st.markdown(f"<div style='color:red;font-weight:bold;'>Error loading yield model: {e}</div>", unsafe_allow_html=True)

        if st.button("Go to NDVI/NVI Prediction"):
            st.session_state.page = "ndvi"
            st.rerun()

    with col2:
        if "yield_graph" in st.session_state:
            fig, ax = plt.subplots()
            ax.plot(st.session_state.yield_graph["x"], st.session_state.yield_graph["y"], marker="o", color="orange")
            ax.set_xlabel("Rainfall (mm)", fontweight="bold")
            ax.set_ylabel("Yield (tons)", fontweight="bold")
            st.pyplot(fig)

# ----------- NDVI/NVI PAGE ----------
if st.session_state.page == "ndvi":
    st.header("🌍 NDVI/NVI Prediction")

    col1, col2 = st.columns([1, 1])

    with col1:
        ndvi = st.number_input("**NDVI Value**")
        nvi = st.number_input("**NVI Value**")

        if st.button("Analyze Vegetation"):
            try:
                if ndvi > 0.5:
                    st.markdown(
                        "<div style='background-color:#ffffffcc;padding:10px;border-radius:5px;"
                        "color:black;font-weight:bold;font-size:18px;'>Vegetation Health: Good 🌱</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<div style='background-color:#ffcccc;padding:10px;border-radius:5px;"
                        "color:black;font-weight:bold;font-size:18px;'>Vegetation Health: Poor ⚠️</div>",
                        unsafe_allow_html=True
                    )

                st.session_state.ndvi_graph = {"ndvi": ndvi, "nvi": nvi}

            except Exception as e:
                st.markdown(f"<div style='color:red;font-weight:bold;'>Error analyzing NDVI/NVI: {e}</div>", unsafe_allow_html=True)

    with col2:
        if "ndvi_graph" in st.session_state:
            fig, ax = plt.subplots()
            ax.scatter(st.session_state.ndvi_graph["ndvi"], st.session_state.ndvi_graph["nvi"], color="blue", s=100)
            ax.set_xlabel("NDVI", fontweight="bold") 
            ax.set_ylabel("NVI", fontweight="bold")
            ax.set_title("Vegetation Index Analysis", fontweight="bold")
            st.pyplot(fig)
