import streamlit as st
import pickle

# Load model, scaler, and label encoder using pickle
with open('letter_classification.pkl', 'rb') as f:
    svm = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Feature names for the sliders
feature_names = [
    "X position of bounding box (x_pos)",
    "Y position of bounding box (y_pos)",
    "Width of bounding box (width)",
    "Height of bounding box (height)",
    "Total number of 'on' pixels (on_pix)",
    "Mean X position of on pixels (x_mean)",
    "Mean Y position of on pixels (y_mean)",
    "Variance of X positions (x_var)",
    "Variance of Y positions (y_var)",
    "Correlation between X and Y positions (xy_corr)",
    "Mean of X² * Y (x2y_mean)",
    "Mean of X * Y² (xy2_mean)",
    "Mean edge count from left to right (x_edge)",
    "Correlation of X edge count with Y (x_edge_ycorr)",
    "Mean edge count from bottom to top (y_edge)",
    "Correlation of Y edge count with X (y_edge_xcorr)"
]

st.title("🔤 Letter Recognition with SVM")
st.write("Use the sliders below to adjust the feature values, and see the predicted letter!")

# Slider inputs for each feature
with st.form("predict_form"):
    st.subheader("Enter Feature Values")
    features = []
    for feature_name in feature_names:
        # Each slider with a range from 0 to 15
        feature_value = st.slider(f"{feature_name}", 0, 15, 7)
        features.append(feature_value)
    
    submit = st.form_submit_button("Predict")

# Prediction process when the form is submitted
if submit:
    # Scale and predict the input
    input_scaled = scaler.transform([features])
    pred_index = svm.predict(input_scaled)
    pred_letter = le.inverse_transform(pred_index)[0]

    # Show prediction result
    st.subheader("Predicted Letter")
    st.success(f"🧠 The model predicts: **{pred_letter}**")