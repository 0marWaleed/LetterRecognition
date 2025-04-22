import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import numpy as np

# Load model, scaler, and encoder
with open('letter_classification.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Feature display names mapped to internal names
feature_display_map = {
    "X position of bounding box (x_pos)": "x_pos",
    "Y position of bounding box (y_pos)": "y_pos",
    "Width of bounding box (width)": "width",
    "Height of bounding box (height)": "height",
    "Total number of 'on' pixels (on_pix)": "on_pix",
    "Mean X position of on pixels (x_mean)": "x_mean",
    "Mean Y position of on pixels (y_mean)": "y_mean",
    "Variance of X positions (x_var)": "x_var",
    "Variance of Y positions (y_var)": "y_var",
    "Correlation between X and Y positions (xy_corr)": "xy_corr",
    "Mean of X² * Y (x2y_mean)": "x2y_mean",
    "Mean of X * Y² (xy2_mean)": "xy2_mean",
    "Mean edge count from left to right (x_edge)": "x_edge",
    "Correlation of X edge count with Y (x_edge_ycorr)": "x_edge_ycorr",
    "Mean edge count from bottom to top (y_edge)": "y_edge",
    "Correlation of Y edge count with X (y_edge_xcorr)": "y_edge_xcorr"
}

feature_names = list(feature_display_map.keys())

# Build UI
root = tk.Tk()
root.title("Letter Recognition")
root.geometry("900x600")
root.config(bg="#d9e6f2")

frame = tk.Frame(root, padx=30, pady=20, bg="#ffffff", bd=2, relief="groove")
frame.place(relx=0.5, rely=0.5, anchor="center")

header = tk.Label(frame, text="🔤 Letter Recognition", font=("Arial", 20, "bold"), bg="#ffffff", fg="#2c3e50")
header.grid(row=0, column=0, columnspan=4, pady=15)

dropdowns = {}

for i in range(8):
    name_left = feature_names[i]
    label_left = tk.Label(frame, text=name_left, bg="#ffffff", anchor="center", font=("Arial", 10))
    label_left.grid(row=i+1, column=0, sticky="ew", pady=5, padx=5)
    cb_left = ttk.Combobox(frame, values=list(range(16)), width=10, state="readonly")
    cb_left.set(7)
    cb_left.grid(row=i+1, column=1, pady=5, padx=5, sticky="ew")
    dropdowns[feature_display_map[name_left]] = cb_left

    name_right = feature_names[i+8]
    label_right = tk.Label(frame, text=name_right, bg="#ffffff", anchor="center", font=("Arial", 10))
    label_right.grid(row=i+1, column=2, sticky="ew", pady=5, padx=5)
    cb_right = ttk.Combobox(frame, values=list(range(16)), width=10, state="readonly")
    cb_right.set(7)
    cb_right.grid(row=i+1, column=3, pady=5, padx=5, sticky="ew")
    dropdowns[feature_display_map[name_right]] = cb_right

def predict():
    try:
        features = [float(dropdowns[f].get()) for f in feature_display_map.values()]
        scaled = scaler.transform([features])
        pred = model.predict(scaled)
        letter = le.inverse_transform(pred)[0]
        messagebox.showinfo("Prediction", f"The model predicts: {letter}")
    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong: {e}")

predict_btn = tk.Button(frame, text="Predict", command=predict, bg="#4CAF50", fg="white", padx=15, pady=8, font=("Arial", 11, "bold"))
predict_btn.grid(row=10, column=0, columnspan=4, pady=30)

root.mainloop()
