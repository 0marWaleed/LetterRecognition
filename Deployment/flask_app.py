from flask import Flask, render_template, request
import pickle

# 1. Load model, scaler, and encoder
with open('letter_classification.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# 2. Map display names → internal feature keys
display_map = {
    "X Position of Bounding Box":      "x_pos",
    "Y Position of Bounding Box":      "y_pos",
    "Box Width":                       "width",
    "Box Height":                      "height",
    "On Pixels Count":                 "on_pix",
    "Mean X Position of On Pixels":    "x_mean",
    "Mean Y Position of On Pixels":    "y_mean",
    "Variance of X Positions":         "x_var",
    "Variance of Y Positions":         "y_var",
    "Correlation X–Y Positions":       "xy_corr",
    "Mean of X² × Y":                  "x2y_mean",
    "Mean of X × Y²":                  "xy2_mean",
    "Left–Right Edge Count Avg":       "x_edge",
    "Corr of X Edge Count with Y":     "x_edge_ycorr",
    "Bottom–Top Edge Count Avg":       "y_edge",
    "Corr of Y Edge Count with X":     "y_edge_xcorr"
}

# Turn into a list of (display, key) tuples
features = list(display_map.items())

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # 3. Gather inputs in the correct order
        vals = [ int(request.form[key]) for _, key in features ]
        # 4. Scale, predict, un‑encode
        scaled = scaler.transform([vals])
        idx    = model.predict(scaled)
        prediction = le.inverse_transform(idx)[0]
    return render_template('index.html',
                           features=features,
                           prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
