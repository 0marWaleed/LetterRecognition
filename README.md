# 🧠 Letter Recognition using Machine Learning

A machine learning project to classify handwritten capital letters (A–Z) based on 16 numerical features. The final model is deployed with **Streamlit**, **Flask**, and **Tkinter** interfaces.

---

## 📊 Dataset

- **Source**: UCI Machine Learning Repository – [Letter Recognition Data Set](https://archive.ics.uci.edu/ml/datasets/letter+recognition)
- **Classes**: 26 uppercase English letters (A–Z)
- **Features**: 16 numerical attributes per sample (extracted from images)

---

## 🧪 Project Workflow

1. **Preprocessing**
   - Convert data from `.csv` to structured format using Pandas
   - Normalize/scale features
   - Encode target labels (A–Z)
   - Split into training/testing sets

2. **Modeling**
   - Trained using `Support Vector Machine (SVM)` (best-performing model)
   - Model serialized with `pickle` to `letter_classification.pkl`

3. **Deployment**
   - Built 3 frontends using:
     - `Streamlit` for quick interactive demos
     - `Flask` for web deployment
     - `Tkinter` for local desktop GUI
---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/0marWaleed/LetterRecognition.git
```

### 2. Install Dependencies

If not using a `requirements.txt`, install manually:

```bash
pip install pandas numpy scikit-learn tensorflow streamlit flask
```

---

## 🖥️ Run the Apps

### ▶️ Streamlit App

```bash
streamlit run streamlit_app.py
```

### 🌐 Flask App

```bash
python flask_app.py
```
Then visit: `http://127.0.0.1:5000`

### 🪟 Tkinter App

```bash
python tkinter_app.py
```

---

## 📈 Model Performance

- **Algorithm**: Support Vector Machine (SVM)  
- **Accuracy**: ~97.8%

---

## 🧰 Tools & Libraries

- Python, NumPy, Pandas
- Scikit-learn
- TensorFlow
- Streamlit, Flask, Tkinter
- Pickle (model saving)
- Matplotlib/Seaborn (for visualization)

---

## ✍️ Author

**Omar Waleed**  
- GitHub: [@0marWaleed](https://github.com/0marWaleed)

---

## 📄 License

This project is licensed under the MIT License.  
Feel free to fork, modify, and use it for educational or commercial purposes!

---
