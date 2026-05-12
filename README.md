# 🤟 Sign Language Translator

An AI-powered Sign Language Translator that detects hand gestures in real time using Computer Vision and Deep Learning. The project uses MediaPipe for hand landmark detection and a TensorFlow/Keras model for gesture classification.

---

## 🚀 Features

* 🎥 Real-time hand gesture detection
* 🖐️ One-hand sign recognition
* 🤖 Deep Learning-based gesture classification
* ⚡ FastAPI backend deployment
* 🌐 Simple frontend using HTML, CSS, and JavaScript
* 📱 Responsive and easy-to-use interface

---

## 🛠️ Tech Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* FastAPI
* Uvicorn

### AI / ML

* TensorFlow / Keras
* MediaPipe
* OpenCV
* NumPy
* Scikit-learn

---

## 📂 Project Structure

```bash
project-folder/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── training model/
│   ├── app.py
│   ├── collectdata.py
│   ├── data.py
│   └── train.py
│
├── server.py
├── requirements.txt
├── model.h5
└── model.json
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/sign-language-translator.git
cd sign-language-translator
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate virtual environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Start the FastAPI server:

```bash
uvicorn server:app --reload
```

Server will run at:

```bash
http://127.0.0.1:8000
```

---

## 🧠 Model Information

The model is trained using:

* Hand landmarks extracted using MediaPipe
* Sequential gesture data
* TensorFlow/Keras neural network

The project currently supports static hand gesture recognition.

---

## 📸 Screenshots
* Home Page
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/d7b661c9-63c6-4d68-9795-8768a9318fb9" />


* Prediction Output
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/d2ac9902-2502-46f6-826f-cf872bfe9d05" />

---

## 🌍 Deployment

### Backend Deployment

You can deploy the FastAPI backend on:

* Render

### Frontend Deployment

You can deploy the frontend on:

* Vercel
---
### 🔗 Live Demo & Project Links

🌐 Live Project

Frontend: [https://your-frontend-link.vercel.app](https://sign-bridge-blond.vercel.app/)
Backend API: [https://your-backend-link.onrender.com](https://signbridge-2-8k7z.onrender.com/)


---
## 📌 Future Improvements

* ✨ Add dynamic sign recognition
* 🌐 Convert signs to speech
* 📱 Mobile app support
* 🧠 Improve model accuracy
* 🌎 Multi-language support

---
## 👩‍💻 Author

**Harshita Suvedi**

If you liked this project, give it a ⭐ on GitHub.
