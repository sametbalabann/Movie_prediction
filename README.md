# 🎬 AI-Powered Movie Recommendation System

This project is a personalized movie recommender built using **TensorFlow Recommenders**.  
It leverages 12M+ user-movie interactions from the **MovieLens 25M dataset**, trained with TensorFlow and served via a **Flask API**, supported by a clean **HTML-CSS-JS frontend**.

---

## 🧰 Technologies Used

- Python, Pandas, TensorFlow, TensorFlow Recommenders
- Flask, Flask-CORS
- HTML, CSS, Vanilla JavaScript
- MovieLens 25M dataset (real-world data)

---

## 🧠 Model Overview

- Users and movies are embedded via `Embedding` layers  
- Main genre is also embedded and added as an additional signal  
- A user's profile is calculated using the **mean vector of watched movies weighted by rating**  
- The system predicts the expected rating of a target movie using the user's profile and genre

---

## 🎯 Example Prediction

User rates:
Matrix → 5 ⭐, Inception → 4 ⭐
→ Prediction for Interstellar → 4.35 ⭐

