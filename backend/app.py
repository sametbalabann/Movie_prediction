from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pandas as pd

app = Flask(__name__)
CORS(app)

MODEL_DIR = "backend/recommender_model"

# 📦 Modelleri Yükle
user_lookup = tf.keras.models.load_model(f"{MODEL_DIR}/user_lookup.keras")
movie_lookup = tf.keras.models.load_model(f"{MODEL_DIR}/movie_lookup.keras")
genre_lookup = tf.keras.models.load_model(f"{MODEL_DIR}/genre_lookup.keras")
user_emb = tf.keras.models.load_model(f"{MODEL_DIR}/user_emb.keras")
movie_emb = tf.keras.models.load_model(f"{MODEL_DIR}/movie_emb.keras")
genre_emb = tf.keras.models.load_model(f"{MODEL_DIR}/genre_emb.keras")
rating_model = tf.keras.models.load_model(f"{MODEL_DIR}/rating_mlp.keras")

# 🎬 Film verisini yükle
movies_df = pd.read_csv("backend/data/ml-25m/movies.csv")
movies_df["movieId"] = movies_df["movieId"].astype(str)
movies_df["title"] = movies_df["title"].str.strip()
movies_df["genres"] = movies_df["genres"].astype(str)

# 🎬 API: Film listesini döndür
@app.route("/movies", methods=["GET"])
def get_movies():
    return jsonify(movies_df[["movieId", "title", "genres"]].rename(columns={"movieId": "movie_id"}).to_dict(orient="records"))

# 🔮 API: Tahmin yap
@app.route("/predict_custom", methods=["POST"])
def predict_custom():
    data = request.get_json()
    try:
        rated_movies = data["rated_movies"]
        target_movie_id = str(data["target_movie_id"])
    except KeyError as e:
        return jsonify({"error": f"Gerekli alan eksik: {str(e)}"}), 400

    if len(rated_movies) < 3:
        return jsonify({"error": "En az 3 film puanlayın."}), 400

    # Kullanıcı profil vektörünü oluştur
    vectors = []
    for entry in rated_movies:
        movie_id = str(entry["movie_id"])
        rating = float(entry["rating"])
        m_id = movie_lookup(tf.constant([movie_id]))
        m_vec = movie_emb(m_id)  # [1, 1, emb_dim]
        m_vec = tf.squeeze(m_vec, axis=[0, 1])  # [emb_dim]
        vectors.append(m_vec * rating)

    user_profile_emb = tf.reduce_mean(tf.stack(vectors, axis=0), axis=0)  # [emb_dim]
    user_profile_emb = tf.expand_dims(user_profile_emb, axis=0)  # [1, emb_dim]

    # 🎯 Hedef film vektörü
    target_movie_row = movies_df[movies_df["movieId"] == target_movie_id]
    if target_movie_row.empty:
        return jsonify({"error": "Hedef film veritabanında bulunamadı."}), 404

    genre_str = target_movie_row.iloc[0]["genres"]
    genre_id = genre_lookup(tf.constant([genre_str]))
    genre_vec = genre_emb(genre_id)  # [1, 1, 8]
    genre_vec = tf.squeeze(genre_vec, axis=[0, 1])  # [8]
    genre_vec = tf.expand_dims(genre_vec, axis=0)   # [1, 8]

    m_id = movie_lookup(tf.constant([target_movie_id]))
    target_movie_vec = movie_emb(m_id)  # [1, 1, emb_dim]
    target_movie_vec = tf.squeeze(target_movie_vec, axis=[0, 1])  # [emb_dim]
    target_movie_vec = tf.expand_dims(target_movie_vec, axis=0)  # [1, emb_dim]

    # 🔗 Tüm girdileri birleştir
    final_input = tf.concat([user_profile_emb, target_movie_vec, genre_vec], axis=1)  # [1, emb_dim*2 + 8]

    # 🔮 Tahmin
    predicted = rating_model(final_input).numpy()[0][0]
    predicted_rating = round(float(predicted) * 4 + 1, 2)  # 0-1 → 1-5 ölçeği

    return jsonify({"predicted_rating": predicted_rating})

# ▶ Uygulama başlat
if __name__ == "__main__":
    app.run(debug=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
