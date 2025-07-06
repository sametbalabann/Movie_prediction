import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import os

def save_layer(layer, input_dtype, path):
    input_ = tf.keras.Input(shape=(1,), dtype=input_dtype)
    model = tf.keras.Model(inputs=input_, outputs=layer(input_))
    model.save(path)

# 📥 Veri yükle (12 milyon satır)
ratings = pd.read_csv("backend/data/ml-25m/ratings.csv", nrows=12_000_000)
movies = pd.read_csv("backend/data/ml-25m/movies.csv")

ratings["userId"] = ratings["userId"].astype(str)
ratings["movieId"] = ratings["movieId"].astype(str)
ratings["rating_normalized"] = (ratings["rating"] - 0.5) / 4.5

movies["movieId"] = movies["movieId"].astype(str)
movies["genres"] = movies["genres"].astype(str)

# 🎯 Genre sadece ilk genre alınır (RAM dostu)
movies["main_genre"] = movies["genres"].apply(lambda x: x.split("|")[0])
merged = ratings.merge(movies[["movieId", "main_genre"]], on="movieId", how="left")

# 🔧 Lookup ve embedding
embedding_dim = 32
user_lookup = tf.keras.layers.StringLookup(vocabulary=merged["userId"].unique(), mask_token=None)
movie_lookup = tf.keras.layers.StringLookup(vocabulary=merged["movieId"].unique(), mask_token=None)
genre_lookup = tf.keras.layers.StringLookup(vocabulary=merged["main_genre"].unique(), mask_token=None)

user_emb = tf.keras.layers.Embedding(len(user_lookup.get_vocabulary()), embedding_dim)
movie_emb = tf.keras.layers.Embedding(len(movie_lookup.get_vocabulary()), embedding_dim)
genre_emb = tf.keras.layers.Embedding(len(genre_lookup.get_vocabulary()), 8)

rating_mlp = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

ds = tf.data.Dataset.from_tensor_slices({
    "userId": tf.constant(merged["userId"].values),
    "movieId": tf.constant(merged["movieId"].values),
    "main_genre": tf.constant(merged["main_genre"].values),
    "rating": tf.constant(merged["rating_normalized"].values, dtype=tf.float32)
}).shuffle(200_000).batch(256).prefetch(tf.data.AUTOTUNE)

# 🧠 Model
class RatingModel(tfrs.models.Model):
    def __init__(self):
        super().__init__()
        self.user_lookup = user_lookup
        self.movie_lookup = movie_lookup
        self.genre_lookup = genre_lookup
        self.user_emb = user_emb
        self.movie_emb = movie_emb
        self.genre_emb = genre_emb
        self.rating_mlp = rating_mlp
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def compute_loss(self, features, training=False):
        uid = self.user_lookup(features["userId"])
        mid = self.movie_lookup(features["movieId"])
        gid = self.genre_lookup(features["main_genre"])
        u = self.user_emb(uid)
        m = self.movie_emb(mid)
        g = self.genre_emb(gid)
        x = tf.concat([u, m, g], axis=1)
        pred = self.rating_mlp(x)
        return self.task(labels=features["rating"], predictions=pred)

# 🚀 Eğitim
model = RatingModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
model.fit(ds, epochs=3)

# 💾 Kayıt
save_dir = "backend/recommender_model"
os.makedirs(save_dir, exist_ok=True)
save_layer(user_lookup, tf.string, f"{save_dir}/user_lookup.keras")
save_layer(movie_lookup, tf.string, f"{save_dir}/movie_lookup.keras")
save_layer(genre_lookup, tf.string, f"{save_dir}/genre_lookup.keras")
save_layer(user_emb, tf.int64, f"{save_dir}/user_emb.keras")
save_layer(movie_emb, tf.int64, f"{save_dir}/movie_emb.keras")
save_layer(genre_emb, tf.int64, f"{save_dir}/genre_emb.keras")
rating_mlp.save(f"{save_dir}/rating_mlp.keras")

print("✅ Eğitim tamamlandı.")
