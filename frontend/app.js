let allMovies = [];
let ratedMovies = [];

window.onload = () => {
  fetch("http://127.0.0.1:5000/movies")
    .then(res => res.json())
    .then(data => {
      allMovies = data;
      console.log("🎬 Film verisi yüklendi:", allMovies.length);
    });
};

function normalize(str) {
  return str.toLowerCase().replace(/[^a-z0-9]/gi, "").trim();
}

function findMovieByTitle(title) {
  const normalizedInput = normalize(title);
  return allMovies.find(m => normalize(m.title) === normalizedInput);
}

function addMovie() {
  const title = document.getElementById("filmSearch").value;
  const movie = findMovieByTitle(title);
  if (!movie) return alert("⚠️ Film bulunamadı.");

  if (ratedMovies.some(m => m.movie_id === movie.movie_id)) {
    alert("⚠️ Bu film zaten eklenmiş.");
    return;
  }

  const rating = prompt("Bu filme verdiğiniz puanı girin (1-5):");
  const parsedRating = parseFloat(rating);

  if (isNaN(parsedRating) || parsedRating < 1 || parsedRating > 5) {
    alert("⚠️ Geçerli bir puan girin (1-5)");
    return;
  }

  ratedMovies.push({ movie_id: movie.movie_id, rating: parsedRating });

  const li = document.createElement("li");
  li.innerHTML = `
    <img src="https://img.omdbapi.com/?apikey=demo&t=${encodeURIComponent(movie.title)}" alt="poster" width="30">
    ${movie.title} → ${parsedRating} ⭐
    <button onclick="removeMovie('${movie.movie_id}')">❌</button>
  `;
  li.id = `movie-${movie.movie_id}`;
  document.getElementById("selectedMoviesList").appendChild(li);

  document.getElementById("filmSearch").value = "";
}

function removeMovie(movieId) {
  ratedMovies = ratedMovies.filter(m => m.movie_id !== movieId);
  const item = document.getElementById(`movie-${movieId}`);
  if (item) item.remove();
}

function predictRating() {
  const title = document.getElementById("targetFilmSearch").value;
  const movie = findMovieByTitle(title);
  if (!title) {
    alert("⚠️ Lütfen tahmin edilecek filmi girin.");
    return;
  }
  if (!movie) return alert("⚠️ Hedef film bulunamadı.");
  if (ratedMovies.length < 3) return alert("⚠️ En az 3 film puanlayın.");

  if (ratedMovies.some(m => m.movie_id === movie.movie_id)) {
    alert("⚠️ Bu film zaten izlenmiş.");
    return;
  }

  fetch("http://127.0.0.1:5000/predict_custom", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      rated_movies: ratedMovies,
      target_movie_id: movie.movie_id
    })
  })
    .then(res => res.json())
    .then(data => {
      const result = document.getElementById("predictionResult");
      if (data.predicted_rating) {
        result.innerText = `🎯 Tahmini Puan: ${data.predicted_rating} ⭐`;
      } else {
        result.innerText = `❌ Hata: ${data.error || "Bilinmeyen hata"}`;
      }
    });
}

// 🔄 Otomatik tamamlama
function getMatchingMovies(query) {
  const q = query.toLowerCase().trim();
  return allMovies
    .filter(m =>
      m.title.toLowerCase().includes(q) &&
      !ratedMovies.some(r => r.movie_id === m.movie_id)
    )
    .slice(0, 10);
}

function showSuggestions(inputId, suggestionsId) {
  const input = document.getElementById(inputId);
  const box = document.getElementById(suggestionsId);
  box.innerHTML = "";

  const matches = getMatchingMovies(input.value);
  if (!input.value || matches.length === 0) return;

  matches.forEach(movie => {
    const div = document.createElement("div");
    div.classList.add("autocomplete-suggestion");
    div.innerText = movie.title;
    div.onclick = () => {
      input.value = movie.title;
      box.innerHTML = "";
    };
    box.appendChild(div);
  });
}

document.getElementById("filmSearch").addEventListener("input", () => {
  showSuggestions("filmSearch", "filmSuggestions");
});

document.getElementById("targetFilmSearch").addEventListener("input", () => {
  showSuggestions("targetFilmSearch", "targetFilmSuggestions");
});
