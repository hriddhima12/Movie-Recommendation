const json = (url, options) => fetch(url, options).then(async response => {
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "Request failed");
  return body;
});

async function loadForm() {
  const userSelect = document.querySelector("#user_id");
  if (!userSelect) return;
  const [users, genres] = await Promise.all([json("/api/users"), json("/api/genres")]);
  userSelect.innerHTML = users.map(user => `<option>${user}</option>`).join("");
  document.querySelector("#preferred_genre").innerHTML = genres.map(genre => `<option>${genre}</option>`).join("");
}

document.querySelector("#recommend-form")?.addEventListener("submit", async event => {
  event.preventDefault();
  const form = new FormData(event.target);
  try {
    const result = await json("/api/recommend", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(Object.fromEntries(form))});
    document.querySelector("#results").classList.remove("hidden");
    document.querySelector("#result-title").textContent = `${result.selected_action} recommendations`;
    document.querySelector("#movies").innerHTML = result.recommendations.map(movie => `<article class="movie-card"><div class="poster">✦</div><div><h3>${movie.title}</h3><p class="meta">${movie.year || "Year unavailable"} · ${movie.average_rating.toFixed(2)} average</p></div></article>`).join("");
  } catch (error) { alert(error.message); }
});

async function loadHistory() {
  const list = document.querySelector("#history-list");
  if (!list) return;
  const entries = await json("/api/history");
  list.innerHTML = entries.length ? entries.map(item => `<article class="history-card"><h3>User ${item.user_id} · ${item.selected_action}</h3><p>${item.preferences.preferred_genre} / ${item.preferences.satisfaction}</p><p>${item.recommendations.join(", ")}</p></article>`).join("") : '<div class="panel">No recommendation sessions yet.</div>';
}

async function loadInsights() {
  const empty = document.querySelector("#insight-empty");
  if (!empty) return;
  try {
    const result = await json("/api/insights");
    empty.classList.add("hidden");
    document.querySelector("#insight-content").classList.remove("hidden");
    document.querySelector("#insight-state").textContent = `${result.state.preferred_genre} / ${result.state.satisfaction}`;
    document.querySelector("#insight-action").textContent = `Selected action: ${result.selected_action} · Q-value: ${result.selected_q_value.toFixed(4)}`;
    document.querySelector("#q-values").innerHTML = result.alternatives.map(item => `<p>${item.genre}<span class="bar" style="width:${Math.max(3, Math.min(100, item.q_value * 20))}%"></span>${item.q_value.toFixed(4)}</p>`).join("");
  } catch (error) { empty.textContent = error.message; }
}

document.querySelector("#clear-history")?.addEventListener("click", async () => { await json("/api/history", {method:"DELETE"}); loadHistory(); });
loadForm();
loadHistory();
loadInsights();
