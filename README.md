# 🎬 CineRL

**A reinforcement learning movie recommender built on MovieLens 1M.**

CineRL treats recommendation as a sequential decision problem. Instead of ranking movies by popularity alone, a Q-learning agent learns which *genre* to recommend next, based on a user's preference profile and satisfaction level. The learned policy is served through an interactive Flask web app.

**Course:** Advanced Machine Learning (23CSE514) · Unit III: Reinforcement Learning

---

## ✨ Highlights

- **Data pipeline:** prepares the raw MovieLens 1M files with a chronological per-user train/test split.
- **Tabular Q-learning:** a small, interpretable agent with 54 states and 18 actions.
- **Fitted Q-learning:** a neural approximation of the Q-function using `MLPRegressor`.
- **Evaluation:** both agents are compared with a popularity baseline using Hit Rate, Precision, Recall and Coverage at K = 10.
- **Web app:** a Flask interface that serves recommendations from the saved Q-table.
- **Full write-up:** project report, charts and exported results included in the repo.

> The web app is **inference-only**. It loads the prepared catalog, training ratings and saved Q-table at startup. It never retrains the agent or changes the research artifacts.

---

## 🔁 The Reinforcement Learning Loop

```
State → Action → Recommendation → Reward → Next State → Policy Update
```

| Element | In CineRL |
|---|---|
| **State** | `(preferred genre, satisfaction level)`: 18 genres × 3 levels = **54 states** |
| **Action** | Choose one of the **18 MovieLens genres** to recommend from |
| **Recommendation** | Top unseen popular movies from the chosen genre |
| **Reward** | Derived from user ratings (see [Reward design](#-reward-design)) |
| **Next state** | Updated preference profile after the user's feedback |
| **Policy** | ε-greedy during training; greedy `argmax Q(s, a)` during inference |

---

## 🖥️ What the App Does

The **Discover** page asks for three inputs:

1. A MovieLens **user ID**
2. A **preferred genre**
3. A **satisfaction level**: `Low`, `Medium` or `High`

The pair *(preferred genre, satisfaction)* becomes the current state. The agent picks the genre with the highest learned Q-value, then returns up to **10 movies** from that genre that the user has not rated in the training split.

Movies are ordered using training-only popularity:
1. Highest rating count
2. Highest average rating, when counts are tied

**Pages**

| Page | Purpose |
|---|---|
| **Discover** | Generate recommendations |
| **Insights** | See the selected action and all Q-values for the latest session |
| **History** | Review or clear locally stored sessions |
| **About** | Short explanation of the RL loop |

---

## 🏗️ Architecture

```
Browser
  │
  ▼
Flask routes (web/routes.py)
  ├── HTML templates + static JS/CSS
  ├── Recommender (web/recommender.py)
  │     ├── data/processed/movies.csv
  │     ├── data/processed/train_ratings.csv
  │     └── data/processed/q_table.csv
  └── HistoryStore (web/history.py)
        └── data/recommendation_history.json
```

`app.py` is the entry point. It creates the Flask app, loads the recommender, sets up the history store and registers the routes.

---

## 📁 Repository Layout

```
CineRL/
├── app.py                          Flask entry point
├── requirements.txt                Runtime dependencies
├── data/
│   ├── README                      MovieLens documentation and license
│   ├── movies.dat                  Raw movie catalog
│   ├── ratings.dat                 Raw ratings
│   ├── users.dat                   Raw user metadata
│   ├── recommendation_history.json Local session history
│   └── processed/                  Prepared data and model outputs
├── notebooks/
│   ├── 01_data_preparation.ipynb   Load, explore and split the data
│   ├── 02_q_learning.ipynb         Train and evaluate tabular Q-learning
│   └── 03_fitted_q_learning.ipynb  Train and evaluate fitted Q-learning
├── web/
│   ├── history.py                  Thread-safe JSON history storage
│   ├── recommender.py              Q-table loading and inference
│   └── routes.py                   HTML and JSON API routes
├── templates/                      Jinja HTML pages
├── static/
│   ├── css/style.css               Styles
│   └── js/script.js                API calls and page interactions
└── reports/
    ├── CineRL_Project_Report.pdf   Final report
    ├── CineRL_Project_Report.docx  Editable report
    └── figures/                    Generated charts
```

---

## 📊 Dataset

CineRL uses **MovieLens 1M** from the GroupLens Research Project (University of Minnesota).
Source: <https://grouplens.org/datasets/movielens/1m/>

| Property | Value |
|---|---|
| Ratings | 1,000,209 |
| Users | 6,040 |
| Movies | ~3,900 |
| Rating scale | 1–5 stars |
| Genres | 18 |

**File formats**

```
ratings.dat: UserID::MovieID::Rating::Timestamp
movies.dat:  MovieID::Title::Genres
users.dat:   UserID::Gender::Age::Occupation::Zip-code
```

**Preprocessing** (`01_data_preparation.ipynb`) reads the `::`-separated files, keeps the original titles and pipe-separated genres, and produces:

- `data/processed/movies.csv`
- `data/processed/train_ratings.csv`
- `data/processed/test_ratings.csv`

Ratings are split **chronologically per user**: the first 80% form the training set and the last 20% the test set. This keeps future ratings out of the training history.

Please read [`data/README`](data/README) for the original usage conditions, acknowledgement requirements and citation details.

---

## 🧠 RL Design

### State

```
(preferred_genre, satisfaction)
```
18 genres × 3 satisfaction levels = **54 states**.

### Actions

Each of the 18 genres is one action:

```
Action, Adventure, Animation, Children's, Comedy, Crime, Documentary,
Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi,
Thriller, War, Western
```

### 🎁 Reward Design

> **TODO:** Fill this in from your notebook. Suggested wording:
>
> The reward is based on the user's rating of the recommended movie. Ratings of 4–5 give a positive reward, a rating of 3 is neutral, and ratings of 1–2 give a negative reward. This directly rewards recommendations the user actually enjoyed and penalises ones they did not.

### Tabular Q-learning

A Q-table with one row per state and one column per genre action, updated with:

```
Q(s, a) ← Q(s, a) + α [ r + γ · max Q(s′, a′) − Q(s, a) ]
```

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Learning rate (α) | 0.1 |
| Discount factor (γ) | 0.9 |
| Initial epsilon | 1.0 |
| Minimum epsilon | 0.05 |
| Epsilon decay | 0.9 |
| Epochs | 5 |
| Max steps per episode | 10 |
| Training users | 200 |

Outputs: [`q_table.csv`](data/processed/q_table.csv) and [`q_learning_config.json`](data/processed/q_learning_config.json).

### Fitted Q-learning (advanced)

The Q-function is approximated with scikit-learn's `MLPRegressor`:

- 1 hidden layer, 32 ReLU units
- Discount factor γ = 0.5
- Learning rate 0.001
- 50,000 sampled transitions, 5 maximum iterations
- 39 input features: 21 state features + 18 action features
- 54 encoded states and 18 actions

Model and outputs live in [`data/processed/fitted_q_learning`](data/processed/fitted_q_learning). The web app currently serves the **tabular** Q-table, not the fitted model.

---

## 📈 Results

All metrics are measured at **K = 10**.

| Model | Hit Rate@10 | Precision@10 | Recall@10 | Catalog Coverage |
|---|---|---|---|---|
| Q-learning | 0.3006 | 0.0527 | 0.0330 | **0.1028** |
| Popularity baseline | **0.3948** | **0.0810** | **0.0477** | 0.0301 |
| Fitted Q-learning | 0.1168 | 0.0185 | 0.0118 | 0.0306 |

Full numbers: [`final_model_comparison.csv`](data/processed/final_model_comparison.csv).

**Interpretation**

- The **popularity baseline** has the best accuracy, which is expected on MovieLens because ratings are heavily concentrated on popular titles.
- **Tabular Q-learning** reaches over **3× the catalog coverage** (10.3% vs 3.0%), so it recommends a much wider range of movies, at some cost in accuracy.
- **Fitted Q-learning** underperforms here. It is included as an advanced comparison and is likely limited by the small state representation, short training and the number of iterations.

More artifacts (training histories, exploration-experiment Q-tables, transition data, recommendation exports) are in [`data/processed`](data/processed), and charts are in [`reports/figures`](reports/figures).

---

## ⚙️ Installation

**Requirements:** Python 3.10+, `pip`

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Runtime dependencies: **Flask 3.x**, **NumPy ≥ 1.24**, **pandas ≥ 2.0**.
(Running the notebooks additionally needs Jupyter, scikit-learn and matplotlib.)

## ▶️ Run the App

From the repository root:

```bash
python app.py
```

Then open <http://127.0.0.1:5000>.

> The server runs Flask in debug mode because this is a local research demo. Do not deploy it to production without proper configuration, authentication, logging and a persistent database.

---

## 🔌 API Reference

| Endpoint | Description |
|---|---|
| `GET /api/users` | User IDs available in the training ratings |
| `GET /api/genres` | The 18 genre actions in the saved Q-table |
| `POST /api/recommend` | Generate recommendations and record the session |
| `GET /api/insights` | State, action, Q-value and alternatives from the latest session (404 if none) |
| `GET /api/history` | Sessions in reverse chronological order |
| `DELETE /api/history` | Clear the local history |

**`POST /api/recommend`**

Request:
```json
{
  "user_id": 1,
  "preferred_genre": "Action",
  "satisfaction": "High"
}
```

Response:
```json
{
  "state": { "preferred_genre": "Action", "satisfaction": "High" },
  "selected_action": "Thriller",
  "selected_q_value": 4.0318,
  "alternatives": [],
  "recommendations": []
}
```

Invalid or missing fields return **HTTP 400** with an `error` message. `DELETE /api/history` returns `{"status": "cleared"}`.

---

## 🔬 Reproducing the Research

Run the notebooks from the `notebooks/` directory in Jupyter:

1. `01_data_preparation.ipynb`: load the raw files, explore distributions, create the train/test split.
2. `02_q_learning.ipynb`: train and evaluate the tabular agent, export the Q-table.
3. `03_fitted_q_learning.ipynb`: train and evaluate the fitted model, export metrics and recommendations.
4. `python app.py`: serve the saved tabular model in the web interface.

All experiments use a fixed seed of **42**. The web app depends on the files in `data/processed`, so re-run the notebooks deliberately and review the outputs before overwriting existing artifacts.

---

## ⚠️ Limitations

- This is a **research/demo system**, not a production recommender.
- The state is intentionally small and ignores detailed preferences, context, recency and demographics.
- The Q-table scores **genres**, not individual movies. Movies within the chosen genre are ranked by training popularity.
- Users and movies missing from the processed training data cannot be selected in the UI.
- History is stored in a local JSON file, capped at the latest 100 sessions. It is not secure or multi-user storage.
- MovieLens data is for research use under the conditions in [`data/README`](data/README) and must not be redistributed or used commercially without permission.

**Future work:** richer state features, item-level actions, a Deep Q-Network, and off-policy evaluation.

---

## 📚 Citation and Acknowledgements

If you publish results based on MovieLens, please cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19. DOI: [10.1145/2827872](https://doi.org/10.1145/2827872)

Other references:
- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
- C. J. C. H. Watkins and P. Dayan, "Q-learning," *Machine Learning*, 8, 1992.

The dataset is provided by the GroupLens Research Project. See [`data/README`](data/README) for the full license and notes.

## 📄 License

No separate software license is declared. The MovieLens dataset has its own usage conditions in [`data/README`](data/README), which take precedence over any assumption about redistributing the bundled data.

## 👤 Author

**Name:** `<Your Name>` · **Roll No.:** `<Your Roll Number>`
