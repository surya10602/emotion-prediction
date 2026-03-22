# Emotion & Decision Engine
## Overview
This repository contains an end-to-end machine learning pipeline designed to understand human emotional states from noisy journal entries, reason through imperfect metadata signals, and decide on meaningful next actions to guide users toward better mental states.

The system is built entirely for local, edge-ready execution without reliance on hosted LLM APIs, prioritizing privacy, low latency, and robustness.

## Architecture & Approach
To handle the messy reality of user data (short texts, contradictory signals, and missing values), I implemented a Hybrid Embedding + Tabular Model Pipeline.

The system operates in three distinct layers:

- **Emotional Understanding (ML Layer):** Analyzes text and physiological metadata to predict the user's emotional state and intensity.

- **Uncertainty Modeling (Probabilistic Layer):** Evaluates the confidence of the predictions to flag ambiguous or conflicting states.

- **Decision & Guidance (Logic Layer):** Uses a contextual rule engine to determine the best intervention (What to do), the optimal timing (When to do it), and generates a human-like supportive message.

## Feature Engineering
- **Textual Data:** Journal entries are often short or vague. I handled missing text gracefully by filling nulls with empty strings, then vectorized the text into dense 384-dimensional embeddings to capture semantic meaning rather than just keywords.

- **Contextual Metadata:** Signals like `sleep_hours`, `energy_level`, and `stress_level` act as critical physiological anchors. Missing values were imputed using the median to prevent data loss. These features were then standardized using `StandardScaler`.

- **Feature Fusion:** The text embeddings and scaled numerical metadata were horizontally stacked (`np.hstack`) to give the downstream models a holistic view of both the user's mind (text) and body (metadata).

## Model Choice
- **Text Embedding:** `all-MiniLM-L6-v2`

  - Why: It is an exceptionally lightweight sentence transformer (~80MB). It is highly suitable for future edge-device deployment (mobile) while still being powerful enough to extract nuanced sentiment from messy, colloquial journal entries.

- **State Classification & Intensity Regression: `XGBoost`**

  - Why: XGBoost excels at handling concatenated tabular data (our fused embeddings + metadata). For emotional state (multi-class), `XGBClassifier` provides robust probability distributions, which is essential for calculating our Uncertainty Flag. For intensity, `XGBRegressor` allows us to predict continuous nuances before clipping/rounding to the required 1-5 scale.

- **Guidance Generation:** A local, rule-based template engine rather than a generative LLM, ensuring zero hallucinations and immediate inference times.

## Setup Instructions

### Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

### Installation
Install the required dependencies using pip:
```Bash
pip install pandas numpy scikit-learn xgboost sentence-transformers
```

### How to Run
Ensure that your dataset files (train.csv and test.csv) are located in the root directory alongside the script.

Execute the pipeline:
```Bash
python pipeline.py
```
The script will automatically process the data, train the models, run the decision engine, and generate the final predictions.csv in the same directory.

## Project Deliverables Generated
Running the pipeline produces predictions.csv with the following schema:

- `id`: User ID

- `predicted_state`: The classified emotional state.

- `predicted_intensity`: The intensity of the emotion (1-5).

- `confidence`: The model's probability score for the winning class (0.0 to 1.0).

- `uncertain_flag`: Binary flag (1 if confidence < 0.45, else 0).

- `what_to_do`: The recommended action (e.g., box_breathing, deep_work).

- `when_to_do`: The optimal timing (e.g., now, later_today).

- `supportive_message`: A locally generated, conversational explanation of the recommendation.
