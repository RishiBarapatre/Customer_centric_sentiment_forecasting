# User-Centric Sentiment Forecasting

**A Modular NLP System for Emotion-Aware, Time-Sensitive Feedback Modeling**

Businesses rely on user feedback to tailor experiences and reduce churn, yet traditional sentiment models fail to capture the individual trajectory of a user's emotional engagement. This project introduces a personalized sentiment engine that accurately forecasts future user sentiment by analyzing past reviews, even when that data is sparse, temporally irregular, and syntactically complex.

Designed for feedback-driven domains (e-commerce, ed-tech, customer support), this system bridges the gap between the syntactic, emotional, and temporal dimensions of user behavior.

## Tech Stack & Tools

* **Machine Learning & NLP:** PyTorch, Graph Neural Networks (GNNs), HuggingFace Transformers (RoBERTa), Seq2Seq Models.
* **Data Processing:** spaCy (Dependency Parsing), NRC Emotion Lexicon.
* **Architectures:** Dual-Graph GCNs, Time-Aware Encoder-Decoder (GRU/Attention).

## Business Impact & Applications

This pipeline goes beyond aggregate sentiment tracking, enabling individual-level predictive intelligence:
* **Customer Churn Modeling:** Detects emerging negative sentiment trajectories before a user uninstalls or cancels.
* **Adaptive Response Generation:** Tailors customer support interactions based on a user's forecasted emotional state.
* **Product Sentiment Drift:** Monitors how user feelings change toward specific product features (aspects) over time.

---

## Core Engineering Challenges & Solutions

### 1. Overcoming the Syntactic Limitations of Transformers (ABSA)
**The Problem:** Standard transformer models parse flat token sequences, ignoring the syntactic dependencies critical for Aspect-Based Sentiment Analysis (ABSA). For example, in "The pasta was amazing, but the service was slow," transformers often fail to map "slow" strictly to "service".

**The Architecture:**
* Implemented Dual-Graph Convolutional Networks (GCNs) .
* Constructed a Contextual Graph for semantic proximity (word co-occurrence) .
* Constructed a Dependency Graph using spaCy to map grammatical roles (e.g., modifiers, subjects) .
* **Impact:** Fusing both graphs extracts highly accurate aspect-opinion-sentiment triplets, increasing interpretability and reliability for multi-aspect reviews .

### 2. Solving the "Cold-Start" User Sparsity Problem
**The Problem:** Standard user embeddings fail when a user has only written 2-3 reviews, making personalized forecasting nearly impossible .

**The Architecture:**
* Extracted domain-agnostic, low-dimensional emotion vectors (Joy, Sadness, Fear, Trust, etc.) using the NRC Emotion Lexicon .
* Represented users by their emotional distribution rather than raw text history .
* Applied K-Means clustering to group sparse users into latent behavioral cohorts .
* **Impact:** Enables robust sentiment forecasting for infrequent reviewers by leveraging the generalized trends of their assigned cohort .

### 3. Modeling Irregular Temporal Dynamics
**The Problem:** Reviews are posted at unpredictable intervals . Traditional sequential models (RNNs) assume fixed-length time steps, causing them to misjudge the relevance of older reviews .

**The Architecture:**
* Engineered a Time-Aware Encoder-Decoder .
* The Encoder processes both the sentiment score and the time elapsed since the last review .
* Integrated a time-decay attention mechanism, allowing the model to dynamically prioritize recent reviews or recognize patterns like sentiment drops after long silences .
* **Impact:** Outperforms baseline sequence models on temporal forecasting by learning the task-specific relevance of time gaps .

---

## Experimental Results

Evaluated on a timestamped, public domain product review dataset.

| Metric | Baseline | Our Model | Improvement |
| :--- | :--- | :--- | :--- |
| **ABSA Accuracy** | 72.5% (BERT) | **84.3%** | **+11.8%**  |
| **Cold-Start User MAE** | 0.89 | **0.63** | **Decrease 29.2%**  |
| **Forecasting RMSE** | 0.72 | **0.48** | **Decrease 33.3%**  |

---

## Documentation & Resources

* **[Visual Walkthrough / Presentation](https://www.canva.com/design/DAGnKB8d3Hg/V-8sYcg_aqRZ-gTNMH9egA/edit):** A high-level overview of the problem space and model architecture .
* **[Technical Report](./Docs/technical_report.pdf):** Detailed documentation covering the mathematical foundations of the Time-Aware Attention mechanism and the Dual-Graph GCN configuration .

---

## Future Roadmap

* **Multilingual Support:** Extending dependency parsing and emotion lexicons to support non-English review datasets .
* **Multimodal Fusion:** Integrating product metadata, user demographics, and session behavior for deeper profiling .
* **Microservice Deployment:** Packaging the inference pipeline as a scalable API for integration into live feedback dashboards .