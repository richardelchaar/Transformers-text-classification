# Multi-class Text Classification with Transformers

This repository documents a comprehensive project on multi-class text classification using the 20 Newsgroups dataset. The project leverages pre-trained transformer models from the Hugging Face ecosystem, detailing a systematic approach from data preprocessing and model fine-tuning to in-depth evaluation, interpretability, and deployment preparation.

-----

## 🎯 Project Goal

The primary objective is to build, fine-tune, and evaluate a robust transformer-based model for classifying text into one of 20 distinct newsgroup categories. The project systematically explores:

  * **Data Preprocessing & Augmentation:** Cleaning raw text and synthetically expanding the training data to improve model generalization.
  * **Model Comparison:** Evaluating different transformer architectures (BERT, DistilBERT, RoBERTa) to select the best baseline.
  * **Hyperparameter Tuning:** Optimizing regularization techniques like dropout.
  * **Performance Evaluation:** Rigorously assessing the final model on a held-out test set using metrics like Accuracy and F1-score.
  * **Model Interpretability:** Gaining insights into the model's decision-making process using techniques like Attention Analysis and LIME.
  * **Deployment & Scalability:** Exporting the model to the ONNX format and benchmarking its inference performance.

-----

## 🚀 Key Findings & TL;DR

  * **Best Model 🏆:** The final, best-performing model is a fine-tuned **`bert-base-uncased`** with a standard classification head and a **dropout rate of 0.05**.
  * **Performance:** The model achieved a **74.1% accuracy** and a **weighted F1-score of 0.744** on the held-out test set.
  * **Strengths & Weaknesses:** The model excels at classifying topics with distinct vocabularies (e.g., `rec.sport.baseball`, `sci.med`) but struggles to differentiate between semantically similar and overlapping categories (e.g., `talk.religion.misc` vs. `alt.atheism`).
  * **Interpretability:** Attention analysis and LIME explanations confirmed that the model focuses on topic-specific keywords. Misclassifications were often driven by ambiguous text or the presence of strong, misleading keywords from an incorrect category.
  * **Deployment Readiness:** The model was successfully exported to the **ONNX format**. Benchmarking on a CPU showed a respectable **\~51ms average latency** for single predictions and a throughput of **\~21.4 samples/sec** with batching, demonstrating its viability for deployment.

-----

## 🛠️ Tech Stack & Libraries

  * **Core Framework:** PyTorch & Hugging Face (`transformers`, `datasets`, `evaluate`, `accelerate`)
  * **Data & Evaluation:** Scikit-learn, NLTK
  * **Interpretability:** LIME
  * **Deployment:** ONNX, ONNX Runtime
  * **Visualization:** Matplotlib, Seaborn

-----

## workflow "Methodology"

The project followed a structured workflow, from data ingestion to final analysis.

### 1\. Data Preparation

  * **Dataset:** `SetFit/20_newsgroups` from the Hugging Face Hub.
  * **Preprocessing:** A comprehensive pipeline was applied to each document to:
    1.  Remove email headers and quotes.
    2.  Strip URLs and email addresses.
    3.  Normalize text (lowercase, remove non-alphanumeric characters).
    4.  Remove common English stopwords.
    5.  Apply Porter stemming to reduce words to their root form.
  * **Data Splitting:** The data was split into **train (70%)**, **validation (15%)**, and **test (15%)** sets. **Stratification** was used to ensure the class distribution was consistent across all splits.
  * **Augmentation:** The training set was augmented using **WordNet-based synonym replacement** on 25% of the samples, increasing its size and diversity.

### 2\. Model Fine-Tuning & Selection

The core of the project involved a multi-stage fine-tuning process managed by the Hugging Face `Trainer` API.

1.  **Baseline Comparison:** `bert-base-uncased`, `distilbert-base-uncased`, and `roberta-base` were fine-tuned for 3 epochs with identical settings. **BERT** was selected for its superior F1-score in this initial test.

2.  **Dropout Tuning:** The selected BERT model was fine-tuned with different dropout rates (0.0, 0.05, 0.1). A rate of **0.05** was chosen as it yielded the best validation F1-score and lowest loss.

3.  **Classification Head Experimentation:** An attempt was made to implement a custom **Self-Attention Pooling head** as an alternative to the standard classification head. However, this custom architecture proved unstable during training, leading to the final selection of the robust and high-performing **Standard FC head**.

### 3\. Evaluation and Interpretation

The final model's behavior was analyzed through multiple lenses:

  * **Quantitative Metrics:** Accuracy, Precision, Recall, and F1-score (both macro and weighted averages) were calculated on the test set.
  * **Confusion Matrix:** A heatmap was generated to visualize class-level performance and identify specific confusion patterns between related topics.
  * **Attention Analysis:** The attention weights of the final transformer layer were inspected to see which tokens the model focused on when making a prediction.
  * **LIME (Local Interpretable Model-agnostic Explanations):** LIME was used to generate instance-specific explanations, highlighting the words that most influenced a particular classification decision.

-----

## 📊 Final Model Performance

The final `bert-base-uncased` model was evaluated on the unseen test set of 2,827 samples.

| Metric              | Score  |
| ------------------- | :----: |
| **Accuracy** | 0.7414 |
| **F1 Score (Weighted)** | 0.7443 |
| **F1 Score (Macro)** | 0.7350 |
| **Precision (Weighted)** | 0.7523 |
| **Recall (Weighted)** | 0.7414 |

#### Per-Class Performance Highlights

  * ✅ **High-Performing Classes (F1 \> 0.85):**
      * `rec.sport.baseball` (0.88)
      * `rec.sport.hockey` (0.88)
      * `sci.med` (0.87)
  * ❌ **Low-Performing Classes (F1 \< 0.60):**
      * `talk.religion.misc` (0.41)
      * `alt.atheism` (0.59)
      * `talk.politics.misc` (0.59)

The results show a clear pattern: the model performs best on topics with unique, specialized vocabularies and struggles most with general, conversational topics that have significant semantic overlap.

-----

## 🤖 Deployment & Scalability (ONNX)

To prepare the model for a production environment, it was exported to the **ONNX** format.

### ONNX Export & Benchmarking

The fine-tuned PyTorch model was successfully converted to `model.onnx`. Inference performance was benchmarked on a **CPU** using ONNX Runtime.

#### Latency (Single Prediction)

| Metric  | Time (ms) |
| :------ | :-------: |
| Average |   51.05   |
| Median  |   50.22   |
| P95     |   58.50   |
| P99     |   64.14   |

#### Throughput (Batch Prediction, BS=32)

| Metric            | Value        |
| :---------------- | :----------: |
| **Samples / Second** | **\~21.42** |

These benchmarks demonstrate that the model is efficient enough for near-real-time applications, especially when leveraging batching. Performance can be further boosted by deploying on a GPU with the CUDA Execution Provider and applying optimizations like INT8 quantization.

-----

## ⚙️ How to Run

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/richardelchaar/Transformers-text-classification.git
    cd Transformers-text-classification
    ```

2.  **Set up the environment:** It is recommended to use a virtual environment. The experiments were run in Google Colab.

    ```bash
    pip install transformers[torch] datasets evaluate accelerate scikit-learn nltk nlpaug shap lime tensorboard matplotlib seaborn onnx onnxruntime --quiet
    ```

3.  **Run the Notebook:** Open and execute the Jupyter Notebook (`.ipynb`). The notebook is structured sequentially:

      * **Section 0-1:** Environment setup and data preprocessing.
      * **Section 2:** Model comparison and tuning.
      * **Section 4:** Final model evaluation on the test set.
      * **Section 5:** Explainability analysis (LIME) and ONNX deployment.

4.  **Outputs:** All model checkpoints, logs, evaluation results, and plots will be saved to the `./output/` directory.

-----

## 📜 Conclusion

This project successfully demonstrates a complete workflow for building a high-performance text classifier using modern transformer models. It highlights the importance of systematic model selection, the challenges of distinguishing between closely related topics, and the practical steps required for model interpretation and deployment. The final BERT-based model provides a strong baseline, while the analysis offers clear directions for future improvements.
