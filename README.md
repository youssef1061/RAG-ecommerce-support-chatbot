# RAG-Based E-commerce Customer Support Chatbot

## Overview

This project integrates language detection, emotion classification, supervised intent classification, and retrieval-augmented customer-support answering into one Python NLP pipeline.

## Modules

1. **Language detection:** Character TF-IDF plus Logistic Regression trained on a sampled subset of `papluca/language-identification`.
2. **Emotion classification:** Pretrained English emotion classifier with a transparent mapping to negative-frustrated, positive-satisfied, and neutral-or-uncertain routing buckets.
3. **Intent classification:** TF-IDF plus Logistic Regression trained on a sampled Bitext customer-support dataset.
4. **RAG:** `all-MiniLM-L6-v2` embeddings, FAISS cosine-similarity retrieval, and an extractive grounded fallback response.

## Routing policy

- Greetings, goodbyes, and gratitude are answered directly.
- Customer-support questions are classified and retrieved from the Bitext knowledge base.
- Complaint/contact intents set `requires_escalation=true`.
- Complaint plus negative emotion sets `priority=high`.
- Weak or unsupported information triggers a safe response rather than an invented policy, order status, or refund decision.

## Results

| Component | Result |
|---|---|
| Language accuracy | [Paste actual output] |
| Language macro F1 | [Paste actual output] |
| Intent accuracy | [Paste actual output] |
| Intent macro F1 | [Paste actual output] |
| RAG Hit@1 | [Paste actual output] |
| RAG Hit@3 | [Paste actual output] |

## Limitations

- This one-hour prototype uses sampled training data for the traditional classifiers.
- Emotion uses a pretrained model rather than task-specific fine-tuning.
- RAG answers are based on a hybrid synthetic Bitext dataset and must not be interpreted as real retailer policies.
- The full intent/emotion/RAG path is evaluated in English only; Arabic is detected but not claimed as fully supported in this deadline version.
- The fallback is extractive and grounded rather than generative because no API key is required.

## How to run

Open the notebook in Google Colab and run cells from top to bottom.
