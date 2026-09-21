# Text Sentiment Analysis (RNN/LSTM)

A deep learning project that classifies movie review sentiment (positive/negative)
using an LSTM-based neural network, trained on the IMDB movie reviews dataset.

## Problem Statement

Understanding customer/user sentiment from text is a common real-world NLP task -
used in product review analysis, social media monitoring, and customer support
prioritization. This project builds an LSTM model to automatically classify
movie reviews as positive or negative.

## Dataset

**IMDB Movie Reviews Dataset** - 50,000 movie reviews (25,000 train, 25,000 test),
labeled positive/negative. Built into `tensorflow.keras.datasets`, downloads
automatically on first run - no manual download needed.

## Approach

1. **Tokenization + Padding** - Reviews converted to integer sequences (top 10,000
   most frequent words), padded/truncated to a fixed length of 200 words.
2. **Model Architecture** - Embedding layer (learns word vector representations) ->
   LSTM layer (captures sequential context) -> Dense layers with Dropout for
   regularization -> Sigmoid output for binary classification.
3. **Training** - Adam optimizer with gradient clipping (`clipnorm=1.0`) to prevent
   training instability, EarlyStopping with best-weights restoration.
4. **Custom Prediction** - The model can classify arbitrary user-typed reviews,
   not just the test set.

## Results

| Metric | Value |
|---|---|
| Validation Accuracy | ~81% |

**Key debugging note:** Initial training showed highly unstable accuracy
(bouncing between 52%-81% across epochs) due to unclipped LSTM gradients.
Adding `clipnorm=1.0` to the Adam optimizer resolved this, producing smooth,
consistent convergence from ~50% to ~91% training accuracy.

## How to Run

```bash
pip install -r requirements.txt
python sentiment_lstm.py
```

## Tech Stack

- Python, TensorFlow/Keras
- LSTM, Embedding layers
- IMDB dataset (built-in)

## Future Improvements

- Try Bidirectional LSTM for richer context
- Use pretrained word embeddings (GloVe/Word2Vec) instead of learning from scratch
- Deploy as a Streamlit app for live review classification
