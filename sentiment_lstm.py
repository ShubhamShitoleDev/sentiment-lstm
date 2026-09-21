#################################################################################
#
#  Project Name  : Text Sentiment Analysis System (RNN/LSTM)
#  Description   : Loads the IMDB movie reviews dataset (50,000 reviews,
#                   labeled positive/negative), tokenizes and pads the text,
#                   then trains an LSTM-based neural network to classify
#                   review sentiment. Includes embedding layer, dropout for
#                   regularization, and evaluation on real custom text input.
#  Date          : 19-Sep-2026
#  Author        : Shubham Shitole
#
#################################################################################

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

Border = "-" * 50

# -----------------------------
# CONFIG
# -----------------------------
VOCAB_SIZE = 10000     # only consider the top 10,000 most common words
MAX_LEN = 200           # pad/truncate every review to 200 words
EMBEDDING_DIM = 32      # size of each word's dense vector representation
BATCH_SIZE = 64
EPOCHS = 10


#################################################################################
#
# Function Name : load_data
# Input :         vocab_size
# Description :   Loads the built-in IMDB dataset from Keras, already split
#                 into train/test and pre-tokenized (words converted to
#                 integer indices, most frequent words get smaller numbers)
# Return Value :  (X_train, y_train), (X_test, y_test)
# Date :          19-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def load_data(vocab_size):
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

    print("Training samples : ", len(X_train))
    print("Test samples : ", len(X_test))
    print("Sample review (as integers) : ", X_train[0][:20], "...")
    print("Sample label : ", y_train[0], "(1 = positive, 0 = negative)")

    return (X_train, y_train), (X_test, y_test)


#################################################################################
#
# Function Name : decode_review
# Input :         encoded_review
# Description :   Converts a tokenized (integer-encoded) review back into
#                 readable English text, using IMDB's word index mapping -
#                 helps verify the data loaded correctly and is human-readable
# Return Value :  Decoded review as a string
# Date :          19-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def decode_review(encoded_review):
    word_index = imdb.get_word_index()
    # IMDB reserves indices 0-3 for special tokens (padding, start, unknown, etc.)
    reverse_word_index = {value + 3: key for key, value in word_index.items()}
    reverse_word_index[0] = "<PAD>"
    reverse_word_index[1] = "<START>"
    reverse_word_index[2] = "<UNK>"
    reverse_word_index[3] = "<UNUSED>"

    decoded = " ".join(reverse_word_index.get(i, "?") for i in encoded_review)
    return decoded


#################################################################################
#
# Function Name : prepare_sequences
# Input :         X_train, X_test, max_len
# Description :   Pads/truncates all reviews to a fixed length (max_len) so
#                 they can be fed into the model as fixed-size input batches.
#                 Shorter reviews get zero-padded, longer ones get truncated.
# Return Value :  X_train_padded, X_test_padded
# Date :          19-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def prepare_sequences(X_train, X_test, max_len):
    X_train_padded = pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test, maxlen=max_len, padding='post', truncating='post')

    print("Padded training shape : ", X_train_padded.shape)
    print("Padded test shape : ", X_test_padded.shape)

    return X_train_padded, X_test_padded


#################################################################################
#
# Function Name : build_lstm_model
# Input :         vocab_size, embedding_dim, max_len
# Description :   Builds a sequential model: Embedding layer (converts word
#                 indices into dense vectors) -> LSTM layer (processes the
#                 sequence, capturing context and order) -> Dense layers for
#                 binary classification (positive/negative)
# Return Value :  Compiled-ready Keras Sequential model
# Date :          19-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def build_lstm_model(vocab_size, embedding_dim, max_len):
    model = models.Sequential([
        layers.Input(shape=(max_len,)),

        # Converts each word index into a dense embedding_dim-sized vector.
        # The model LEARNS these embeddings during training - words with
        # similar sentiment context end up with similar vectors.
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),

        # LSTM processes the sequence of word vectors one at a time,
        # maintaining a memory (cell state) that captures context across
        # the whole review. return_sequences=False means we only keep the
        # final output (summary of the whole sequence), not every step.
        layers.LSTM(64, return_sequences=False),

        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),

        # Single neuron with sigmoid activation - outputs a probability
        # between 0 and 1 (close to 1 = positive, close to 0 = negative)
        layers.Dense(1, activation='sigmoid')
    ])

    return model


#################################################################################
#
# Function Name : train_model
# Input :         model, X_train, y_train, X_test, y_test, batch_size, epochs
# Description :   Compiles the model with binary cross-entropy loss (since
#                 this is a binary classification problem) and Adam
#                 optimizer, then trains with early stopping to prevent
#                 overfitting
# Return Value :  history object from model.fit
# Date :          19-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def train_model(model, X_train, y_train, X_test, y_test, batch_size, epochs):
    print(Border)
    print("Training : LSTM Sentiment Model")
    print(Border)

    # clipnorm caps how large any single gradient update can be - LSTMs are
    # prone to occasional large/unstable gradients during training, which
    # shows up as accuracy bouncing wildly between epochs. Clipping the
    # gradient norm keeps updates smooth and training stable.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stop]
    )

    return history


#################################################################################
#
# Function Name : predict_custom_review
# Input :         model, review_text, max_len, word_index
# Description :   Takes a raw English sentence typed by the user, converts it
#                 into the same tokenized+padded format the model expects,
#                 and returns the predicted sentiment with confidence
# Return Value :  sentiment label (string), confidence score (float)
# Date :          19-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def predict_custom_review(model, review_text, max_len, word_index):
    words = review_text.lower().split()
    # +3 offset matches IMDB's reserved index convention (0-3 are special tokens)
    encoded = [1]  # <START> token
    for word in words:
        idx = word_index.get(word, 2) + 3  # 2 = <UNK> for unknown words
        if idx < VOCAB_SIZE:
            encoded.append(idx)
        else:
            encoded.append(2)

    padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0][0]

    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    return sentiment, confidence


#################################################################################
#
# Function Name : main
# Input :         None
# Description :   Runs the full sentiment analysis pipeline end-to-end: load
#                 data, preprocess, build model, train, evaluate, and test
#                 with custom example reviews
# Date :          19-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def main():
    print(Border)
    print("Step 1 : Load the Data")
    print(Border)

    (X_train, y_train), (X_test, y_test) = load_data(VOCAB_SIZE)

    print(Border)
    print("Step 2 : Decode a Sample Review (sanity check)")
    print(Border)

    print(decode_review(X_train[0])[:300], "...")

    print(Border)
    print("Step 3 : Prepare Sequences (Padding)")
    print(Border)

    X_train_padded, X_test_padded = prepare_sequences(X_train, X_test, MAX_LEN)

    print(Border)
    print("Step 4 : Build LSTM Model")
    print(Border)

    model = build_lstm_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN)
    model.summary()

    print(Border)
    print("Step 5 : Train Model")
    print(Border)

    history = train_model(model, X_train_padded, y_train, X_test_padded, y_test, BATCH_SIZE, EPOCHS)

    print(Border)
    print("Step 6 : Final Results")
    print(Border)

    # IMPORTANT: history.history['val_accuracy'][-1] would give the LAST
    # trained epoch's accuracy, not the best one - even though
    # EarlyStopping already restored the best weights into the model.
    # We explicitly re-evaluate the restored model here to report its
    # TRUE accuracy, matching what was actually saved.
    test_loss, test_acc = model.evaluate(X_test_padded, y_test, verbose=0)
    print(f"Final Validation Accuracy (best restored model) : {test_acc*100:.2f}%")
    print(f"Final Validation Loss (best restored model) : {test_loss:.4f}")

    model.save("sentiment_lstm_model.keras")
    print("Model saved as sentiment_lstm_model.keras")

    print(Border)
    print("Step 7 : Test with Custom Reviews")
    print(Border)

    word_index = imdb.get_word_index()

    sample_reviews = [
        "This movie was absolutely fantastic, the acting was superb and I loved every minute of it",
        "Terrible film, complete waste of time, poor acting and boring story",
        "It was okay, not great but not terrible either, average movie"
    ]

    for review in sample_reviews:
        sentiment, confidence = predict_custom_review(model, review, MAX_LEN, word_index)
        print(f"Review: \"{review}\"")
        print(f"  -> Predicted: {sentiment} (confidence: {confidence*100:.1f}%)\n")


if __name__ == "__main__":
    main()