from keras.datasets import imdb
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence


def main():
    top_words = 5000
    max_review_length = 500
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
    x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, y_train, epochs=3, batch_size=64)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == "__main__":
    main()
