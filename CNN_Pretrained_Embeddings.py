from helper import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Embedding
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim
from keras.layers import Conv1D, GlobalMaxPooling1D

# Paths
tweets_path = "D:/tweets.csv"
w_emb_path = 'D:/GoogleNews-vectors-negative300.bin'

# Read data and preprocess
tweet_data = read_data(tweets_path)
tweet_data["Tweet"] = tweet_data["Tweet"].apply(preprocess_tweet)

tweet_data = Filter_tweets(tweet_data, True)

# Transform tweets to list of integers and add pad
number_of_features = 5000
tokenizer = Tokenizer(num_words=number_of_features, split=' ')
tokenizer.fit_on_texts(tweet_data['filtered_text'].values)
X = tokenizer.texts_to_sequences(tweet_data['filtered_text'].values)
X = pad_sequences(X)

word_index = tokenizer.word_index

embedding_dims = 300

# Load embeddings
model = gensim.models.KeyedVectors.load_word2vec_format(w_emb_path, binary=True)

# Create embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dims))
for word, i in word_index.items():
    if word in model.wv.vocab:
        embedding_vector = model.wv[word]
        embedding_matrix[i] = embedding_vector

# Encode labels
le = LabelEncoder()
tweet_data["Category"] = le.fit_transform(tweet_data["Category"])

y = tweet_data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

# Split the validation set
validation_X = X_test[:int(len(X_test) / 2)]
X_test = X_test[int(len(X_test) / 2):]

validation_y = y_test[:int(len(y_test) / 2)]
y_test = y_test[int(len(y_test) / 2):]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


#################################################################################################################################################
#################################################################################################################################################
# KERAS models
#################################################################################################################################################

# Create model
def create_model(learn_rate=0.01, filter_size=8, kernel_size=5):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dims, weights=[embedding_matrix], input_length=X.shape[1],
                        trainable=False))
    model.add(Dense(10, activation="relu", kernel_regularizer="l2"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv1D(filter_size, kernel_size, padding='same', activation='tanh', strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(100, activation="tanh", kernel_regularizer="l2"))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(3, activation='softmax'))

    model.compile(Adam(lr=learn_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


model = KerasClassifier(build_fn=create_model, verbose=2)

# Grid search
batch_size = [16, 32, 64, 128]
epochs = [20]
learn_rate = [0.01, 0.1, 0.2]
neurons1 = [5, 10, 20, 30]
kernel_size = [3, 5]
filter_size = [6, 8]

param_grid = dict(batch_size=batch_size, epochs=epochs, learn_rate=learn_rate, kernel_size=kernel_size,
                  filter_size=filter_size)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring="f1_micro", cv=3)
grid_search = grid.fit(X_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)

model = grid_search.best_estimator_

# Plots
# Plot Train/Test curve
plotTrainTestLines("CNN(" + str(number_of_features) + " features)", model, X_train, y_train, validation_X, validation_y)

# Get probabilities from model
probs = model.predict_proba(validation_X)

# Calculate precision/recall values
prec_rec_dict = precision_recall_values(probs, validation_y.values)

# Plot Precision/Recall curve
plotPrecRecCurve(X_train, y_train, validation_X, validation_y, {'CNN': prec_rec_dict})

# Test score
print("CNN test score : " + str(model.score(X_test, y_test.values)))
