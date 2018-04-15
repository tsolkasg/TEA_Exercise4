from helper import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Bidirectional, LSTM, Embedding
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from sklearn.model_selection import GridSearchCV
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Paths
tweets_path = "D:/tweets.csv"
emoticons_path = "D:/emoticons.txt"

# Read data and preprocess
tweet_data = read_data(tweets_path)
tweet_data["Tweet"] = tweet_data["Tweet"].apply(preprocess_tweet)

tweet_data = Filter_tweets(tweet_data, True)

# Transform tweets to list of integers and add pad
number_of_features = 2000
tokenizer = Tokenizer(num_words=number_of_features, split=' ')
tokenizer.fit_on_texts(tweet_data['filtered_text'].values)
X = tokenizer.texts_to_sequences(tweet_data['filtered_text'].values)
X = pad_sequences(X, 40)
print(X.shape)

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
def create_model(learn_rate=0.01, dropout_rate=0.5, weight_constraint=2, lstm_size=10, embedding_size=5):
    model = Sequential()
    model.add(Embedding(number_of_features, embedding_size, input_length=X.shape[1]))
    model.add(Bidirectional(
        LSTM(lstm_size, return_sequences=False, dropout=dropout_rate, recurrent_dropout=dropout_rate,
             kernel_regularizer='l2', kernel_constraint=maxnorm(weight_constraint))))
    # model.add(Dense(5,activation='tanh', kernel_constraint=maxnorm(weight_constraint),kernel_regularizer='l2'))
    # model.add(BatchNormalization())
    # model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(Adam(lr=learn_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


model = KerasClassifier(build_fn=create_model, verbose=2)

# Grid search
batch_size = [16, 32, 64, 128]
epochs = [20]
learn_rate = [0.02, 0.1, 0.2]
dropout_rate = [0.5, 0.6, 0.7, 0.8]
lstm_size = [4, 10]
embedding_size = [3, 5]

param_grid = dict(batch_size=batch_size, epochs=epochs, learn_rate=learn_rate, dropout_rate=dropout_rate,
                  lstm_size=lstm_size, embedding_size=embedding_size)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring="f1_micro", cv=3)
grid_search = grid.fit(X_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)

model = grid_search.best_estimator_

# Plots
# Plot Train/Test curve
plotTrainTestLines("LSTM(" + str(number_of_features) + " features)", model, X_train, y_train, validation_X,
                   validation_y)

# Get probabilities from model
probs = model.predict_proba(validation_X)

# Calculate precision/recall values
prec_rec_dict = precision_recall_values(probs, validation_y.values)

# Plot Precision/Recall curve
plotPrecRecCurve(X_train, y_train, validation_X, validation_y, {'LSTM': prec_rec_dict})

# Test score
print("LSTM test score : " + str(model.score(X_test, y_test.values)))
