from helper import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.util import ngrams
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from sklearn.model_selection import GridSearchCV

# Paths
tweets_path = "D:/tweets.csv"
positive_words_path = 'D:/positive-words.txt'
negative_words_path = 'D:/negative-words.txt'
emoticons_path = "D:/emoticons.txt"


#################################################################################################################################################
#################################################################################################################################################
# Load/Preprocess data - same as Assignment 2
#################################################################################################################################################


# Create columns of bigrams
def create_bigram_columns(bigrams):
    cols = list()
    for bigr in bigrams:
        word1, word2 = bigr
        bigr = "_".join((word1, word2))
        cols.append(bigr)

    return cols


# Assign 0/1 to bigram columns for each row
def count_bigrams_boolean(df_row):
    tokens = df_row["tokenized_text"]
    for bigram in ngrams(tokens, 2):
        word1, word2 = bigram
        bigram = "_".join((word1, word2))
        if bigram in df_row.index:
            df_row[bigram] = 1
    return df_row


# Read data and preprocess
tweet_data = read_data(tweets_path)
tweet_data["Tweet"] = tweet_data["Tweet"].apply(preprocess_tweet)

# Read positive/negative words
positive_words = read_words(positive_words_path)
negative_words = read_words(negative_words_path)

# Read emoticons
read_emoticons(emoticons_path)

tweet_data = create_base_features(tweet_data, positive_words, negative_words, True)

# Get all feature columns
base_features = list(tweet_data.columns.values)
base_features.remove('Category')
base_features.remove('Tweet')
# print(base_features)

# Encode labels
le = LabelEncoder()
tweet_data["Category"] = le.fit_transform(tweet_data["Category"])
tweet_data["Category"].head()

train_X, test_X, train_y, test_y = train_test_split(tweet_data[base_features], tweet_data['Category'],
                                                    stratify=tweet_data["Category"], test_size=0.2)
print("train size : " + str(len(train_X)))
print("test size : " + str(test_X.shape[0]))

# Get all bigrams
train_bigrams = {}
for train_tokens in train_X["tokenized_text"]:
    for bigram in ngrams(train_tokens, 2):
        train_bigrams[bigram] = train_bigrams.get(bigram, 0) + 1

print("train bigrams :", len(train_bigrams))

# Keep frequent bigrams
train_bigrams_filtered = {}
for k, v in train_bigrams.items():
    if v > 2:
        train_bigrams_filtered[k] = v

print("train bigrams filtered :", len(train_bigrams_filtered))

# Add bigram columns to train/test dataframes and fill with 0
train_bigram_columns = create_bigram_columns(train_bigrams_filtered)

train_X = pd.concat([train_X, pd.DataFrame(columns=train_bigram_columns)])
train_X.fillna(0, inplace=True)
test_X = pd.concat([test_X, pd.DataFrame(columns=train_bigram_columns)])
test_X.fillna(0, inplace=True)

train_X = train_X.apply(count_bigrams_boolean, axis=1)
test_X = test_X.apply(count_bigrams_boolean, axis=1)

# print(le.inverse_transform(0))  # negative
# print(le.inverse_transform(1))  # neutral
# print(le.inverse_transform(2))  # positive

features = base_features
features.remove('tokenized_text')

# Filter the desired columns
train_X = train_X[features]
test_X = test_X[features]

number_of_columns = train_X.shape[1]

# Split the validation set
validation_X = test_X[:int(len(test_X) / 2)]
test_X = test_X[int(len(test_X) / 2):]

validation_y = test_y[:int(len(test_y) / 2)]
test_y = test_y[int(len(test_y) / 2):]


#################################################################################################################################################
#################################################################################################################################################
# KERAS models
#################################################################################################################################################

# Create model
def create_model(learn_rate=0.01, dropout_rate=0.5, weight_constraint=3, neurons1=5, neurons2=5):
    model = Sequential()
    model.add(Dense(neurons1, activation='tanh', kernel_constraint=maxnorm(weight_constraint), kernel_regularizer='l2',
                    input_shape=(number_of_columns,)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons2, activation='tanh', kernel_constraint=maxnorm(weight_constraint), kernel_regularizer='l2'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(Adam(lr=learn_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


model = KerasClassifier(build_fn=create_model, verbose=2)

# Grid search
batch_size = [16, 32, 64, 128]
epochs = [30]
learn_rate = [0.01, 0.1, 0.2]
dropout_rate = [0.5, 0.6, 0.7]
neurons1 = [5, 10, 20, 30]

param_grid = dict(batch_size=batch_size, epochs=epochs, learn_rate=learn_rate, dropout_rate=dropout_rate,
                  neurons1=neurons1)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring="f1_micro", cv=3)
grid_search = grid.fit(train_X, train_y)
print(grid_search.best_score_)
print(grid_search.best_params_)

model = grid_search.best_estimator_

# Plots
# Plot Train/Test curve
plotTrainTestLines("MLP(" + str(number_of_columns) + " features)", model, train_X.values, train_y, validation_X.values,
                   validation_y)

# Get probabilities from model
probs = model.predict_proba(validation_X.values)

# Calculate precision/recall values
prec_rec_dict = precision_recall_values(probs, validation_y.values)

# Plot Precision/Recall curve
plotPrecRecCurve(train_X.values, train_y, validation_X.values, validation_y, {'MLP': prec_rec_dict})

# Test score
print("MLP test score : " + str(model.score(test_X.values, test_y.values)))
