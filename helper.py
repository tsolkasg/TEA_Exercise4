import pandas as pd
import re
import nltk
from pandas import Series
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import metrics
from collections import OrderedDict

# Dictionary : Emoticon to sentiment(boolean : True for positive, False for negative)
emoticons = {}


# Read data
def read_data(path):
    df = pd.read_csv(path, header=0, quotechar='"')
    # print(len(df))

    positive_tweets = len(df[df["Category"] == "positive"])
    negative_tweets = len(df[df["Category"] == "negative"])
    neutral_tweets = len(df[df["Category"] == "neutral"])
    all_tweets = positive_tweets + negative_tweets + neutral_tweets
    # print(negative_tweets / all_tweets, neutral_tweets / all_tweets, positive_tweets / all_tweets)
    # print(negative_tweets , neutral_tweets , positive_tweets )

    # print(df.head(5))
    # return df[:2000]
    return df


# Read emoticons
def read_emoticons(path):
    with open(path) as f:
        for line in f:
            if "positive" in line.lower():
                is_positive = True
                continue
            elif "negative" in line.lower():
                is_positive = False
                continue
            emoticons[line.strip()] = is_positive


# Preprocess tweets
def preprocess_tweet(text):
    text = re.sub(r"\n", " ", text)  # Replace new lines with space
    text = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' url_link ', text)  # Replaces URLs with 'url_link'
    text = re.sub(r'@[\S]+', 'user_mention', text)  # Replace @user with 'user_mention'
    text = re.sub(r'\.{2,}', ' ', text)  # Replace 2 or more dots with space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one space
    text = re.sub(r'\?{2,}', '?', text)  # Replace 2 or more questionmarks with one ?
    text = re.sub(r'!{2,}', '!', text)  # Replace 2 or more exclamationmarks with one !
    text = re.sub(r'\${2,}', '$', text)  # Replace 2 or more dollar signs with one $
    text = re.sub(r'\*{2,}', '*', text)  # Replace 2 or more * with one *
    text = re.sub(r'(\?+\!+)+', '?!', text)  # Replace multiple ??!!??!?! with ?!
    text = re.sub(r'(\!+\?+)+', '?!', text)  # Replace multiple !!!!??!?!? with !?
    text = re.sub(r'(.)\1+', r'\1\1', text)  # Replace 3 or more occurences of any character to 2 occurences
    text = re.sub(r"(-|\')", '', text)  # Remove characters - and '
    text = re.sub(r'(")', '', text)  # Remove character "
    text = re.sub(r"\s?[0-9]+\.?[0-9]*", ' __number__ ', text)  # Replace numbers with __number__

    return text


def is_emoticon(word):
    return word in emoticons


def is_positive_emoticon(emoticon):
    if emoticon in emoticons:
        return emoticons[emoticon]
    return False


# Read positive/negative words
def read_words(path):
    words = list()

    with open(path) as f:
        for word in f:
            words.append(word.strip())

    return words


# Create features
def create_base_features(df, positive_words, negative_words, lemma=True):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stopwords = nltk.corpus.stopwords.words("english")
    # stopwords = list()
    whitelist = ["n't", "nt", "not"]

    positive_list = list()
    negative_list = list()
    questionmark_list = list()
    exclamationmark_list = list()
    usernames_list = list()
    hashtags_list = list()
    links_list = list()
    upercase_list = list()
    tokens_list = list()
    positive_words_list = list()
    negative_words_list = list()
    numbers_list = list()
    # symbols_list = list()

    for text in df["Tweet"]:
        positive_emoticons = 0
        negative_emoticons = 0
        questionmark_count = 0
        exclamationmark_count = 0
        usernames_count = 0
        hashtags_count = 0
        upercase_count = 0
        links_count = 0
        positive_words_count = 0
        negative_words_count = 0
        numbers_count = 0
        # symbols_count = 0
        tokens = list()
        words = word_tokenize(str(text))
        for word in words:
            word = str(word).strip()
            if (word == "''" or word == '""'):
                continue
            if is_emoticon(word):
                if is_positive_emoticon(word):
                    positive_emoticons += 1
                    word = "emo_pos"
                else:
                    negative_emoticons += 1
                    word = "emo_neg"
            elif (word == "?"):
                questionmark_count += 1
            elif (word == "!"):
                exclamationmark_count += 1
            elif (word == "user_mention"):
                usernames_count += 1
            elif (word == "__number__"):
                numbers_count += 1
            # elif (word == "__symbols__"):
            #     symbols_count += 1
            elif ("#" in word):
                hashtags_count += 1
                word.replace("#", "")
            elif (word == "url_link"):
                links_count += 1
            else:
                if (word.lower() in positive_words):
                    positive_words_count += 1
                elif (word.lower() in negative_words):
                    negative_words_count += 1
                if (word.isupper()):
                    upercase_count += 1

            # last thing to do!!!
            if (len(word) > 1):
                if word not in stopwords or word in whitelist:
                    if (lemma):
                        word = lemmatizer.lemmatize(word.lower())
                    else:
                        word = stemmer.stem(word.lower())
                    tokens.append(word.lower())

        positive_list.append(positive_emoticons)
        negative_list.append(negative_emoticons)
        questionmark_list.append(questionmark_count)
        exclamationmark_list.append(exclamationmark_count)
        usernames_list.append(usernames_count)
        hashtags_list.append(hashtags_count)
        links_list.append(links_count)
        upercase_list.append(upercase_count)
        positive_words_list.append(positive_words_count)
        negative_words_list.append(negative_words_count)
        numbers_list.append(numbers_count)
        # symbols_list.append(symbols_count)
        # tokens_list.append(" ".join(tokens))
        tokens_list.append(tokens)

    df["tokenized_text"] = Series(tokens_list, index=df.index)
    df["positive_emot"] = Series(positive_list, index=df.index)
    df["negative_emot"] = Series(negative_list, index=df.index)
    df["questionmarks"] = Series(questionmark_list, index=df.index)
    df["exclamationmarks"] = Series(exclamationmark_list, index=df.index)
    df["usernames"] = Series(usernames_list, index=df.index)
    df["hashtags"] = Series(hashtags_list, index=df.index)
    df["upercases"] = Series(upercase_list, index=df.index)
    df["positive_words"] = Series(positive_words_list, index=df.index)
    df["negative_words"] = Series(negative_words_list, index=df.index)
    df["numbers"] = Series(numbers_list, index=df.index)
    # df["all_symbols"] = Series(symbols_list, index=df.index)
    df["links"] = Series(links_list, index=df.index)

    return df


# Filter tweets
def Filter_tweets(df, lemma=True):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stopwords = nltk.corpus.stopwords.words("english")
    whitelist = ["n't", "nt", "not"]

    tokens_list = list()

    for text in df["Tweet"]:
        tokens = list()
        words = word_tokenize(str(text))
        for word in words:
            word = str(word).strip()
            if (word == "''" or word == '""'):
                continue
            if is_emoticon(word):
                if is_positive_emoticon(word):
                    word = "emo_pos"
                else:
                    word = "emo_neg"
            elif ("#" in word):
                word.replace("#", "")

            # last thing to do!
            if (len(word) > 1):
                if word not in stopwords or word in whitelist:
                    if (lemma):
                        word = lemmatizer.lemmatize(word.lower())
                    else:
                        word = stemmer.stem(word.lower())
                    tokens.append(word.lower())

        tokens_list.append(" ".join(tokens))

    df["filtered_text"] = Series(tokens_list, index=df.index)

    return df


# used to reset weights at each step
initial_weights = 0

# return the results of the classification in a dictionary
def benchmark(iteration_number, model, train_X, train_y, test_X, test_y, baseline=False):
    global initial_weights
    if (baseline):
        # predict always the most popopular class (which is 2)
        pred = 2 * np.ones(test_X.shape[0])
    else:
        if (iteration_number == 1):
            initial_weights = model.model.get_weights()
        else:
            model.model.set_weights(initial_weights)
        model.fit(train_X, train_y)
        pred = model.predict(test_X)
    # print(pred)
    f1 = metrics.f1_score(test_y, pred, average='micro')
    print("F1 score: %f " % (f1))
    accuracy = metrics.accuracy_score(test_y, pred)
    # print(" Acc: %f "%(accuracy))
    result = {'f1': f1, 'accuracy': accuracy, 'train size': len(train_y), 'test size': len(test_y), 'predictions': pred}
    return result


# Plot train/test lines
# split train set in 10 pieces and train the model at first with 10% of data
# then 20% of data then 30% of data etc and store the results on train and test data
# in a vector inside a dictionary
# data = numpy arrays
def plotTrainTestLines(title, model, X_train_np, df_y_train, X_test_np, df_y_test):
    train_x_s_s = X_train_np
    train_y_s_s = df_y_train.values
    test_x_s_s = X_test_np
    test_y_s_s = df_y_test.values

    train_x_s_s, train_y_s_s = shuffle(train_x_s_s, train_y_s_s)

    results = {}
    results['train_size'] = []
    results['base_classifier'] = []
    results['on_test'] = []
    results['on_train'] = []
    for i in range(1, 11):
        if (i == 10):
            train_x_part = train_x_s_s
            train_y_part = train_y_s_s
        else:
            to = int(i * (train_x_s_s.shape[0] / 10))
            train_x_part = train_x_s_s[0:to, :]
            train_y_part = train_y_s_s[0:to]
        # print(train_x_part.shape)

        # Train size
        results['train_size'].append(train_x_part.shape[0])

        # Train
        result = benchmark(i, model, train_x_part, train_y_part, train_x_part, train_y_part)
        # pprint(result)
        # results['on_train'].append(result['accuracy'])
        results['on_train'].append(result['f1'])

        # Test
        result = benchmark(i, model, train_x_part, train_y_part, test_x_s_s, test_y_s_s)
        # results['on_test'].append(result['accuracy'])
        results['on_test'].append(result['f1'])
        # pprint(result)

        # Base classifier
        result = benchmark(i, None, train_x_part, train_y_part, test_x_s_s, test_y_s_s, True)
        # results['base_classifier'].append(result['accuracy'])
        results['base_classifier'].append(result['f1'])
        # print(result)

    # pylab.rcParams['figure.figsize'] = (20, 6)

    # Create the plot
    fontP = FontProperties()
    fontP.set_size('small')
    fig = plt.figure()
    fig.suptitle('Learning Curves : ' + title, fontsize=20)
    ax = fig.add_subplot(111)
    ax.axis([0, train_x_part.shape[0] + 100, 0, 1.1])
    line_up, = ax.plot(results['train_size'], results['on_train'], 'o-', label='Train', color="blue")
    line_down, = ax.plot(results['train_size'], results['on_test'], 'o-', label='Validation', color="orange")
    line_base, = ax.plot(results['train_size'], results['base_classifier'], 'o-', label='Baseline', color="green")

    plt.xlabel('N. of training instances', fontsize=18)
    plt.ylabel('F1 score', fontsize=16)
    plt.legend([line_up, line_down, line_base], ['Train', 'Validation', 'Baseline'],
               prop=fontP)
    plt.grid(True)

    plt.show()

    # fig.savefig('temp.png')


# Calculate recall, precision values from probabilities (for classifiers that don't implement decision_function )
def precision_recall_values(probabilities, test_y_values):
    prob_dict = {}
    # For different values of x
    for x in np.arange(0.0, 1.001, 0.05):
        tp0 = tp1 = tp2 = 0
        fp0 = fp1 = fp2 = 0
        fn0 = fn1 = fn2 = 0
        # Update values according to predicted probabilities
        for i, pred_row in enumerate(probabilities):
            if (pred_row[0] >= x):
                if (test_y_values[i] == 0):
                    tp0 += 1
                else:
                    fp0 += 1
            else:
                if (test_y_values[i] == 0):
                    fn0 += 1
            if (pred_row[1] >= x):
                if (test_y_values[i] == 1):
                    tp1 += 1
                else:
                    fp1 += 1
            else:
                if (test_y_values[i] == 1):
                    fn1 += 1
            if (pred_row[2] >= x):
                if (test_y_values[i] == 2):
                    tp2 += 1
                else:
                    fp2 += 1
            else:
                if (test_y_values[i] == 2):
                    fn2 += 1

        # Calculate precision/recall values
        if (tp0 + tp1 + tp2 > 0):
            prec = (tp0 + tp1 + tp2) / (tp0 + tp1 + tp2 + fp0 + fp1 + fp2)
            recall = (tp0 + tp1 + tp2) / (tp0 + tp1 + tp2 + fn0 + fn1 + fn2)
            prob_dict[recall] = prec

    # Add value for recall = 0
    min_recall = min(prob_dict.keys())
    prob_dict[0] = prob_dict[min_recall]

    # Sort values
    prob_dict = OrderedDict(sorted(prob_dict.items(), key=lambda z: z[0], reverse=False))

    # Add them to list
    x = list()
    y = list()
    for i in prob_dict:
        # print(i,prob_dict[i])
        x.append(i)
        y.append(prob_dict[i])

    result = {"x": x, "y": y}
    return result


# Plot recall vs precision curve
# classifiers must be already fitted
# extraData : for information about classifiers that don't implement decision_function
def plotPrecRecCurve(X_train_np, y_train_df, x_test_np, y_test_df, extraData=None):
    colors = ['r', 'g', 'y', 'b']

    test_y_modified = np.zeros((len(y_test_df.values), 3))
    print(test_y_modified.shape)
    test_y_modified[:, 0] = y_test_df == 0
    test_y_modified[:, 1] = y_test_df == 1
    test_y_modified[:, 2] = y_test_df == 2

    color_iter = 0

    if extraData is not None:
        for data_label, data in extraData.items():
            plt.plot(data["x"], data["y"], label=data_label)

    baseline_predictions = np.zeros((x_test_np.shape[0], 3))
    baseline_predictions[:, 2] = 1  # always predict the most popular class
    baseline_data = precision_recall_values(baseline_predictions, y_test_df.values)
    plt.plot(baseline_data["x"], baseline_data["y"], label="baseline")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend()
    # plt.title('3-class Average Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()
