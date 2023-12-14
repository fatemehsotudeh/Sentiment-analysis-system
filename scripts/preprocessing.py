import re
import string
import demoji
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

def handle_mentions(tweet):
    return re.sub(r"@[A-Za-z0-9]+", "", tweet)

def handle_hashtags(tweet):
    return re.sub(r"#", "", tweet)

def remove_urls(tweet):
    return re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)

def handle_emoticons_and_emoji(tweet):
    return demoji.replace(tweet)

def remove_retweet_tags(tweet):
    return re.sub(r"RT ", "", tweet)

def handle_special_characters(tweet):
    return re.sub(r'[^\w\s]', '', tweet)

def handle_contractions_and_slang(tweet, contractions=None):
    contractions = contractions or {
        "don't": "do not",
        "can't": "cannot",
    }
    return ' '.join(contractions.get(word, word) for word in tweet.split())


def tokenize_twitter(tweet):
    tokenizer = TweetTokenizer()
    return tokenizer.tokenize(tweet)

def filter_short_words(tokens, min_length=2):
    return [token for token in tokens if len(token) >= min_length]


def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    return [token for token in tokens if token.lower() not in stop_words]

def stem_tokens(tokens):
    # Use a stemming algorithm, such as Porter Stemmer
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def handle_numeric_data(tokens, replace_with="<NUM>"):
    return [replace_with if token.isdigit() else token for token in tokens]

def preprocess_tweet(tweet, contractions=None):
    tweet = handle_mentions(tweet)
    tweet = handle_hashtags(tweet)
    tweet = remove_urls(tweet)
    tweet = handle_emoticons_and_emoji(tweet)
    tweet = remove_retweet_tags(tweet)
    tweet = handle_special_characters(tweet)
    tweet = handle_contractions_and_slang(tweet, contractions)

    tokens = tokenize_twitter(tweet)
    tokens = filter_short_words(tokens)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    tokens = lemmatize_tokens(tokens)
    tokens = handle_numeric_data(tokens)

    preprocessed_tweet = " ".join(tokens)

    return preprocessed_tweet