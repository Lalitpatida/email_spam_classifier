from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer():
    return TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

def transform_data(tf, X_train, X_test):
    X_train_features = tf.fit_transform(X_train)
    X_test_features = tf.transform(X_test)
    return X_train_features, X_test_features