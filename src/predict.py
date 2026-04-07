import pickle

def load_model():
    model = pickle.load(open('models/stack.pkl', 'rb'))
    tf = pickle.load(open('models/tfidf.pkl', 'rb'))
    return model, tf

def predict(text):
    model, tf = load_model()
    vec = tf.transform([text])
    result = model.predict(vec)
    return "Spam" if result[0] == 0 else "Ham"