import os
import pickle

def save_models(models, tf):
    os.makedirs("models", exist_ok=True)

    names = ['lr', 'dt', 'knn', 'rf', 'stack']

    for name, model in zip(names, models):
        with open(f'models/{name}.pkl', 'wb') as f:
            pickle.dump(model, f)

    with open('models/tfidf.pkl', 'wb') as f:
        pickle.dump(tf, f)