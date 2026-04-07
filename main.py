from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import get_vectorizer, transform_data
from src.train import train_models
from src.evaluate import evaluate
from src.save_model import save_models

# Load & preprocess
df = load_data("data/dataset.csv")
X, Y = preprocess_data(df)
print("++++++++++++++++++++++=")
# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature engineering
tf = get_vectorizer()
X_train_f, X_test_f = transform_data(tf, X_train, X_test)

# Train
models = train_models(X_train_f, Y_train)

# Evaluate
results = evaluate(models, X_train_f, X_test_f, Y_train, Y_test)
print(results)

# Save results
with open("results.txt", "w") as f:
    f.write(results.to_string(index=False))

# Save models
save_models(models, tf)