from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def evaluate(models, X_train, X_test, Y_train, Y_test):
    names = ['Logistic Regression', 'Decision Tree', 'KNN', 'Random Forest', 'Stacking']
    
    results = []

    for name, model in zip(names, models):
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        results.append({
            'Model': name,
            'Train Accuracy': accuracy_score(Y_train, train_pred),
            'Test Accuracy': accuracy_score(Y_test, test_pred),
            'Precision': precision_score(Y_test, test_pred),
            'Recall': recall_score(Y_test, test_pred),
            'F1 Score': f1_score(Y_test, test_pred)
        })

    df = pd.DataFrame(results)
    return df