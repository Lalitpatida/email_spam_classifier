from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC

def train_models(X_train, Y_train):
    lr = LogisticRegression()
    dt = DecisionTreeClassifier()
    knn = KNeighborsClassifier()
    rf = RandomForestClassifier()

    lr.fit(X_train, Y_train)
    dt.fit(X_train, Y_train)
    knn.fit(X_train, Y_train)
    rf.fit(X_train, Y_train)

    estimators = [('lr', lr), ('dt', dt), ('knn', knn), ('rf', rf)]
    stack = StackingClassifier(estimators, final_estimator=SVC(kernel='linear'))
    stack.fit(X_train, Y_train)

    return lr, dt, knn, rf, stack