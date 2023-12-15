from sklearn.svm import LinearSVC


def train_svm_model(X_train, y_train):
    svm_model = LinearSVC(random_state=0, tol=1e-5, verbose=True, max_iter=1000, dual=False)
    svm_model.fit(X_train, y_train)
    return svm_model
