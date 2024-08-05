from sklearn.neural_network import MLPClassifier

def train_neural_network(Xtrain, ytrain, hidden_layer_sizes=(3,), batch_size=50, max_iter=100, random_state=123):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, batch_size=batch_size, max_iter=max_iter, random_state=random_state)
    model.fit(Xtrain, ytrain)
    return model

def predict(model, Xtest):
    return model.predict(Xtest)

def get_loss_curve(model):
    return model.loss_curve_
