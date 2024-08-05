from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_model(ytest, ypred):
    conf_matrix = confusion_matrix(ytest, ypred)
    accuracy = accuracy_score(ytest, ypred)
    return conf_matrix, accuracy

def plot_loss_curve(loss_values):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
