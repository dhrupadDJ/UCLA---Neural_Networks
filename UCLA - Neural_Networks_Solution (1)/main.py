from data_preprocessing.load_data import load_data
from data_preprocessing.preprocess_data import convert_target, drop_unnecessary_columns, create_dummies, split_data, scale_data
from models.neural_network import train_neural_network, predict, get_loss_curve
from evaluation.evaluation import evaluate_model, plot_loss_curve

# Load the data
file_path = 'Admission.csv'
df = load_data(file_path)

# Convert target variable into categorical
df = convert_target(df)

# Drop unnecessary columns
df = drop_unnecessary_columns(df, ['Serial_No'])

# Create dummy variables for categorical features
df = create_dummies(df, columns=['University_Rating', 'Research'])

# Split the data into train and test sets
xtrain, xtest, ytrain, ytest = split_data(df, 'Admit_Chance')

# Scale the data
Xtrain, Xtest = scale_data(xtrain, xtest)

# Train the neural network model
model = train_neural_network(Xtrain, ytrain)

# Make predictions
ypred = predict(model, Xtest)

# Evaluate the model
conf_matrix, accuracy = evaluate_model(ytest, ypred)
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy: {accuracy}")

# Plot loss curve
loss_values = get_loss_curve(model)
plot_loss_curve(loss_values)
