

# UCLA Neural Networks Solution

## Project Overview
This project utilizes a neural network model to predict university admission chances based on various factors such as GRE scores, TOEFL scores, and university ratings. The dataset, `Admission.csv`, includes applicant data which is processed and fed into a neural network to forecast the likelihood of admission. The aim is to provide an insightful tool for prospective students to assess their admission chances.

## Getting Started

### Prerequisites
Before running the project, ensure you have Python installed along with these necessary libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow or keras

You can install these packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib tensorflow
```

### Installation
To get a local copy up and running, follow these simple steps:

```bash
git clone https://github.com/dhrupadDJ/UCLA---Neural_Networks
cd UCLA-Neural_Networks_Solution
```

### File Structure
- `data_preprocessing/`
  - `load_data.py` - Module to load data from CSV.
  - `preprocess_data.py` - Functions to preprocess data such as converting targets, dropping columns, and encoding categorical features.
- `models/`
  - `neural_network.py` - Contains functions to build, train, and predict using the neural network.
- `evaluation/`
  - `evaluation.py` - Functions to evaluate the model including accuracy and loss plotting.
- `Admission.csv` - Dataset file containing applicant data.

### Running the Code
Execute the main script to run the model training and evaluation:

```bash
python main.py
```

Replace `main.py` with the actual name of your script if different.

## Usage
This project can be used by educational consultants and students to understand factors influencing admission probabilities. It provides a quantitative tool to estimate the impact of improving certain scores or credentials.

## Contributing
Interested in contributing? Great! Here's how you can:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

