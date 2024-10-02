import matplotlib.pyplot as plt
import numpy as np
import wandb
import yaml

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
from ucimlrepo import fetch_ucirepo

import matplotlib
matplotlib.use('Agg')  # Use Agg to avoid GUI issues


def plot_learning_curve(estimator, X_train, y_train, cv, scoring, ylabel, title, train_sizes=np.linspace(0.1, 1.0, 20), ylim=None):
    """
    Generates learning curve data and returns a plot.

    Parameters:
    - estimator: The machine learning model (pipeline) to evaluate.
    - X_train: Training data features.
    - y_train: Training data labels.
    - cv: Cross-validation strategy.
    - scoring: Scoring method.
    - ylabel: Label for the y-axis.
    - title: Title of the plot.
    - file_name: Name of the file to save the plot.
    - train_sizes: Array of training sizes to use for generating the learning curve.
    - ylim: Y-axis limits.
    """
    # Generate learning curve data
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_train, y_train, cv=cv,
        train_sizes=train_sizes,
        scoring=scoring, n_jobs=-1)

    # Calculate Mean and Standard Deviation of Training and Test Scores
    train_means = np.mean(train_scores, axis=1)
    train_stds = np.std(train_scores, axis=1)
    test_means = np.mean(test_scores, axis=1)
    test_stds = np.std(test_scores, axis=1)

    # Plot a learning curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_sizes, train_means, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_means - train_stds, train_means + train_stds, color='blue', alpha=0.15)
    plt.plot(train_sizes, test_means, label='Cross-validation score', color='green', marker='o')
    plt.fill_between(train_sizes, test_means - test_stds, test_means + test_stds, color='green', alpha=0.15)
    plt.title(title)
    plt.xlabel('Training Data Size')
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    if ylim:
        plt.ylim(ylim)

    return plt


# Define training function
def main():
    # Initialize a wandb run
    wandb.init()

    # Fetch the dataset from UCI
    aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890)
    X = aids_clinical_trials_group_study_175.data.features
    y = aids_clinical_trials_group_study_175.data.targets.squeeze()

    random_state = 42

    # Split the data into training and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    # Set up CV with stratification
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Hyperparameters
    C = wandb.config.C
    coef0 = wandb.config.coef0
    degree = wandb.config.degree

    # Set up a pipeline with a standard scaler and an SVM model with polynomial kernel
    svm_pipe = make_pipeline(StandardScaler(), SVC(kernel='poly', C=C, coef0=coef0, degree=degree))

    # Fit the model
    svm_pipe.fit(X_train, y_train)

    # Predict on the test set
    y_pred = svm_pipe.predict(X_test)

    # Calculate the weighted f1 score
    weighted_f1_score = f1_score(y_test, y_pred, average='weighted')

    # Log a learning curve to wandb
    plot_learning_curve(svm_pipe, X_train, y_train, cv=cv, scoring='f1_weighted',
                        title='SVM Learning Curve (Polynomial Kernel)', ylabel='F1 Score (Weighted)', ylim=(0.5, 1.0))
    wandb.log({"learning_curve": wandb.Image(plt)})
    plt.close()

    # Log a confusion matrix to wandb
    wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels=svm_pipe.classes_)

    # Generate the classification report as a dictionary
    class_report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Convert the classification report into a wandb Table
    columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    table_data = []
    for label, metrics in class_report_dict.items():
        if label not in ["accuracy"]:  # Exclude accuracy from the classification report
            table_data.append([
                label,
                metrics["precision"],
                metrics["recall"],
                metrics["f1-score"],
                metrics["support"]
            ])

    # Create a wandb Table for the classification report
    class_report_table = wandb.Table(columns=columns, data=table_data)

    # Save the report and F1 score in the run summary
    wandb.run.summary["classification_report_table"] = class_report_table
    wandb.run.summary["F1 Score (Weighted)"] = weighted_f1_score

    # Finish the wandb run
    wandb.finish()


if __name__ == '__main__':
    # Load the sweep configuration
    with open('sweep_config.yml') as file:
        sweep_config = yaml.safe_load(file)

    # Initialize a wandb sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project='svm-poly-kernel')

    # Run the sweep
    wandb.agent(sweep_id, function=main)
