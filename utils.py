import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import classification_report


def plot_learning_curve(estimator, X_train, y_train, cv, scoring, ylabel, title, file_name='learning_curve.png', train_sizes=np.linspace(0.1, 1.0, 20), ylim=None, show=False):
    """
    Generates learning curve data and plots it.

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
    - show: Whether to display the plot or save it to a file.
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
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_means, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_means - train_stds, train_means + train_stds, color='blue', alpha=0.15)
    plt.plot(train_sizes, test_means, label='Cross-validation score', color='green', marker='o')
    plt.fill_between(train_sizes, test_means - test_stds, test_means + test_stds, color='green', alpha=0.15)
    plt.title(title)
    plt.xlabel('Training Data Size')
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid(True)

    if ylim:
        plt.ylim(ylim)

    plt.show() if show else plt.savefig(file_name, bbox_inches='tight', dpi=200)
    plt.close()


def plot_validation_curve(estimator, X_train, y_train, param_name, param_range, cv, scoring, title, xlabel, ylabel='Score', file_name='validation_curve.png', show=False):
    """
    Generates validation curve data and plots it.

    Parameters:
    - estimator: The machine learning model (pipeline) to evaluate.
    - X_train: Training data features.
    - y_train: Training data labels.
    - param_name: Name of the parameter to vary.
    - param_range: Range of values for the parameter.
    - cv: Cross-validation strategy.
    - scoring: Scoring method.
    - file_name: Name of the file to save the plot.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - show: Whether to display the plot or save it to a file.
    """
    # Generate validation curve data
    train_scores, test_scores = validation_curve(
        estimator, X_train, y_train,
        param_name=param_name,
        param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1)

    # Calculate mean and standard deviation for training and test set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, 'o-', color="blue", label="Training score")
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.15)
    plt.plot(param_range, test_mean, 'o-', color="green", label="Cross-validation score")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="green", alpha=0.15)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.xticks(param_range)
    plt.grid(True)

    plt.show() if show else plt.savefig(file_name, bbox_inches='tight', dpi=200)
    plt.close()


def generate_classification_report(estimator, X_test, y_test, title='Classification Report', file_name='classification_report.txt'):
    """
    Makes predictions, prints a classification report, and saves it to a file.

    Parameters:
    - estimator: The model to evaluate.
    - X_test: Test data features.
    - y_test: Test data labels.
    - title: Title for the classification report.
    - file_name: Name of the file to save the report.
    """
    # Evaluate the model on the test set
    y_pred = estimator.predict(X_test)

    # Generate a report
    report = classification_report(y_test, y_pred)

    # Print the report
    print(f"{title}:\n", report)

    # Save the report to a text file
    with open(file_name, 'w') as f:
        f.write(f"{title}:\n")
        f.write(report)
        f.close()
