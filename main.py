import os
import utils
import wandb

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
from ucimlrepo import fetch_ucirepo

import matplotlib
matplotlib.use('Agg')  # Use Agg to avoid GUI issues


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

    # Log the learning curve
    utils.plot_learning_curve(svm_pipe, X_train, y_train, cv=cv, scoring='f1_weighted',
                              file_name='output/svm_learning_curve.png',
                              title='SVM Learning Curve (Polynomial Kernel)', ylabel='F1 Score (Weighted)')
    wandb.log({"learning_curve": wandb.Image('output/svm_learning_curve.png')})

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

    # Create a wandb Table and log it
    class_report_table = wandb.Table(columns=columns, data=table_data)
    wandb.log({"classification_report_table": class_report_table})

    # Log the f1_weighted score to wandb
    wandb.log({"F1 Score (Weighted)": weighted_f1_score})

    # Finish the wandb run
    wandb.finish()

# Define the sweep configuration
sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'F1 Score (Weighted)',
        'goal': 'maximize'
    },
    'parameters': {
        'C': {
            'values': [0.1, 0.3, 0.7, 1.0, 1.5]
        },
        'coef0': {
            'values': [0.1, 0.3, 0.7, 1.0, 1.5]
        },
        'degree': {
            'values': [1, 2, 3, 4, 5]
        }
    }
}

if __name__ == '__main__':
    if not os.path.exists('output'):
        os.makedirs('output')

    # Initialize a wandb sweep
    sweep_id = wandb.sweep(sweep_config, project='ml-experiments')

    # Run the sweep
    wandb.agent(sweep_id, function=main, count=10)
