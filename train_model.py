# Import Libraries
import os
import itertools

import pandas as pd
import numpy as np
import matplotlib

# Use a non-GUI backend for GitHub Actions
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    f1_score,
    recall_score,
    accuracy_score
)

sns.set(style='white')


# --------------------------------------------------
# Load Data
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset = pd.read_csv(os.path.join(BASE_DIR, 'iris.csv'))

# Clean Feature Names
dataset.columns = [
    colname.strip().replace(' (cm)', '').replace(' ', '_')
    for colname in dataset.columns
]

print("Dataset Loaded Successfully")
print(dataset.head())
print("Dataset Shape:", dataset.shape)


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

dataset['sepal_length_width_ratio'] = (
    dataset['sepal_length'] / dataset['sepal_width']
)

dataset['petal_length_width_ratio'] = (
    dataset['petal_length'] / dataset['petal_width']
)


# Select Features
dataset = dataset[
    [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
        'sepal_length_width_ratio',
        'petal_length_width_ratio',
        'target'
    ]
]


# --------------------------------------------------
# Split Data
# --------------------------------------------------

train_data, test_data = train_test_split(
    dataset,
    test_size=0.2,
    random_state=44,
    stratify=dataset['target']
)


# Prepare Training and Testing Data

X_train = train_data.drop('target', axis=1).values.astype('float32')
y_train = train_data['target'].values.astype('int32')

X_test = test_data.drop('target', axis=1).values.astype('float32')
y_test = test_data['target'].values.astype('int32')


# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------

logreg = LogisticRegression(
    C=0.0001,
    solver='lbfgs',
    max_iter=100
)

logreg.fit(X_train, y_train)

predictions_lr = logreg.predict(X_test)


# Logistic Regression Metrics

cm_lr = confusion_matrix(y_test, predictions_lr)

f1_lr = f1_score(y_test, predictions_lr, average='micro')

prec_lr = precision_score(
    y_test, predictions_lr, average='micro', zero_division=0
)

recall_lr = recall_score(
    y_test, predictions_lr, average='micro', zero_division=0
)

train_acc_lr = accuracy_score(
    y_train, logreg.predict(X_train)
) * 100

test_acc_lr = accuracy_score(
    y_test, predictions_lr
) * 100


# --------------------------------------------------
# Random Forest Classification
# --------------------------------------------------

rf_reg = RandomForestClassifier(random_state=44)

rf_reg.fit(X_train, y_train)

predictions_rf = rf_reg.predict(X_test)


# Random Forest Metrics

f1_rf = f1_score(y_test, predictions_rf, average='micro')

prec_rf = precision_score(
    y_test, predictions_rf, average='micro', zero_division=0
)

recall_rf = recall_score(
    y_test, predictions_rf, average='micro', zero_division=0
)

train_acc_rf = accuracy_score(
    y_train, rf_reg.predict(X_train)
) * 100

test_acc_rf = accuracy_score(
    y_test, predictions_rf
) * 100


# --------------------------------------------------
# Confusion Matrix Plotting Function
# --------------------------------------------------

def plot_cm(
    cm,
    target_name,
    title="Confusion Matrix",
    normalize=True
):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    plt.figure(figsize=(12, 8))

    plt.imshow(
        cm,
        interpolation='nearest',
        cmap=plt.get_cmap('Blues')
    )

    plt.title(title)
    plt.colorbar()

    if target_name is not None:
        tick_marks = np.arange(len(target_name))

        plt.xticks(tick_marks, target_name, rotation=45)
        plt.yticks(tick_marks, target_name)

    if normalize:
        cm = cm.astype('float') / cm.sum(
            axis=1
        )[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(
        range(cm.shape[0]),
        range(cm.shape[1])
    ):

        if normalize:
            text_value = "{:0.4f}".format(cm[i, j])
        else:
            text_value = "{:,}".format(cm[i, j])

        plt.text(
            j,
            i,
            text_value,
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.tight_layout()

    plt.ylabel('True Label')

    plt.xlabel(
        'Predicted Label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
            accuracy, misclass
        )
    )

    plt.savefig(
        os.path.join(BASE_DIR, 'ConfusionMatrix.png'),
        dpi=120
    )

    plt.close()


# Plot Confusion Matrix

target_name = np.array([
    'setosa',
    'versicolor',
    'virginica'
])

plot_cm(
    cm_lr,
    target_name,
    title="Confusion Matrix (Logistic Regression)",
    normalize=True
)


# --------------------------------------------------
# Random Forest Feature Importance
# --------------------------------------------------

importances = rf_reg.feature_importances_

labels = dataset.columns[:-1]

feature_df = pd.DataFrame(
    list(zip(labels, importances)),
    columns=['feature', 'importance']
)

features = feature_df.sort_values(
    by='importance',
    ascending=False
)


# Feature Importance Plot

plt.figure(figsize=(12, 6))

ax = sns.barplot(
    x='importance',
    y='feature',
    data=features
)

ax.set_xlabel('Importance', fontsize=14)
ax.set_ylabel('Feature', fontsize=14)

ax.set_title(
    'Random Forest Feature Importances',
    fontsize=12
)

plt.tight_layout()

plt.savefig(
    os.path.join(BASE_DIR, 'FeatureImportance.png'),
    dpi=120
)

plt.close()


# --------------------------------------------------
# Save Model Scores to File
# --------------------------------------------------

with open(
    os.path.join(BASE_DIR, 'scores.txt'),
    'w'
) as score:

    score.write("Random Forest Train Accuracy: %2.1f%%\n" % train_acc_rf)
    score.write("Random Forest Test Accuracy: %2.1f%%\n" % test_acc_rf)
    score.write("F1 Score: %2.1f%%\n" % (f1_rf * 100))
    score.write("Recall Score: %2.1f%%\n" % (recall_rf * 100))
    score.write("Precision Score: %2.1f%%\n" % (prec_rf * 100))

    score.write("\n\n")

    score.write("Logistic Regression Train Accuracy: %2.1f%%\n" % train_acc_lr)
    score.write("Logistic Regression Test Accuracy: %2.1f%%\n" % test_acc_lr)
    score.write("F1 Score: %2.1f%%\n" % (f1_lr * 100))
    score.write("Recall Score: %2.1f%%\n" % (recall_lr * 100))
    score.write("Precision Score: %2.1f%%\n" % (prec_lr * 100))


# --------------------------------------------------
# Print Results
# --------------------------------------------------

print("\nLogistic Regression Results")
print("----------------------------")
print("Training Accuracy:", train_acc_lr)
print("Testing Accuracy:", test_acc_lr)

print("\nRandom Forest Results")
print("---------------------")
print("Training Accuracy:", train_acc_rf)
print("Testing Accuracy:", test_acc_rf)

print("\nModel Training Completed Successfully!")

print("\nGenerated Files:")
print("1. ConfusionMatrix.png")
print("2. FeatureImportance.png")
print("3. scores.txt")
