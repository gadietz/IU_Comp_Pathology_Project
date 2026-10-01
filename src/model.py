import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    roc_auc_score, 
    roc_curve
)
import matplotlib.pyplot as plt

NPZ_PATH = './output/pneumoniamnist.npz'

def load_and_preprocess_data():
    #Loads images from NPZ, flattens them, and normalizes pixel intensities to [0, 1].
    
    data = np.load(NPZ_PATH)
    
    # Extract images & labels
    X_train = data['train_images']
    y_train = data['train_labels'].squeeze()
    
    X_val = data['val_images']
    y_val = data['val_labels'].squeeze()
    
    X_test = data['test_images']
    y_test = data['test_labels'].squeeze()

    # Flatten (28x28 -> 784 features) and scale pixels [0, 255] -> [0.0, 1.0]
    X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_val_flat   = X_val.reshape(X_val.shape[0], -1) / 255.0
    X_test_flat  = X_test.reshape(X_test.shape[0], -1) / 255.0

    return (X_train_flat, y_train), (X_val_flat, y_val), (X_test_flat, y_test)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data()


#combine train and validation sets to use in grid search to find best params
X_combined = np.vstack((X_train, X_val))
Y_combined = np.concatenate((y_train, y_val))

#Create index array: -1 for Training samples, 0 for Validation samples
#PredefinedSplit will fit on indices marked -1 and score on index 0
test_fold = np.array([-1] * len(X_train) + [0] * len(X_val))
ps = PredefinedSplit(test_fold) 

param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [500, 1000]
}

# Run GridSearch
model = LogisticRegression()
grid_search = GridSearchCV(
    estimator=model, 
    param_grid=param_grid, 
    scoring='accuracy', 
    cv=ps,                  # Uses exact train/val split
    refit=False,            # Prevents automatic refitting on combined train+val
    n_jobs=-1
)

grid_search.fit(X_combined, Y_combined)

print("Best Parameters on Validation Set:", grid_search.best_params_)
print(f"Best Validation Accuracy: {grid_search.best_score_:.4f}")

#Output
#Best Parameters on Validation Set: {'C': 0.1, 'max_iter': 500, 'solver': 'liblinear'}
#Best Validation Accuracy: 0.9580

#Selected these parameters for final model
model = LogisticRegression(**grid_search.best_params_)
model.fit(X_train, y_train)

test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"Final Test Accuracy: {test_acc:.4f}")

test_preds = model.predict(X_test)
test_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for Pneumonia class

# Compute Confusion Matrix metrics
# Matrix format: [[TN, FP], [FN, TP]]
cm = confusion_matrix(y_test, test_preds)
tn, fp, fn, tp = cm.ravel()

# Metrics
acc = accuracy_score(y_test, test_preds)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall for Pneumonia
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Recall for Normal
auc = roc_auc_score(y_test, test_probs)

print("\n" + "-"*35)
print("FINAL TEST SET EVALUATION")
print("-"*35)
print(f"Accuracy:                    {acc:.4f} ({acc*100:.2f}%)")
print(f"Sensitivity (Pneumonia Recall): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
print(f"Specificity (Normal Recall):    {specificity:.4f} ({specificity*100:.2f}%)")
print(f"ROC AUC Score:               {auc:.4f}")
    
print("\nConfusion Matrix breakdown:")
print(f"  True Negatives  (Normal correctly identified):    {tn}")
print(f"  False Positives (Normal misclassified as Pneumonia): {fp}")
print(f"  False Negatives (Pneumonia missed):              {fn}")
print(f"  True Positives  (Pneumonia correctly identified): {tp}")

# Plot & Save Confusion Matrix & ROC Curve

OUTPUT_DIR = './output'
EVAL_PLOT_PATH = os.path.join(OUTPUT_DIR, 'model_evaluation.png')

#Generate 1x2 Plot Grid (Confusion Matrix & ROC Curve)
os.makedirs(OUTPUT_DIR, exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=['Normal (0)', 'Pneumonia (1)']
)
disp.plot(cmap='Blues', ax=ax1, values_format='d', colorbar=False)
ax1.set_title("Test Confusion Matrix")

# Subplot 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, test_probs)
ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate (1 - Specificity)')
ax2.set_ylabel('True Positive Rate (Sensitivity)')
ax2.set_title('Test ROC Curve')
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(EVAL_PLOT_PATH, dpi=150)
plt.close()

print(f"\nSaved combined evaluation plot to: {EVAL_PLOT_PATH}")