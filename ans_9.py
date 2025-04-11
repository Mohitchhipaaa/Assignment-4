import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

# Step 1: Create dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Step 2: Split into train, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize classifier
model = xgb.XGBClassifier(
    eval_metric="logloss",
    use_label_encoder=False,
    n_estimators=200,
    learning_rate=0.1
)

# Step 4: Fit with early stopping
evals_result = {}
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    early_stopping_rounds=10,
    verbose=False,
    evals_result=evals_result
)

# Step 5: Plot training vs validation error
train_logloss = evals_result['validation_0']['logloss']
val_logloss = evals_result['validation_1']['logloss']

plt.plot(train_logloss, label='Train Log Loss')
plt.plot(val_logloss, label='Validation Log Loss')
plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.title('Training vs Validation Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()