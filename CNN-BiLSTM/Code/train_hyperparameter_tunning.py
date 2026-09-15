


import numpy as np
import matplotlib.pyplot as plt


from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


from dataset_loader_new import prepare_generators
#from model import build_model
from model_bilstm import build_model
from config import *

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import matthews_corrcoef

import matplotlib.pyplot as plt
import seaborn as sns
# ======================================================
# Load Dataset
# ======================================================
print("\n Preparing Generators......\n")
train_generator,val_generator,test_generator=prepare_generators()
print("Training Batches:",len(train_generator))
print("Validation Batches:",len(val_generator))
print("Testing Batches:",len(test_generator))


# ======================================================
# Class Weights
# ======================================================

print("\nCalculating Class Weights...")

# train_labels = np.array(train_generator.labels)

# weights = compute_class_weight(
#     class_weight="balanced",
#     classes=np.unique(train_labels),
#     y=train_labels
# )

# class_weights = {
#     0: weights[0],
#     1: weights[1]
# }

#print(class_weights)




# ======================================================
# Build Model
# ======================================================

model = build_model()

model.summary()


# ======================================================
# Callbacks
# ======================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_NAME,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# reduce_lr = ReduceLROnPlateau(
#     monitor="val_loss",
#     factor=0.5,
#     patience=5,
#     verbose=1
# )


# ======================================================
# Train
# ======================================================

history = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=EPOCHS,

    #class_weight=class_weights,

    callbacks=[
      early_stop,
      checkpoint
     # reduce_lr
    ],

    verbose=1
)

#from tensorflow.keras.models import load_model
#model=load_model(MODEL_NAME)


# ======================================================
# Evaluate
# ======================================================


# =====================================================
# Prediction
# =====================================================

y_prob = model.predict(
    test_generator,
    verbose=1
)

y_prob = y_prob.ravel()
print("y_prob:",y_prob)
print("NaN:",np.isnan(y_prob).any())
print("shape:",y_prob.shape)

y_pred = (y_prob > 0.5).astype(int)


y_test=[]
for i in range(len(test_generator)):
    _,y=test_generator[i]
    y_test.extend(y)
y_test=np.array(y_test)
print("test labels:",y_test)
print("unique classes:",np.unique(y_test))





# =====================================================
# Metrics
# =====================================================
print("length of y_test:",len(y_test))
print("length of y_prob:",len(y_prob))
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_prob)

mcc = matthews_corrcoef(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

specificity = TN / (TN + FP)

print("\n")
print("="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"Accuracy      : {accuracy*100:.2f}%")
print(f"Precision     : {precision*100:.2f}%")
print(f"Recall        : {recall*100:.2f}%")
print(f"Sensitivity   : {recall*100:.2f}%")
print(f"Specificity   : {specificity*100:.2f}%")
print(f"F1 Score      : {f1*100:.2f}%")
print(f"ROC AUC       : {auc:.4f}")
print(f"MCC           : {mcc:.4f}")

print("="*60)

print("\nClassification Report\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal","MI"]
))


# ======================================================
# Accuracy Plot
# ======================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"])

plt.plot(history.history["val_accuracy"])

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(["Train","Validation"])

plt.grid(True)
plt.savefig("accuracy_plot(bilstm:denoised).png")

plt.close()


# ======================================================
# Loss Plot
# ======================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"])

plt.plot(history.history["val_loss"])

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(["Train","Validation"])

plt.grid(True)
plt.savefig("loss_plot(bilstm:denoised).png")


plt.close()


print("\nTraining Completed Successfully")
