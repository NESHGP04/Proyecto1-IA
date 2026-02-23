import kagglehub 
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from mlp_model import MLP

# Download latest version 
# path = kagglehub.dataset_download("jasminebach/basiccolornames") 
# print("Path to dataset files:", path)

path = "/Users/Camila/.cache/kagglehub/datasets/jasminebach/basiccolornames/versions/1"

file_path = os.path.join(path, "final_data_colors.csv")

df = pd.read_csv(file_path)

print(df.head())
print(df.columns)

#inputs normalizados 
X = df[["red", "green", "blue"]].values / 255.0

labels = df["label"].unique()
print("Labels encontrados:", labels)

#diccionarios
label_to_idx = {label: i for i, label in enumerate(labels)}
idx_to_label = {i: label for label, i in label_to_idx.items()}

#labels a números
y_numeric = df["label"].map(label_to_idx).values

num_classes = len(labels)

y = np.zeros((len(y_numeric), num_classes))
y[np.arange(len(y_numeric)), y_numeric] = 1

#80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

model = MLP(
    input_size=3,
    hidden1=64,
    hidden2=32,
    output_size=num_classes
)

#entrenar
print("Entrenando")

model.train(X_train, y_train, epochs=3000, lr=0.01)

print("Entrenamiento finalizado.")

#evaluar modelo
predictions = model.predict(X_test)
true_labels = np.argmax(y_test, axis=1)

accuracy = np.mean(predictions == true_labels)

print("Accuracy en test:", accuracy)

#guardar modelo entrenado
np.save("W1.npy", model.W1)
np.save("b1.npy", model.b1)
np.save("W2.npy", model.W2)
np.save("b2.npy", model.b2)
np.save("W3.npy", model.W3)
np.save("b3.npy", model.b3)

#diccionarios
np.save("label_to_idx.npy", label_to_idx)
np.save("idx_to_label.npy", idx_to_label)

print("Modelo guardado correctamente.")