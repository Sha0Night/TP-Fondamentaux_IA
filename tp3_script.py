import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import sys

# Rediriger la sortie standard vers un fichier pour la récupération des résultats
sys.stdout = open('resultats_tp3.txt', 'w', encoding='utf-8')

# Étape 1
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Pantalon', 'Pull', 'Robe', 'Manteau',
               'Sandale', 'Chemise', 'Sneaker', 'Sac', 'Bottine']

print("=== EXPLORATION DU CATALOGUE ===")
print(f"Train : {X_train.shape[0]} images de {X_train.shape[1]}x{X_train.shape[2]} pixels")
print(f"Test : {X_test.shape[0]} images")
print(f"Pixels : min={X_train.min()}, max={X_train.max()} (niveaux de gris 0–255)")
print(f"Catégories : {len(class_names)}")

print("\nDistribution des catégories (train) :")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"{class_names[u]:12s} : {c:5d} articles ({c/len(y_train)*100:.1f}%)")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    idx = np.where(y_train == i)[0][0]
    ax.imshow(X_train[idx], cmap='gray')
    ax.set_title(class_names[i], fontsize=10)
    ax.axis('off')

plt.suptitle('Catalogue Zalando — 1 exemple par catégorie', fontsize=13)
plt.tight_layout()
plt.savefig('catalogue_samples.png')
plt.close()

# Étape 2
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm = X_test.astype('float32') / 255.0

print(f"\nAprès normalisation : min={X_train_norm.min():.1f}, max={X_train_norm.max():.1f}")

X_train_flat = X_train_norm.reshape(-1, 784)
X_test_flat = X_test_norm.reshape(-1, 784)

print(f"Forme pour ML classique (aplatie) : {X_train_flat.shape}")
print(f"Forme pour réseau dense (grille) : {X_train_norm.shape}")

# Étape 3
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_flat, y_train)

y_pred_rf = rf.predict(X_test_flat)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"\nRandom Forest — Accuracy : {acc_rf*100:.1f}%")

# Étape 4
model_dense = keras.Sequential([
    keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model_dense.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n=== RESUME MODELE DENSE ===")
model_dense.summary()

history_dense = model_dense.fit(
    X_train_norm, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.15,
    verbose=1
)

# Étape 5
X_train_cnn = X_train_norm.reshape(-1, 28, 28, 1)
X_test_cnn  = X_test_norm.reshape(-1, 28, 28, 1)

print(f"\nForme pour CNN : {X_train_cnn.shape}")

model_cnn = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])

model_cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n=== RESUME MODELE CNN ===")
model_cnn.summary()

history_cnn = model_cnn.fit(
    X_train_cnn, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.15,
    verbose=1
)

# Étape 6
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_dense.history['accuracy'], 'b-', label='Train')
axes[0].plot(history_dense.history['val_accuracy'], 'r-', label='Validation')
axes[0].set_title('Réseau Dense (MLP)')
axes[0].set_xlabel('Époque')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_cnn.history['accuracy'], 'b-', label='Train')
axes[1].plot(history_cnn.history['val_accuracy'], 'r-', label='Validation')
axes[1].set_title('CNN')
axes[1].set_xlabel('Époque')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Courbes d'apprentissage — Diagnostic overfitting", fontsize=13)
plt.tight_layout()
plt.savefig('learning_curves.png')
plt.close()

# Étape 7
loss_dense, acc_dense = model_dense.evaluate(X_test_norm, y_test, verbose=0)
loss_cnn, acc_cnn = model_cnn.evaluate(X_test_cnn, y_test, verbose=0)

print("\n" + "="*55)
print(" COMPARAISON DES 3 APPROCHES — COMITÉ TECHNIQUE")
print("="*55)
print(f"{'Modèle':<20s}{'Accuracy':>10s}{'Taux erreur':>12s}")
print("-"*55)

print(f"{'Random Forest':<20s}{acc_rf*100:>9.1f}%{(1-acc_rf)*100:>11.1f}%")
print(f"{'Réseau Dense':<20s}{acc_dense*100:>9.1f}%{(1-acc_dense)*100:>11.1f}%")
print(f"{'CNN':<20s}{acc_cnn*100:>9.1f}%{(1-acc_cnn)*100:>11.1f}%")

print("-"*55)
print(f" Objectif business : taux d'erreur < 5.0%")
print("="*55)

y_pred_cnn = model_cnn.predict(X_test_cnn, verbose=0)
y_pred_classes = np.argmax(y_pred_cnn, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=class_names, yticklabels=class_names
)
plt.title('Matrice de confusion — CNN (articles Zalando)')
plt.ylabel('Catégorie réelle')
plt.xlabel('Catégorie prédite')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix_cnn.png')
plt.close()

print("\n=== RAPPORT PAR CATÉGORIE — CNN ===")
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# Étape 8
errors = np.where(y_pred_classes != y_test)[0]
print(f"\nErreurs : {len(errors)} / {len(y_test)} ({len(errors)/len(y_test)*100:.1f}%)")

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    idx = errors[i]
    ax.imshow(X_test[idx], cmap='gray')

    confiance = y_pred_cnn[idx, y_pred_classes[idx]] * 100
    ax.set_title(
        f"Prédit : {class_names[y_pred_classes[idx]]} ({confiance:.0f}%)\n"
        f"Réel : {class_names[y_test[idx]]}",
        fontsize=8,
        color='red'
    )
    ax.axis('off')

plt.suptitle('CNN — Articles mal classifiés (analyse qualité)', fontsize=13)
plt.tight_layout()
plt.savefig('erreurs_cnn.png')
plt.close()

print("\n=== TOP 5 CONFUSIONS LES PLUS FRÉQUENTES ===")
confusions = {}
for real, pred in zip(y_test[errors], y_pred_classes[errors]):
    pair = (class_names[real], class_names[pred])
    confusions[pair] = confusions.get(pair, 0) + 1

top_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:5]
for (real, pred), count in top_confusions:
    print(f"{real:12s} → classifié comme {pred:12s} : {count} erreurs")

# Étape 9
first_conv_layer = model_cnn.layers[0]
filters, biases = first_conv_layer.get_weights()
print(f"\nFiltres : {filters.shape}")

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(filters[:, :, 0, i], cmap='gray')
    ax.set_title(f'F{i+1}', fontsize=7)
    ax.axis('off')

plt.suptitle('Filtres appris par le CNN — 1re couche Conv2D (3x3)', fontsize=13)
plt.tight_layout()
plt.savefig('filtres_conv.png')
plt.close()

activation_model = keras.Model(
    inputs=model_cnn.inputs,
    outputs=model_cnn.layers[0].output
)

sample_idx = np.where(y_test == 7)[0][0]
sample = X_test_cnn[sample_idx:sample_idx+1]
activations = activation_model.predict(sample, verbose=0)
print(f"\nActivations : {activations.shape}")

fig, axes = plt.subplots(1, 9, figsize=(16, 2.5))
axes[0].imshow(X_test[sample_idx], cmap='gray')
axes[0].set_title('Original', fontsize=9)
axes[0].axis('off')

for i in range(8):
    axes[i+1].imshow(activations[0, :, :, i], cmap='viridis')
    axes[i+1].set_title(f'Filtre {i+1}', fontsize=9)
    axes[i+1].axis('off')

plt.suptitle(f"Activations du CNN — {class_names[y_test[sample_idx]]}", fontsize=12)
plt.tight_layout()
plt.savefig('activations_conv.png')
plt.close()

# Fin du script
sys.stdout.close()
