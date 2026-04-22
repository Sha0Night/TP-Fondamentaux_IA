import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import shap

# ==========================================
# Étape 2 — Préparation des données
# ==========================================

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 
    'label', 'difficulty'
]

url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+_20Percent.txt"
print("Chargement des données NSL-KDD...")
df = pd.read_csv(url, header=None, names=columns)

df['label_binary'] = (df['label'] != 'normal').astype(int)
print(f"Forme : {df.shape}")
print(f"Normal (0) : {(df['label_binary']==0).sum()} | Attaque (1) : {(df['label_binary']==1).sum()}")

for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop(['label', 'difficulty', 'label_binary'], axis=1)
y = df['label_binary']
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")
print(f"Nombre de features : {len(feature_names)}")

# ==========================================
# Étape 3 — Modélisation
# ==========================================

print("\nEntraînement des modèles...")
modeles = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
}

resultats = {}

for nom, modele in modeles.items():
    modele.fit(X_train_sc, y_train)
    pred = modele.predict(X_test_sc)
    acc = accuracy_score(y_test, pred)
    f1_mac = f1_score(y_test, pred, average='macro')
    f1_wei = f1_score(y_test, pred, average='weighted')
    resultats[nom] = {'accuracy': acc, 'f1_macro': f1_mac, 'f1_weighted': f1_wei}
    print(f"{nom:22s} → Accuracy : {acc*100:.1f}% | F1-macro : {f1_mac:.3f} | F1-weighted : {f1_wei:.3f}")

df_resultats = pd.DataFrame(resultats).T
print("\n=== TABLEAU COMPARATIF ===")
print(df_resultats.round(3))

df_resultats[['accuracy', 'f1_macro']].plot(kind='bar', figsize=(9, 4))
plt.title('Comparaison des modèles')
plt.ylabel('Score')
plt.xticks(rotation=20)
plt.ylim(0.8, 1.0)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig('comparaison_modeles.png')
print("Graphique sauvegardé : comparaison_modeles.png")
plt.close()

# ==========================================
# Étape 4 — Évaluation approfondie
# ==========================================

# Random Forest and XGBoost performed very similarly, let's use XGBoost as best or Random Forest
NOM_MEILLEUR = "Random Forest"
meilleur = modeles[NOM_MEILLEUR]
y_pred = meilleur.predict(X_test_sc)

print(f"\n=== RAPPORT DÉTAILLÉ — {NOM_MEILLEUR} ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matrice de confusion — {NOM_MEILLEUR}')
plt.ylabel('Réalité')
plt.xlabel('Prédiction')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Graphique sauvegardé : confusion_matrix.png")
plt.close()

# Évolution Random Forest
print("\nÉvaluation de l'évolution du Random Forest...")
n_estimators_range = [10, 25, 50, 100, 200]
scores_rf = []
for n in n_estimators_range:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_temp.fit(X_train_sc, y_train)
    pred_temp = rf_temp.predict(X_test_sc)
    scores_rf.append(f1_score(y_test, pred_temp, average='macro'))

plt.figure(figsize=(8, 4))
plt.plot(n_estimators_range, scores_rf, 'g-o')
plt.xlabel("Nombre d'arbres (n_estimators)")
plt.ylabel("F1-score macro")
plt.title("Évolution des performances — Random Forest")
plt.grid(True)
plt.tight_layout()
plt.savefig('rf_evolution.png')
print("Graphique sauvegardé : rf_evolution.png")
plt.close()

# ==========================================
# Étape 5 — Explicabilité SHAP
# ==========================================

print("\nGénération de l'explicabilité SHAP...")
rf_model = modeles["Random Forest"]
X_sample = X_test_sc[:200]

# XGBoost or Random Forest SHAP
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    sv = shap_values[1]
elif len(shap_values.shape) == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

plt.figure()
shap.summary_plot(
    sv, X_sample, feature_names=feature_names,
    max_display=15, show=False
)
plt.title("SHAP — Top 15 variables influentes")
plt.tight_layout()
plt.savefig('shap_summary.png', bbox_inches='tight')
print("Graphique sauvegardé : shap_summary.png")
plt.close()

print("\nPipeline terminé avec succès.")
