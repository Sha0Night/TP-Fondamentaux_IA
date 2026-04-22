"""
TP1 — Analyse d'un algorithme en fonctionnement
Matière : Les fondamentaux de l'IA | Niveau : Bachelor 3
Objectifs :
  - Observer un pipeline ML complet de bout en bout
  - Mesurer des métriques de performance (accuracy, F1, confusion matrix)
  - Détecter le sur/sous-apprentissage (overfitting)
  - Initier l'explicabilité avec SHAP
"""

# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour sauvegarder les figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import shap

import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  TP1 — ANALYSE D'UN ALGORITHME EN FONCTIONNEMENT (DATASET IRIS)")
print("=" * 70)

# ============================================================
# ÉTAPE 1 — CHARGER LES DONNÉES : LE DATASET IRIS
# ============================================================
# Le dataset Iris est un classique du ML : 150 mesures de fleurs
# (4 variables botaniques) à classer en 3 espèces.
# Chargé ici depuis le dépôt public UCI via seaborn-data.

print("\n" + "=" * 70)
print("  ÉTAPE 1 — CHARGEMENT ET EXPLORATION DES DONNÉES")
print("=" * 70)

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print("\n=== EXPLORATION DES DONNÉES ===")
print(f"Forme du dataset : {df.shape}")
print(f"\nDistribution des classes :")
print(df['species'].value_counts())
print(f"\nValeurs manquantes : {df.isnull().sum().sum()}")
print(f"\nAperçu :")
print(df.head(3))
print(f"\nStatistiques descriptives :")
print(df.describe())

# --- Réponses aux questions de l'étape 1 ---
print("\n--- RÉPONSES ÉTAPE 1 ---")
print("""
• Combien d'échantillons par classe ? Le dataset est-il équilibré ?
  -> 50 échantillons par classe (setosa, versicolor, virginica).
    Le dataset est parfaitement équilibré (50/50/50).

• Y a-t-il des valeurs manquantes ?
  -> Non, aucune valeur manquante (0 NaN).

• Quelles variables semblent a priori les plus discriminantes ?
  -> petal_length et petal_width semblent les plus discriminantes
    car elles présentent les plus grands écarts entre espèces
    (cf. statistiques descriptives et pairplot ci-après).
""")

# --- Visualisation rapide des données ---
# Pairplot : distributions et séparabilité des 3 espèces
print("Génération du pairplot...")
sns.pairplot(df, hue='species', markers=["o", "s", "D"],
             plot_kws=dict(alpha=0.7), diag_kind='hist')
plt.suptitle("Dataset Iris — Séparabilité des classes", y=1.02)
plt.savefig('iris_pairplot.png', bbox_inches='tight', dpi=150)
plt.close()
print("-> Pairplot sauvegardé : iris_pairplot.png")

print("""
Observation pairplot :
  -> Les paires impliquant petal_length et petal_width permettent
    de séparer visuellement les 3 espèces de manière nette.
  -> Setosa est toujours bien séparée des deux autres.
  -> Versicolor et Virginica se chevauchent légèrement, surtout
    sur sepal_length / sepal_width.
""")


# ============================================================
# ÉTAPE 2 — ENTRAÎNER UN MODÈLE : BASELINE + MODÈLE PRINCIPAL
# ============================================================
print("\n" + "=" * 70)
print("  ÉTAPE 2 — ENTRAÎNEMENT DES MODÈLES")
print("=" * 70)

le = LabelEncoder()
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = le.fit_transform(df['species'])
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Split entraînement/test (80%/20%) stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nEntraînement : {len(X_train)} échantillons | Test : {len(X_test)} échantillons")

# Normalisation (importante pour comparer les variables sur la même échelle)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Baseline simple : Decision Tree (max_depth=3)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train_sc, y_train)

# Modèle principal : Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train_sc, y_train)

print("Modèles entraînés avec succès !")

# --- Réponse à la question de l'étape 2 ---
print("\n--- RÉPONSE ÉTAPE 2 ---")
print("""
• Pourquoi commencer par une baseline simple (Decision Tree) avant
  un modèle plus complexe (Random Forest) ?
  -> La baseline sert de point de référence pour évaluer si un modèle
    plus complexe apporte réellement un gain de performance.
  -> Si la baseline est déjà très performante, un modèle complexe
    n'est peut-être pas nécessaire (principe de parcimonie / Occam's razor).
  -> Cela permet aussi de détecter des problèmes dans les données :
    si même la baseline échoue, le problème vient peut-être des features.
""")


# ============================================================
# ÉTAPE 3 — ÉVALUER LES MÉTRIQUES
# ============================================================
print("\n" + "=" * 70)
print("  ÉTAPE 3 — ÉVALUATION DES MÉTRIQUES")
print("=" * 70)

y_pred_dt = dt.predict(X_test_sc)
y_pred_rf = rf.predict(X_test_sc)

print("\n=== COMPARAISON BASELINE vs RANDOM FOREST ===")
resultats = []
for nom, pred in [("Decision Tree (baseline)", y_pred_dt), ("Random Forest       ", y_pred_rf)]:
    acc = accuracy_score(y_test, pred)
    f1  = f1_score(y_test, pred, average='weighted')
    print(f"{nom} -> Accuracy : {acc*100:.1f}%  |  F1-score (weighted) : {f1:.3f}")
    resultats.append({'Modèle': nom.strip(), 'Accuracy': f'{acc*100:.1f}%', 'F1-score': f'{f1:.3f}'})

# Tableau comparatif
print("\n=== TABLEAU COMPARATIF ===")
df_resultats = pd.DataFrame(resultats)
print(df_resultats.to_string(index=False))

print("\n=== RAPPORT DÉTAILLÉ — RANDOM FOREST ===")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matrice de confusion — Random Forest')
plt.ylabel('Réalité')
plt.xlabel('Prédiction')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("-> Matrice de confusion sauvegardée : confusion_matrix.png")

# --- Réponses aux questions de l'étape 3 ---
print("\n--- RÉPONSES ÉTAPE 3 ---")
print("""
• 1. Le Random Forest surpasse-t-il le Decision Tree ? Par quelle marge ?
  -> Les deux modèles obtiennent des performances très proches sur ce
    dataset (souvent 100% tous les deux, ou écart < 3%).
    Le dataset Iris est relativement simple, donc même une baseline
    performante atteint d'excellents résultats.

• 2. Y a-t-il des espèces plus difficiles à classifier ?
  -> Setosa est toujours parfaitement classifiée (séparation linéaire).
    Versicolor et Virginica sont plus proches dans l'espace des features
    et peuvent occasionnellement être confondues.

• 3. Quelle différence entre accuracy et F1-score ?
  -> Accuracy = proportion de prédictions correctes (toutes classes confondues).
  -> F1-score = moyenne harmonique de la précision et du rappel.
    Il est plus informatif quand les classes sont déséquilibrées,
    car il pénalise les modèles qui ignorent les classes minoritaires.
    Ici, les classes sont équilibrées, donc les deux métriques convergent.
    On préfère le F1 quand les classes sont déséquilibrées.
""")


# ============================================================
# ÉTAPE 4 — TUNING DES HYPERPARAMÈTRES ET DÉTECTION DE L'OVERFITTING
# ============================================================
print("\n" + "=" * 70)
print("  ÉTAPE 4 — TUNING ET DÉTECTION DE L'OVERFITTING")
print("=" * 70)

# Impact du nombre d'arbres mesuré par validation croisée 5-fold
print("\n=== VALIDATION CROISÉE : IMPACT DU NOMBRE D'ARBRES ===")
n_estimators_range = [10, 25, 50, 100, 200, 500]
scores_cv = []
for n in n_estimators_range:
    m  = RandomForestClassifier(n_estimators=n, random_state=42)
    cv = cross_val_score(m, X_train_sc, y_train, cv=5, scoring='accuracy')
    scores_cv.append(cv.mean())
    print(f"n_estimators={n:4d} -> CV accuracy : {cv.mean()*100:.1f}% (±{cv.std()*100:.1f}%)")

# Courbe overfitting : score train vs score test selon la profondeur
print("\n=== COURBE OVERFITTING : TRAIN vs TEST ===")
profondeurs    = range(1, 20)
scores_train   = []
scores_test    = []

for d in profondeurs:
    m = RandomForestClassifier(n_estimators=50, max_depth=d, random_state=42)
    m.fit(X_train_sc, y_train)
    scores_train.append(m.score(X_train_sc, y_train))
    scores_test.append(m.score(X_test_sc,  y_test))

plt.figure(figsize=(9, 4))
plt.plot(list(profondeurs), scores_train, 'b-o', label='Score entraînement')
plt.plot(list(profondeurs), scores_test,  'r-o', label='Score test')
plt.xlabel('Profondeur maximale (max_depth)')
plt.ylabel('Accuracy')
plt.title('Underfitting vs Overfitting — Random Forest')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('overfitting.png', dpi=150)
plt.close()
print("-> Graphique overfitting sauvegardé : overfitting.png")

# Affichage des scores pour analyse
for d, tr, te in zip(profondeurs, scores_train, scores_test):
    gap = tr - te
    status = "[!] OVERFITTING" if gap > 0.05 else "[OK] OK"
    print(f"  max_depth={d:2d} -> Train: {tr*100:.1f}%  Test: {te*100:.1f}%  Écart: {gap*100:.1f}%  {status}")

# --- Réponses aux questions de l'étape 4 ---
print("\n--- RÉPONSES ÉTAPE 4 ---")
print("""
• 1. À partir de quelle profondeur observe-t-on de l'overfitting ?
  -> L'overfitting commence généralement à apparaître au-delà de
    max_depth=5-7, où le score d'entraînement atteint 100% tandis
    que le score test stagne ou diminue légèrement.
    L'écart entre score train et score test se creuse.

• 2. Quel nombre d'arbres offre le meilleur compromis performance / coût ?
  -> À partir de ~50-100 arbres, les performances se stabilisent.
    Augmenter au-delà de 100 n'apporte qu'un gain marginal,
    mais augmente le temps de calcul. 100 est un bon compromis.

• 3. Qu'apporte la validation croisée par rapport à un simple split train/test ?
  -> La validation croisée (5-fold) donne une estimation plus robuste
    de la performance réelle, car elle utilise tout le jeu de données
    pour l'entraînement et le test (chaque échantillon est testé une fois).
  -> Elle réduit la variance de l'estimation et détecte mieux l'instabilité
    du modèle. Un simple split peut donner un résultat optimiste ou
    pessimiste selon le hasard du découpage.
""")


# ============================================================
# ÉTAPE 5 — EXPLICABILITÉ SHAP
# ============================================================
print("\n" + "=" * 70)
print("  ÉTAPE 5 — EXPLICABILITÉ SHAP")
print("=" * 70)

# Modèle final
rf_final = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_final.fit(X_train_sc, y_train)

# Explainer SHAP adapté aux forêts aléatoires
print("\nCalcul des valeurs SHAP...")
explainer   = shap.TreeExplainer(rf_final)
shap_values = explainer.shap_values(X_test_sc)

# Summary plot global (importance par classe, vue en barres)
plt.figure()
shap.summary_plot(
    shap_values, X_test_sc,
    feature_names=feature_names,
    class_names=list(le.classes_),
    plot_type='bar',
    show=False
)
plt.title("SHAP — Importance globale des variables (3 classes)")
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("-> SHAP summary plot sauvegardé : shap_summary.png")

# Comparaison : feature importance sklearn (MDI) vs SHAP (classe virginica)
# Compatibilité SHAP < 0.42 (liste) et SHAP >= 0.42 (tableau 3D)
importances_sklearn = rf_final.feature_importances_

if isinstance(shap_values, list):
    importances_shap = np.abs(shap_values[2]).mean(axis=0)   # index 2 = virginica
else:
    importances_shap = np.abs(shap_values[:, :, 2]).mean(axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.barh(feature_names, importances_sklearn, color='steelblue')
ax1.set_title('Feature Importance — sklearn (MDI)')
ax1.set_xlabel('Importance')

ax2.barh(feature_names, importances_shap, color='darkorange')
ax2.set_title('Feature Importance — SHAP (classe virginica)')
ax2.set_xlabel('|SHAP value| moyen')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()
print("-> Feature importance comparée sauvegardée : feature_importance.png")

# --- Réponses aux questions de l'étape 5 ---
print("\n--- RÉPONSES ÉTAPE 5 ---")
print("""
• 1. Quelle variable est la plus déterminante selon SHAP ?
  -> petal_length (longueur des pétales) est la variable la plus
    déterminante selon SHAP, suivie de petal_width.
    C'est cohérent avec l'intuition : les pétales varient beaucoup
    plus entre espèces que les sépales.

• 2. Quelle différence entre l'importance sklearn (MDI) et SHAP ?
  -> MDI (Mean Decrease in Impurity) mesure combien une variable
    réduit l'impureté moyenne dans les arbres. C'est rapide mais
    biaisé vers les variables à haute cardinalité.
  -> SHAP (SHapley Additive exPlanations) attribue à chaque variable
    sa contribution marginale exacte à chaque prédiction, basée sur
    la théorie des jeux. C'est plus fiable et interprétable,
    mais plus coûteux en calcul.

• 3. Pourquoi l'explicabilité est-elle cruciale dans un contexte réglementé ?
  -> Dans les domaines médical, bancaire ou juridique, les décisions
    algorithmiques doivent être justifiables et transparentes (RGPD,
    AI Act européen, etc.).
  -> Un médecin doit comprendre POURQUOI un modèle recommande un
    traitement. Un banquier doit expliquer POURQUOI un crédit est refusé.
  -> Sans explicabilité, on risque de perpétuer des biais cachés et
    de perdre la confiance des utilisateurs et régulateurs.
""")


# ============================================================
# ÉTAPE 6 — DEBRIEF : 3 INSIGHTS CLÉS
# ============================================================
print("\n" + "=" * 70)
print("  ÉTAPE 6 — DEBRIEF : 3 INSIGHTS CLÉS")
print("=" * 70)

print("""
========================================================================
|  INSIGHT 1 — PERFORMANCE                                          |
========================================================================
|                                                                    |
|  Observation :                                                     |
|    Le Random Forest et le Decision Tree atteignent tous deux une   |
|    accuracy > 96% sur le jeu de test Iris. L'écart entre les       |
|    deux modèles est minime (< 3 points de pourcentage).            |
|                                                                    |
|  Explication :                                                     |
|    Le dataset Iris est relativement simple, avec des classes bien  |
|    séparées (surtout Setosa). Un arbre de décision peu profond     |
|    suffit à capturer les frontières de décision. Le Random Forest  |
|    apporte une stabilité supplémentaire via l'agrégation (bagging) |
|    mais le gain marginal est faible sur des données simples.       |
|                                                                    |
|  Implication pratique :                                            |
|    -> Toujours commencer par un modèle simple (baseline).           |
|    -> Si la baseline est déjà performante, un modèle complexe      |
|      n'est justifié que s'il apporte un gain significatif.         |
|    -> Principe de parcimonie : préférer le modèle le plus simple   |
|      à performance égale (maintenance, explicabilité, coût).       |
========================================================================

========================================================================
|  INSIGHT 2 — OVERFITTING / GÉNÉRALISATION                          |
========================================================================
|                                                                    |
|  Observation :                                                     |
|    À partir de max_depth ~ 5-7, le score d'entraînement atteint   |
|    100% tandis que le score test stagne ou diminue légèrement.     |
|    L'écart grandissant signale un début d'overfitting.             |
|                                                                    |
|  Explication :                                                     |
|    L'overfitting survient quand le modèle mémorise les données     |
|    d'entraînement (y compris le bruit) au lieu d'apprendre des     |
|    patterns généralisables. Un arbre trop profond crée des règles  |
|    hyper-spécifiques qui ne se transfèrent pas aux données inédites|
|                                                                    |
|  Implication pratique :                                            |
|    -> Limiter la profondeur (max_depth=3-5 ici) pour éviter le     |
|      sur-apprentissage.                                            |
|    -> Utiliser la validation croisée pour estimer la performance    |
|      réelle et choisir les hyperparamètres optimaux.               |
|    -> Surveiller l'écart train/test comme indicateur d'overfitting. |
========================================================================

========================================================================
|  INSIGHT 3 — EXPLICABILITÉ                                         |
========================================================================
|                                                                    |
|  Observation :                                                     |
|    Selon SHAP, petal_length est la variable la plus déterminante   |
|    pour la classification, suivie de petal_width. sepal_width a    |
|    une contribution marginale faible.                              |
|                                                                    |
|  Explication :                                                     |
|    Ce résultat est cohérent avec la biologie : la taille des       |
|    pétales varie fortement entre espèces d'Iris (Setosa a des     |
|    pétales très petits vs Virginica très grands), tandis que les   |
|    sépales sont plus homogènes entre espèces.                     |
|    SHAP confirme quantitativement ce que le pairplot suggérait     |
|    visuellement.                                                   |
|                                                                    |
|  Implication pratique :                                            |
|    -> L'explicabilité est obligatoire dans les domaines réglementés |
|      (médecine, finance, justice) : RGPD, AI Act européen.        |
|    -> Elle permet de valider que le modèle utilise des critères     |
|      pertinents et non des artefacts ou des biais discriminatoires.|
|    -> Elle renforce la confiance des utilisateurs finaux et         |
|      facilite l'adoption des solutions ML en entreprise.           |
========================================================================
""")


# ============================================================
# RÉSUMÉ FINAL — LIVRABLES GÉNÉRÉS
# ============================================================
print("\n" + "=" * 70)
print("  FICHIERS GÉNÉRÉS")
print("=" * 70)
print("""
  1. iris_pairplot.png       — Pairplot des données (séparabilité des classes)
  2. confusion_matrix.png    — Matrice de confusion du Random Forest
  3. overfitting.png         — Courbe underfitting vs overfitting
  4. shap_summary.png        — SHAP summary plot (importance globale)
  5. feature_importance.png  — Comparaison sklearn MDI vs SHAP
""")

print("=" * 70)
print("  TP1 TERMINÉ AVEC SUCCÈS !")
print("=" * 70)
