# 📁 Recommandation Intelligente de Formations

## 🎯 Objectif du Projet

Ce projet a pour but de développer un système intelligent capable de recommander les **trois formations les plus pertinentes** à chaque membre à partir de leurs caractéristiques personnelles, académiques, et associatives.

L’approche repose sur un modèle de machine learning supervisé entraîné sur un jeu de données réel, comprenant des informations sur les étudiants et les formations qu’ils ont suivies.

---

## 🧠 Approche Adoptée

### 1. Prétraitement des données
- Nettoyage des noms de formations (fusion de doublons : *Negociations* vs *Negotiation*).
- Correction des Filières invalides via un petit modèle de classification supervisée (Random Forest).
- Transformation multi-label : chaque membre peut avoir jusqu'à **trois formations**.
- Encodage des catégories (`Sexe`, `Experience_Professionnelle`, `Cellule`) et mise à l’échelle automatique des données d’entrée.
- Encodage des formations avec `LabelEncoder`.

### 2. Modélisation avec XGBoost
- Utilisation du modèle `XGBClassifier` avec `multi:softprob` pour obtenir des probabilités par formation.
- Entraînement sur des données explosées (chaque ligne correspond à une formation d’un membre).
- Agrégation des prédictions par membre via la **moyenne des probabilités**.

---

## 🔐 Seuil de Confiance (Threshold)

Par défaut, un modèle de classification prédit les 3 classes les plus probables. Certaines de ces prédictions peuvent être peu fiables.

Pour y remédier, nous avons introduit un **seuil de confiance dynamique** (`threshold = 0.1`) :

- Ne retenir que les formations ayant une probabilité supérieure au seuil.
- Permettre au modèle de ne pas proposer systématiquement 3 formations si la confiance n’est pas suffisante.

Une analyse de seuil a été réalisée pour trouver la meilleure balance entre couverture et précision.

---

## 📊 Résultats Obtenus

L’évaluation repose sur plusieurs métriques adaptées aux problèmes multi-labels :

| Métrique                                 | Score         | Interprétation                                  |
|------------------------------------------|---------------|-------------------------------------------------|
| ✅ **Hit Rate (1/3)**                    | 98.91%        | Le modèle propose **au moins une formation correcte** dans 98.5% des cas |
| ✅ **2/3 Match Accuracy**                | 78.93%        | Deux formations sont correctes dans 75% des cas |
| 🎯 **Exact Match (3/3)**                 | 56.54%        | Trois formations parfaitement prédites dans 45% des cas |
| 🎯 **F1 Score (micro)**                  | 88.43%        | Bonne qualité globale de prédiction             |
| 🎯 **F1 Score (macro)**                  | 88.32%        | Performance moyenne sur toutes les classes      |
| ⚠️  **Subset Accuracy (Strict)**         | 56.54%        | Trois formations exactes et uniquement          |
| ⚠️  **Hamming Loss**                     | 1.71%         | 10.5% des prédictions sont incorrectes          |

Although the micro and macro F1 scores are high due to partial correct predictions (one or two formations correct out of three), the strict subset accuracy remains around 56%, reflecting the difficulty of achieving exact full matches.
---

## 🌐 Application Web

Une application web a été développée pour exploiter ce modèle :

- **Backend** : FastAPI (serveur Python léger pour l'API de prédiction)
- **Frontend** : React.js (formulaire dynamique pour saisir les caractéristiques utilisateur)
- Affichage dynamique des 3 meilleures formations recommandées avec les scores de confiance.
- Mapping intelligent des ID de formations vers leurs noms réels.

---

## 🛠️ Instructions

**Installation :**

```bash
pip install -r requirements.txt
```
**Exécution :**

Lancer l'API backend :
```bash
uvicorn backend.app:app --reload
```
Lancer l'application React frontend :
```bash
uvicorn backend.app:app --reload
```