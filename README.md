# 🌸 Projet Classification des Fleurs Iris

## 📋 Table des matières
- [Présentation](#présentation)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Exercices](#exercices)
- [Déploiement](#déploiement)
- [Tests](#tests)
- [Auteur](#auteur)

---

## 📖 Présentation

Ce projet est réalisé dans le cadre du module **Introduction à l'IA et Machine Learning (INFO4111)** de l'École Normale Supérieure de Yaoundé.

### Objectifs du TP
1. ✅ Familiarisation avec Python pour la data science
2. ✅ Utilisation des bibliothèques de ML (scikit-learn, pandas, numpy)
3. ✅ Exploration et visualisation de données
4. ✅ Création et entraînement de modèles de classification
5. ✅ Évaluation des performances
6. ✅ Déploiement avec Flask et Streamlit

### Dataset
Le dataset **Iris** contient 150 échantillons de fleurs iris répartis en 3 espèces :
- 🌺 **Iris Setosa**
- 🌷 **Iris Versicolor**
- 🌸 **Iris Virginica**

Chaque échantillon possède 4 caractéristiques :
- Longueur du sépale (cm)
- Largeur du sépale (cm)
- Longueur du pétale (cm)
- Largeur du pétale (cm)

---

## 📁 Structure du projet

```
iris_classification_project/
│
├── data/
│   └── iris.csv                          # Dataset
│
├── notebooks/
│   ├── 01_exploration_donnees.ipynb      # Exercices 1-5
│   ├── 02_modelisation.ipynb             # Étapes 3-6
│   └── 03_optimisation.ipynb             # Étape 7
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model_training.py
│   ├── visualization.py
│   └── utils.py
│
├── models/
│   ├── knn_model.pkl                     # Modèle KNN
│   ├── scaler.pkl                        # Scaler
│   ├── best_model.pkl                    # Meilleur modèle
│   └── models_comparison.csv             # Comparaison
│
├── flask_app/
│   └── app.py                            # API Flask
│
├── streamlit_app/
│   └── dashboard.py                      # Dashboard
│
├── visualizations/                       # Graphiques sauvegardés
│
├── test_api.py                           # Tests de l'API
├── requirements.txt                      # Dépendances
└── README.md                             # Ce fichier
```

---

## 🔧 Installation

### Prérequis
- **Python 3.8+** installé
- **Anaconda** (recommandé) ou environnement virtuel Python
- **VS Code** avec extension Python et Jupyter

### Étape 1 : Cloner ou créer le projet

```bash
# Créer le dossier du projet
mkdir iris_classification_project
cd iris_classification_project
```

### Étape 2 : Créer l'environnement virtuel

**Option A : Avec Anaconda (Recommandé)**
```bash
# Créer l'environnement
conda create -n iris_env python=3.10 -y

# Activer l'environnement
conda activate iris_env
```

**Option B : Avec venv**
```bash
# Créer l'environnement
python -m venv iris_env

# Activer
# Windows:
iris_env\Scripts\activate
# Mac/Linux:
source iris_env/bin/activate
```

### Étape 3 : Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 4 : Télécharger le dataset

**Option A : Téléchargement automatique**
```python
# Créer et exécuter ce script Python
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df.to_csv('data/iris.csv', index=False)
print("✅ Dataset créé : data/iris.csv")
```

**Option B : Téléchargement manuel**
- Télécharger depuis : https://archive.ics.uci.edu/ml/datasets/iris
- Placer dans `data/iris.csv`

---

## 🚀 Utilisation

### 1️⃣ Exploration des données (Notebooks)

#### Dans VS Code :
1. Ouvrir VS Code dans le dossier du projet
2. Installer l'extension "Jupyter" de Microsoft
3. Créer un nouveau notebook ou ouvrir ceux fournis
4. Sélectionner le kernel Python (iris_env)
5. Exécuter les cellules une par une

#### Dans Jupyter Notebook classique :
```bash
# Lancer Jupyter
jupyter notebook

# Ou Jupyter Lab
jupyter lab
```

### 2️⃣ Exécuter les exercices

Les notebooks sont organisés par étapes :

**📓 Notebook 1 : Exploration (Exercices 1-5)**
- Distribution des espèces
- Analyse des variables quantitatives
- Nuages de points
- Boîtes à moustaches
- Corrélations

**📓 Notebook 2 : Modélisation (Étapes 3-6)**
- Préparation des données
- Entraînement du modèle KNN
- Évaluation des performances
- Matrice de confusion

**📓 Notebook 3 : Optimisation (Étape 7)**
- Optimisation des hyperparamètres
- Comparaison de différents algorithmes
- Sélection du meilleur modèle

### 3️⃣ Lancer l'API Flask

```bash
# Se placer dans le dossier flask_app
cd flask_app

# Lancer le serveur
python app.py
```

L'API sera accessible sur : **http://127.0.0.1:5000**

#### Endpoints disponibles :

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Documentation de l'API |
| `/health` | GET | Statut de l'API |
| `/models` | GET | Informations sur les modèles |
| `/predict` | POST | Prédire une espèce |
| `/predict/batch` | POST | Prédictions par lot |

#### Exemple de requête :

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### 4️⃣ Lancer le Dashboard Streamlit

**Dans un nouveau terminal :**

```bash
# Activer l'environnement
conda activate iris_env  # ou source iris_env/bin/activate

# Se placer dans streamlit_app
cd streamlit_app

# Lancer Streamlit
streamlit run dashboard.py
```

Le dashboard sera accessible sur : **http://localhost:8501**

#### Fonctionnalités du dashboard :
- 🏠 Page d'accueil avec statistiques
- 📈 Exploration interactive des données
- 🤖 Prédiction en temps réel
- 📊 Analyse des performances
- 🔍 Comparaison des espèces

---

## 📝 Exercices

### Exercice 1 : Distribution des espèces
- Afficher l'effectif de chaque espèce
- Créer différents types de graphiques
- Déterminer la meilleure représentation

**Code de démarrage :**
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../data/iris.csv')
print(df['species'].value_counts())

sns.countplot(data=df, x='species')
plt.title('Distribution des espèces')
plt.show()
```

### Exercice 2 : Variables quantitatives
- Résumer chaque variable (moyenne, médiane, écart-type)
- Créer des histogrammes
- Analyser les distributions

### Exercice 3 : Étude bivariée
- Créer des nuages de points
- Analyser les corrélations
- Identifier les relations entre variables

### Exercice 4 : Boxplots
- Comparer les distributions par espèce
- Identifier les outliers
- Interpréter les différences

### Exercice 5 : Corrélations
- Calculer la matrice de corrélation
- Créer un graphique radar
- Proposer des visualisations avancées

---

## 🧪 Tests

### Tester l'API Flask

```bash
# Lancer les tests automatiques
python test_api.py
```

Ce script teste :
- ✅ La connexion à l'API
- ✅ Tous les endpoints
- ✅ Les prédictions pour chaque espèce
- ✅ La gestion des erreurs
- ✅ Les prédictions par lot

### Tests manuels avec Postman

1. **Installer Postman** : https://www.postman.com/downloads/
2. **Importer la collection** (créer les requêtes manuellement)
3. **Tester chaque endpoint**

---

## 📊 Performances attendues

Avec le modèle KNN optimisé, vous devriez obtenir :
- **Accuracy** : ~96-98%
- **Precision** : ~96-98%
- **Recall** : ~96-98%
- **F1-Score** : ~96-98%

Les meilleurs modèles sont généralement :
1. 🥇 SVM (RBF kernel)
2. 🥈 KNN optimisé
3. 🥉 Random Forest

---

## 🔍 Dépannage

### Problème : Le modèle ne se charge pas
**Solution** :
```bash
# Vérifier que les fichiers existent
ls models/
# Doit afficher : knn_model.pkl, scaler.pkl, best_model.pkl

# Réentraîner si nécessaire
# Exécuter le notebook 02_modelisation.ipynb
```

### Problème : L'API Flask ne démarre pas
**Solution** :
```bash
# Vérifier l'installation de Flask
pip install flask --upgrade

# Vérifier le port 5000
# Sur Windows :
netstat -an | findstr 5000
# Sur Mac/Linux :
lsof -i :5000

# Tuer le processus si nécessaire
```

### Problème : Streamlit ne se connecte pas à l'API
**Solution** :
1. Vérifier que Flask est démarré
2. Vérifier l'URL dans `dashboard.py` (ligne `API_URL`)
3. Essayer le mode "En local" dans le dashboard

### Problème : Erreur d'import de bibliothèques
**Solution** :
```bash
# Réinstaller toutes les dépendances
pip install -r requirements.txt --force-reinstall

# Ou installer individuellement
pip install pandas scikit-learn matplotlib seaborn flask streamlit
```

---

## 📚 Ressources supplémentaires

### Documentation
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Pandas](https://pandas.pydata.org/docs/)
- [Flask](https://flask.palletsprojects.com/)
- [Streamlit](https://docs.streamlit.io/)

### Tutoriels
- [Machine Learning avec Python](https://www.youtube.com/watch?v=7eh4d6sabA0)
- [Flask Tutorial](https://www.youtube.com/watch?v=Z1RJmh_OqeA)
- [Streamlit Tutorial](https://www.youtube.com/watch?v=JwSS70SZdyM)

### Dataset Iris
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- [Article original de R.A. Fisher](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1469-1809.1936.tb02137.x)

---

## 👨‍💻 Auteur

**Nom** : Votre Nom  
**Module** : Introduction à l'IA et Machine Learning (INFO4111)  
**Institution** : École Normale Supérieure de Yaoundé  
**Année** : 2024-2025  
**Enseignant** : Dr. Stéphane C.K. TEKOUABOU

---

## 📄 Licence

Ce projet est réalisé à des fins éducatives dans le cadre du module INFO4111.

---

## 🙏 Remerciements

- Dr. Stéphane C.K. TEKOUABOU pour l'encadrement
- UCI Machine Learning Repository pour le dataset
- La communauté open-source Python

---

## 📞 Support

Pour toute question ou problème :
1. Consulter la section [Dépannage](#dépannage)
2. Vérifier les [Issues GitHub](#) (si applicable)
3. Contacter l'enseignant

---

**Bonne chance avec votre TP ! 🚀🌸**