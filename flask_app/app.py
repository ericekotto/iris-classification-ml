"""
Application Flask - API de Prédiction des Espèces d'Iris
==========================================================

Cette API permet de prédire l'espèce d'une fleur iris 
à partir de ses caractéristiques morphologiques.

Endpoints:
- GET  /          : Page d'accueil (documentation)
- POST /predict   : Prédiction d'une espèce
- GET  /health    : Vérification du statut de l'API
- GET  /models    : Liste des modèles disponibles

Auteur: Votre Nom
Date: Janvier 2025
"""

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Initialisation de l'application Flask
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Chemin vers les modèles
MODELS_PATH = '../models/'

# Charger le modèle et le scaler au démarrage de l'application
try:
    with open(os.path.join(MODELS_PATH, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    with open(os.path.join(MODELS_PATH, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    print("✅ Modèle et scaler chargés avec succès !")
    
except FileNotFoundError as e:
    print(f"❌ Erreur : Fichier de modèle non trouvé - {e}")
    print("⚠️  Veuillez d'abord entraîner le modèle en exécutant les notebooks.")
    model = None
    scaler = None

# Noms des caractéristiques attendues
FEATURE_NAMES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Mapping des espèces
SPECIES_INFO = {
    'setosa': {
        'nom_scientifique': 'Iris setosa',
        'nom_commun': 'Iris sétacé',
        'description': 'Petites fleurs avec des pétales courts et larges',
        'couleur': 'Violet pâle à blanc'
    },
    'versicolor': {
        'nom_scientifique': 'Iris versicolor',
        'nom_commun': 'Iris versicolore',
        'description': 'Taille moyenne avec des pétales plus longs',
        'couleur': 'Bleu-violet'
    },
    'virginica': {
        'nom_scientifique': 'Iris virginica',
        'nom_commun': 'Iris de Virginie',
        'description': 'Grandes fleurs avec des pétales longs et larges',
        'couleur': 'Violet foncé'
    }
}


@app.route('/')
def home():
    """
    Page d'accueil avec documentation de l'API
    """
    documentation = {
        "nom": "API de Classification d'Iris",
        "version": "1.0.0",
        "description": "API pour prédire l'espèce d'une fleur iris",
        "endpoints": [
            {
                "route": "/",
                "methode": "GET",
                "description": "Documentation de l'API"
            },
            {
                "route": "/predict",
                "methode": "POST",
                "description": "Prédire l'espèce d'iris",
                "body_exemple": {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }
            },
            {
                "route": "/health",
                "methode": "GET",
                "description": "Vérifier l'état de l'API"
            },
            {
                "route": "/models",
                "methode": "GET",
                "description": "Informations sur les modèles disponibles"
            }
        ],
        "caracteristiques_requises": FEATURE_NAMES,
        "especes_possibles": list(SPECIES_INFO.keys()),
        "statut": "✅ Opérationnelle" if model is not None else "❌ Modèle non chargé"
    }
    
    return jsonify(documentation)


@app.route('/health', methods=['GET'])
def health():
    """
    Endpoint de santé pour vérifier que l'API fonctionne
    """
    health_status = {
        "statut": "healthy" if model is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "modele_charge": model is not None,
        "scaler_charge": scaler is not None
    }
    
    status_code = 200 if model is not None else 503
    return jsonify(health_status), status_code


@app.route('/models', methods=['GET'])
def models_info():
    """
    Informations sur les modèles disponibles
    """
    if model is None:
        return jsonify({"erreur": "Aucun modèle chargé"}), 503
    
    model_info = {
        "type_modele": type(model).__name__,
        "parametres": str(model.get_params()) if hasattr(model, 'get_params') else "Non disponible",
        "caracteristiques": FEATURE_NAMES,
        "nombre_caracteristiques": len(FEATURE_NAMES),
        "especes": list(SPECIES_INFO.keys())
    }
    
    return jsonify(model_info)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint principal de prédiction
    
    Reçoit un JSON avec les caractéristiques d'une fleur iris
    et retourne la prédiction de l'espèce
    
    Exemple de requête:
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    """
    
    # Vérifier que le modèle est chargé
    if model is None or scaler is None:
        return jsonify({
            "erreur": "Modèle non disponible",
            "message": "Le modèle n'a pas été chargé. Veuillez entraîner le modèle d'abord."
        }), 503
    
    try:
        # Récupérer les données JSON de la requête
        data = request.get_json()
        
        if not data:
            return jsonify({
                "erreur": "Données manquantes",
                "message": "Aucune donnée fournie dans la requête"
            }), 400
        
        # Vérifier que toutes les caractéristiques sont présentes
        missing_features = [f for f in FEATURE_NAMES if f not in data]
        if missing_features:
            return jsonify({
                "erreur": "Caractéristiques manquantes",
                "manquantes": missing_features,
                "requises": FEATURE_NAMES
            }), 400
        
        # Extraire les valeurs dans le bon ordre
        features = [float(data[f]) for f in FEATURE_NAMES]
        
        # Validation des valeurs (doivent être positives)
        if any(f < 0 for f in features):
            return jsonify({
                "erreur": "Valeurs invalides",
                "message": "Toutes les mesures doivent être positives"
            }), 400
        
        # Créer un array numpy et le normaliser
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # Faire la prédiction
        prediction = model.predict(features_scaled)[0]
        
        # Obtenir les probabilités si le modèle le supporte
        probabilities = {}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            classes = model.classes_ if hasattr(model, 'classes_') else list(SPECIES_INFO.keys())
            probabilities = {
                classe: float(prob) for classe, prob in zip(classes, proba)
            }
        
        # Construire la réponse
        response = {
            "prediction": prediction,
            "informations_espece": SPECIES_INFO.get(prediction, {}),
            "probabilites": probabilities,
            "caracteristiques_fournies": {
                name: value for name, value in zip(FEATURE_NAMES, features)
            },
            "timestamp": datetime.now().isoformat(),
            "modele_utilise": type(model).__name__
        }
        
        # Log de la prédiction
        print(f"[{datetime.now()}] Prédiction: {prediction} | Probabilités: {probabilities}")
        
        return jsonify(response), 200
    
    except ValueError as e:
        return jsonify({
            "erreur": "Erreur de format",
            "message": f"Les valeurs doivent être numériques: {str(e)}"
        }), 400
    
    except Exception as e:
        return jsonify({
            "erreur": "Erreur serveur",
            "message": str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Endpoint pour prédire plusieurs fleurs à la fois
    
    Exemple de requête:
    {
        "fleurs": [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
        ]
    }
    """
    
    if model is None or scaler is None:
        return jsonify({
            "erreur": "Modèle non disponible"
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'fleurs' not in data:
            return jsonify({
                "erreur": "Format invalide",
                "message": "Le JSON doit contenir une clé 'fleurs' avec une liste"
            }), 400
        
        fleurs = data['fleurs']
        
        if not isinstance(fleurs, list):
            return jsonify({
                "erreur": "Format invalide",
                "message": "'fleurs' doit être une liste"
            }), 400
        
        predictions = []
        
        for idx, fleur in enumerate(fleurs):
            try:
                # Vérifier les caractéristiques
                missing = [f for f in FEATURE_NAMES if f not in fleur]
                if missing:
                    predictions.append({
                        "index": idx,
                        "erreur": f"Caractéristiques manquantes: {missing}"
                    })
                    continue
                
                # Extraire et prédire
                features = [float(fleur[f]) for f in FEATURE_NAMES]
                features_array = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features_array)
                
                prediction = model.predict(features_scaled)[0]
                
                # Probabilités
                probabilities = {}
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_scaled)[0]
                    classes = model.classes_ if hasattr(model, 'classes_') else list(SPECIES_INFO.keys())
                    probabilities = {
                        classe: float(prob) for classe, prob in zip(classes, proba)
                    }
                
                predictions.append({
                    "index": idx,
                    "prediction": prediction,
                    "probabilites": probabilities,
                    "caracteristiques": {name: value for name, value in zip(FEATURE_NAMES, features)}
                })
                
            except Exception as e:
                predictions.append({
                    "index": idx,
                    "erreur": str(e)
                })
        
        return jsonify({
            "nombre_predictions": len(predictions),
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({
            "erreur": "Erreur serveur",
            "message": str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Gestionnaire d'erreur 404"""
    return jsonify({
        "erreur": "Route non trouvée",
        "message": "Cette route n'existe pas. Consultez la documentation à la route /"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Gestionnaire d'erreur 500"""
    return jsonify({
        "erreur": "Erreur serveur interne",
        "message": "Une erreur inattendue s'est produite"
    }), 500


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🌸 API DE CLASSIFICATION D'IRIS")
    print("=" * 80)
    print(f"📊 Modèle chargé : {type(model).__name__ if model else 'Aucun'}")
    print(f"🔧 Scaler chargé : {'Oui' if scaler else 'Non'}")
    print("=" * 80)
    print("\n🚀 Démarrage du serveur Flask...")
    print("📍 L'API sera accessible sur : http://127.0.0.1:5000")
    print("📖 Documentation : http://127.0.0.1:5000/")
    print("\n⚠️  Appuyez sur CTRL+C pour arrêter le serveur\n")
    
    # Lancer l'application
    app.run(debug=True, host='0.0.0.0', port=5000)