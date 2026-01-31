"""
Dashboard Streamlit - Interface de Prédiction des Iris
======================================================

Application web interactive pour visualiser les données Iris
et effectuer des prédictions en temps réel.

Fonctionnalités:
- Visualisation des données
- Prédiction interactive
- Analyse des performances du modèle
- Comparaison des espèces

Auteur: Votre Nom
Date: Janvier 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Configuration de la page
st.set_page_config(
    page_title="Classification des Iris",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Chemins des fichiers
DATA_PATH = 'data/iris.csv'
MODEL_PATH = 'models/iris_model.joblib'

# Fonction pour charger les données
@st.cache_data
def load_data():
    """Charge le dataset iris"""
    try:
        df = pd.read_csv(DATA_PATH,sep=';')
        return df
    except FileNotFoundError:
        st.error("❌ Fichier iris.csv non trouvé. Veuillez vérifier le chemin.")
        return None

# Fonction pour charger le modèle
@st.cache_resource
def load_model_and_scaler():
    """Charge le modèle et le scaler"""
    try:
        with open(os.path.join(MODELS_PATH, 'best_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(MODELS_PATH, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.warning("⚠️ Modèle non trouvé. Certaines fonctionnalités seront limitées.")
        return None, None

# Fonction pour faire une prédiction via l'API
def predict_via_api(features):
    """Effectue une prédiction via l'API Flask"""
    try:
        data = {
            'sepal_length': features[0],
            'sepal_width': features[1],
            'petal_length': features[2],
            'petal_width': features[3]
        }
        response = requests.post(f"{API_URL}/predict", json=data, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

# Fonction pour faire une prédiction locale
def predict_local(features, model, scaler):
    """Effectue une prédiction en local"""
    if model is None or scaler is None:
        return None
    
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    prediction = model.predict(features_scaled)[0]
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features_scaled)[0]
        classes = model.classes_ if hasattr(model, 'classes_') else ['setosa', 'versicolor', 'virginica']
        probabilities = {classe: float(prob) for classe, prob in zip(classes, proba)}
    else:
        probabilities = {}
    
    return {
        'prediction': prediction,
        'probabilites': probabilities
    }

# Informations sur les espèces
SPECIES_INFO = {
    'setosa': {
        'emoji': '🌺',
        'nom_scientifique': 'Iris setosa',
        'nom_commun': 'Iris sétacé',
        'description': 'Petites fleurs avec des pétales courts et larges',
        'couleur': 'Violet pâle à blanc',
        'color_code': '#FF6B6B'
    },
    'versicolor': {
        'emoji': '🌷',
        'nom_scientifique': 'Iris versicolor',
        'nom_commun': 'Iris versicolore',
        'description': 'Taille moyenne avec des pétales plus longs',
        'couleur': 'Bleu-violet',
        'color_code': '#4ECDC4'
    },
    'virginica': {
        'emoji': '🌸',
        'nom_scientifique': 'Iris virginica',
        'nom_commun': 'Iris de Virginie',
        'description': 'Grandes fleurs avec des pétales longs et larges',
        'couleur': 'Violet foncé',
        'color_code': '#45B7D1'
    }
}

# ===============================================================================
# INTERFACE PRINCIPALE
# ===============================================================================

def main():
    """Fonction principale de l'application"""
    
    # En-tête
    st.markdown('<div class="main-header">🌸 Dashboard Classification des Iris 🌸</div>', 
                unsafe_allow_html=True)
    
    # Charger les données et le modèle
    df = load_data()
    model, scaler = load_model_and_scaler()
    
    if df is None:
        st.error("Impossible de charger les données. Veuillez vérifier votre configuration.")
        return
    
    # Barre latérale - Navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Sélectionnez une page:",
        ["🏠 Accueil",
         "📈 Exploration des Données",
         "🤖 Prédiction Interactive",
         "📊 Performances du Modèle",
         "🔍 Analyse Comparative"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Projet:** Classification des Iris\n\n"
        "**Dataset:** Iris (150 échantillons)\n\n"
        "**Modèle:** Machine Learning\n\n"
        "**Module:** INFO4111"
    )
    
    # ===== PAGE ACCUEIL =====
    if page == "🏠 Accueil":
        show_home_page(df, model)
    
    # ===== PAGE EXPLORATION =====
    elif page == "📈 Exploration des Données":
        show_exploration_page(df)
    
    # ===== PAGE PRÉDICTION =====
    elif page == "🤖 Prédiction Interactive":
        show_prediction_page(model, scaler)
    
    # ===== PAGE PERFORMANCES =====
    elif page == "📊 Performances du Modèle":
        show_performance_page(df, model, scaler)
    
    # ===== PAGE ANALYSE COMPARATIVE =====
    elif page == "🔍 Analyse Comparative":
        show_comparison_page(df)


# ===============================================================================
# PAGE ACCUEIL
# ===============================================================================

def show_home_page(df, model):
    """Affiche la page d'accueil"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Total d'échantillons", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🌸 Espèces", df['Species'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔢 Caractéristiques", len(df.columns) - 1)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Présentation du projet
    st.markdown("## 📖 À propos du projet")
    st.write("""
    Ce projet vise à classifier automatiquement les espèces d'iris en fonction de leurs
    caractéristiques morphologiques. Le dataset contient 150 échantillons de fleurs iris
    répartis en trois espèces différentes.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Objectifs")
        st.write("""
        - Explorer et visualiser les données
        - Entraîner un modèle de classification
        - Évaluer les performances
        - Déployer une application interactive
        """)
    
    with col2:
        st.markdown("### 🔧 Technologies utilisées")
        st.write("""
        - **Python** : Langage de programmation
        - **Scikit-learn** : Machine Learning
        - **Flask** : API REST
        - **Streamlit** : Interface web
        - **Pandas/Numpy** : Traitement des données
        - **Matplotlib/Seaborn** : Visualisation
        """)
    
    st.markdown("---")
    
    # Aperçu des données
    st.markdown("## 🔍 Aperçu des données")
    
    tab1, tab2, tab3 = st.tabs(["📋 Premières lignes", "📊 Statistiques", "🎨 Distribution"])
    
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab2:
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab3:
        fig, ax = plt.subplots(figsize=(10, 6))
        df['Species'].value_counts().plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_title('Distribution des espèces', fontsize=16, fontweight='bold')
        ax.set_xlabel('Espèce')
        ax.set_ylabel('Nombre d\'échantillons')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    # Informations sur les espèces
    st.markdown("---")
    st.markdown("## 🌸 Les trois espèces d'Iris")
    
    cols = st.columns(3)
    
    for idx, (species, info) in enumerate(SPECIES_INFO.items()):
        with cols[idx]:
            st.markdown(f"### {info['emoji']} {species.capitalize()}")
            st.write(f"**{info['nom_scientifique']}**")
            st.write(f"*{info['nom_commun']}*")
            st.write(f"📝 {info['description']}")
            st.write(f"🎨 Couleur: {info['couleur']}")


# ===============================================================================
# PAGE EXPLORATION DES DONNÉES
# ===============================================================================

def show_exploration_page(df):
    """Affiche la page d'exploration des données"""
    
    st.title("📈 Exploration des Données")
    
    # Choix de la visualisation
    viz_type = st.selectbox(
        "Sélectionnez le type de visualisation:",
        ["Distribution des variables",
         "Nuages de points",
         "Boîtes à moustaches",
         "Matrice de corrélation",
         "Graphique radar"]
    )
    
    if viz_type == "Distribution des variables":
        show_distribution_plots(df)
    
    elif viz_type == "Nuages de points":
        show_scatter_plots(df)
    
    elif viz_type == "Boîtes à moustaches":
        show_box_plots(df)
    
    elif viz_type == "Matrice de corrélation":
        show_correlation_matrix(df)
    
    elif viz_type == "Graphique radar":
        show_radar_chart(df)


def show_distribution_plots(df):
    """Affiche les histogrammes de distribution"""
    
    st.markdown("### 📊 Distribution des variables quantitatives")
    
    variables = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    titles = ['Longueur du sépale', 'Largeur du sépale', 
              'Longueur du pétale', 'Largeur du pétale']
    
    col1, col2 = st.columns(2)
    
    for idx, (var, title) in enumerate(zip(variables, titles)):
        with col1 if idx % 2 == 0 else col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Histogramme avec courbe de densité
            df[var].hist(bins=20, ax=ax, alpha=0.7, edgecolor='black')
            ax.axvline(df[var].mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Moyenne: {df[var].mean():.2f}')
            ax.axvline(df[var].median(), color='blue', linestyle='--', linewidth=2,
                      label=f'Médiane: {df[var].median():.2f}')
            
            ax.set_title(f'Distribution - {title}', fontsize=14, fontweight='bold')
            ax.set_xlabel(f'{title} (cm)')
            ax.set_ylabel('Fréquence')
            ax.legend()
            ax.grid(alpha=0.3)
            
            st.pyplot(fig)
            plt.close()


def show_scatter_plots(df):
    """Affiche les nuages de points"""
    
    st.markdown("### 🔵 Nuages de points")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Axe X:", 
                            ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],
                            index=2)
    
    with col2:
        y_var = st.selectbox("Axe Y:", 
                            ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],
                            index=3)
    
    # Graphique interactif avec Plotly
    fig = px.scatter(df, x=x_var, y=y_var, color='Species',
                     title=f'{x_var.replace("_", " ").title()} vs {y_var.replace("_", " ").title()}',
                     labels={x_var: x_var.replace("_", " ").title(),
                            y_var: y_var.replace("_", " ").title()},
                     color_discrete_map={'setosa': '#FF6B6B',
                                        'versicolor': '#4ECDC4',
                                        'virginica': '#45B7D1'})
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def show_box_plots(df):
    """Affiche les boîtes à moustaches"""
    
    st.markdown("### 📦 Boîtes à moustaches")
    
    variable = st.selectbox("Sélectionnez une variable:",
                           ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
    
    fig = px.box(df, x='Species', y=variable, color='Species',
                 title=f'{variable.replace("_", " ").title()} par espèce',
                 color_discrete_map={'setosa': '#FF6B6B',
                                    'versicolor': '#4ECDC4',
                                    'virginica': '#45B7D1'})
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def show_correlation_matrix(df):
    """Affiche la matrice de corrélation"""
    
    st.markdown("### 🔗 Matrice de corrélation")
    
    variables = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    correlation = df[variables].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Matrice de corrélation des caractéristiques', 
                fontsize=16, fontweight='bold')
    
    st.pyplot(fig)
    plt.close()


def show_radar_chart(df):
    """Affiche un graphique radar"""
    
    st.markdown("### 🎯 Graphique Radar - Moyennes par espèce")
    
    variables = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    species_means = df.groupby('Species')[variables].mean()
    
    fig = go.Figure()
    
    for species in species_means.index:
        fig.add_trace(go.Scatterpolar(
            r=species_means.loc[Species].values,
            theta=[v.replace('_', ' ').title() for v in variables],
            fill='toself',
            name=species.capitalize()
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 7])
        ),
        showlegend=True,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ===============================================================================
# PAGE PRÉDICTION INTERACTIVE
# ===============================================================================

def show_prediction_page(model, scaler):
    """Affiche la page de prédiction interactive"""
    
    st.title("🤖 Prédiction Interactive")
    
    st.write("""
    Entrez les mesures d'une fleur iris pour prédire son espèce.
    Vous pouvez utiliser les curseurs ou entrer les valeurs manuellement.
    """)
    
    # Mode de prédiction
    st.markdown("### ⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prediction_mode = st.radio(
            "Mode de prédiction:",
            ["🔌 Via API Flask", "💻 En local (modèle chargé)"]
        )
    
    with col2:
        input_mode = st.radio(
            "Mode de saisie:",
            ["🎚️ Curseurs", "⌨️ Manuel"]
        )
    
    st.markdown("---")
    st.markdown("### 📏 Mesures de la fleur")
    
    # Saisie des caractéristiques
    if input_mode == "🎚️ Curseurs":
        col1, col2 = st.columns(2)
        
        with col1:
            sepal_length = st.slider("Longueur du sépale (cm)", 4.0, 8.0, 5.8, 0.1)
            sepal_width = st.slider("Largeur du sépale (cm)", 2.0, 4.5, 3.0, 0.1)
        
        with col2:
            petal_length = st.slider("Longueur du pétale (cm)", 1.0, 7.0, 4.3, 0.1)
            petal_width = st.slider("Largeur du pétale (cm)", 0.1, 2.5, 1.3, 0.1)
    
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            sepal_length = st.number_input("Longueur du sépale (cm)", 4.0, 8.0, 5.8, 0.1)
            sepal_width = st.number_input("Largeur du sépale (cm)", 2.0, 4.5, 3.0, 0.1)
        
        with col2:
            petal_length = st.number_input("Longueur du pétale (cm)", 1.0, 7.0, 4.3, 0.1)
            petal_width = st.number_input("Largeur du pétale (cm)", 0.1, 2.5, 1.3, 0.1)
    
    features = [sepal_length, sepal_width, petal_length, petal_width]
    
    # Affichage des valeurs saisies
    st.markdown("#### 📋 Résumé des mesures")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sépale L", f"{sepal_length} cm")
    col2.metric("Sépale l", f"{sepal_width} cm")
    col3.metric("Pétale L", f"{petal_length} cm")
    col4.metric("Pétale l", f"{petal_width} cm")
    
    # Bouton de prédiction
    st.markdown("---")
    
    if st.button("🔮 Prédire l'espèce", type="primary", use_container_width=True):
        
        # Effectuer la prédiction
        if prediction_mode == "🔌 Via API Flask":
            with st.spinner("🔄 Appel de l'API en cours..."):
                result = predict_via_api(features)
            
            if result is None:
                st.error("""
                ❌ **Impossible de contacter l'API Flask**
                
                Assurez-vous que le serveur Flask est démarré:
                ```bash
                cd flask_app
                python app.py
                ```
                """)
                return
        
        else:
            if model is None or scaler is None:
                st.error("❌ Modèle non chargé. Veuillez entraîner le modèle d'abord.")
                return
            
            result = predict_local(features, model, scaler)
        
        # Afficher le résultat
        if result:
            prediction = result['prediction']
            probabilities = result.get('probabilites', {})
            
            species_data = SPECIES_INFO[prediction]
            
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            st.markdown(f"## {species_data['emoji']} Espèce prédite: **{prediction.upper()}**")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"**{species_data['nom_scientifique']}**")
                st.write(f"*{species_data['nom_commun']}*")
                st.write(f"📝 {species_data['description']}")
                st.write(f"🎨 {species_data['couleur']}")
            
            with col2:
                if probabilities:
                    st.markdown("### 📊 Probabilités par espèce")
                    
                    # Graphique des probabilités
                    prob_df = pd.DataFrame({
                        'Espèce': list(probabilities.keys()),
                        'Probabilité': [p * 100 for p in probabilities.values()]
                    })
                    
                    fig = px.bar(prob_df, x='Espèce', y='Probabilité',
                                color='Espèce',
                                color_discrete_map={'setosa': '#FF6B6B',
                                                   'versicolor': '#4ECDC4',
                                                   'virginica': '#45B7D1'})
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Afficher les pourcentages
                    for species, prob in probabilities.items():
                        st.progress(prob, text=f"{species.capitalize()}: {prob*100:.1f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recommandations
            st.markdown("---")
            st.markdown("### 💡 Analyse de la prédiction")
            
            max_prob = max(probabilities.values()) if probabilities else 1.0
            
            if max_prob > 0.95:
                st.success("✅ **Très haute confiance** - La prédiction est très fiable.")
            elif max_prob > 0.80:
                st.info("ℹ️ **Bonne confiance** - La prédiction est fiable.")
            else:
                st.warning("⚠️ **Confiance modérée** - Il pourrait y avoir une ambiguïté.")


# ===============================================================================
# PAGE PERFORMANCES DU MODÈLE
# ===============================================================================

def show_performance_page(df, model, scaler):
    """Affiche les performances du modèle"""
    
    st.title("📊 Performances du Modèle")
    
    if model is None:
        st.warning("⚠️ Modèle non chargé. Veuillez d'abord entraîner le modèle.")
        return
    
    # Charger les résultats de comparaison
    try:
        results_df = pd.read_csv(os.path.join(MODELS_PATH, 'models_comparison.csv'))
        
        st.markdown("### 🏆 Comparaison des modèles")
        st.dataframe(results_df.sort_values('Accuracy (%)', ascending=False), 
                    use_container_width=True)
        
        # Graphiques de comparaison
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(results_df, x='Modèle', y='Accuracy (%)',
                        title='Accuracy par modèle',
                        color='Accuracy (%)',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(results_df, x='Modèle', y='F1-Score (%)',
                        title='F1-Score par modèle',
                        color='F1-Score (%)',
                        color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
    
    except FileNotFoundError:
        st.info("ℹ️ Fichier de comparaison non trouvé. Exécutez le notebook d'optimisation.")


# ===============================================================================
# PAGE ANALYSE COMPARATIVE
# ===============================================================================

def show_comparison_page(df):
    """Affiche l'analyse comparative des espèces"""
    
    st.title("🔍 Analyse Comparative des Espèces")
    
    st.markdown("### 📊 Statistiques par espèce")
    
    variable = st.selectbox("Sélectionnez une variable:",
                           ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
    
    # Tableau comparatif
    stats = df.groupby('Species')[variable].agg(['mean', 'std', 'min', 'max'])
    stats.columns = ['Moyenne', 'Écart-type', 'Minimum', 'Maximum']
    
    st.dataframe(stats.style.highlight_max(axis=0, color='lightgreen')
                           .highlight_min(axis=0, color='lightcoral'),
                use_container_width=True)
    
    # Visualisation comparative
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.violin(df, x='Species', y=variable, color='Species',
                       title=f'Distribution de {variable}',
                       color_discrete_map={'setosa': '#FF6B6B',
                                          'versicolor': '#4ECDC4',
                                          'virginica': '#45B7D1'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, x='Species', y=variable, color='Species',
                    title=f'Boxplot de {variable}',
                    color_discrete_map={'setosa': '#FF6B6B',
                                       'versicolor': '#4ECDC4',
                                       'virginica': '#45B7D1'})
        st.plotly_chart(fig, use_container_width=True)


# ===============================================================================
# LANCEMENT DE L'APPLICATION
# ===============================================================================

if __name__ == '__main__':
    main()