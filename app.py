import streamlit as st
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="TP-KNN MNIST", page_icon="👥", layout="centered")
st.header("EL MEHDI - IAENG")
st.title("👥 TP-KNN : Détection du Chiffre 0")
st.write("Classification binaire avec le Dataset MNIST (0 vs Autres chiffres)")

# 1. Chargement des données (Optimisé à 700 images)
@st.cache_data
def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data[:700] / 255.0, mnist.target[:700]
    return X, y

with st.spinner("Chargement des données..."):
    X, y = load_data()

# 2. Transformation pour le KNN (Approche Binaire)
y_binary = np.where(y == '0', 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# 3. Entraînement du KNN (Mis en cache)
@st.cache_resource
def train_knn(X_t, y_t):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_t, y_t)
    return model

with st.spinner("Entraînement du modèle KNN..."):
    clf = train_knn(X_train, y_train)

# 4. Calcul optimisé des prédictions de test (Mis en cache)
@st.cache_data
def evaluate_model(_model, X_te, y_te):
    preds = _model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    return preds, acc

preds, acc = evaluate_model(clf, X_test, y_test)

# Affichage des performances
st.write("### 📊 Performance du modèle")
st.success(f"Précision (Accuracy) globale sur l'ensemble de test : **{acc * 100:.2f}%**")

# 5. Interface interactive de test
st.write("### 🔍 Tester le modèle de manière interactive")
max_idx = len(X_test) - 1
idx = st.number_input(
    f"Choisissez un index d'image du dataset de test (0 à {max_idx}) :", 
    min_value=0, 
    max_value=max_idx, 
    value=0
)

image_to_test = X_test[idx]
true_label = y_test[idx]

label_text = "C'est un 0" if true_label == 1 else "Autre chiffre"

# Affichage de l'image MNIST correspondante
st.image(
    image_to_test.reshape(28, 28), 
    caption=f"Classe réelle : {label_text}", 
    width=150
)

# Prédiction en temps réel pour l'image sélectionnée
prediction = preds[idx]

st.write("---")
if prediction == 1:
    st.write("Prédiction du modèle : **C'est un 0 !**")
else:
    st.write("Prédiction du modèle : **Anomalie (Ce n'est pas un 0)**")
