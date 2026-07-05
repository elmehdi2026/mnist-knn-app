import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="TP KNN - Master IAENG", layout="wide")
st.title("TP KNN : Vote Local, Voisinage et Malédiction de la Dimension")
st.subheader("EL MEHDI - Master IAENG")

# --- 1. CHARGEMENT OPTIMISÉ DU DATASET ---
@st.cache_data
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    # Sélection des chiffres 0 et 1 pour la clarté du voisinage
    mask = (y == 0) | (y == 1)
    return X[mask], y[mask]

X, y = load_mnist()

# --- INTERFACE CORPS ---
col_ctrl, col_visu = st.columns([1, 2])

with col_ctrl:
    st.header("⚙️ Configuration du KNN")
    
    # 1. Sélection de l'image de test
    img_idx = st.slider("Sélectionner une image de test", 0, len(X) - 1, 10)
    image_test = X[img_idx]
    label_vrai = y[img_idx]
    
    st.markdown("---")
    st.write("### 🎛️ Hyperparamètres")
    
    # Choisir le nombre de voisins (Impairs pour éviter les égalités)
    K = st.slider("Nombre de voisins (K)", min_value=1, max_value=15, value=3, step=2)
    
    mode = st.radio("Type d'analyse :", 
                    ["Test 1 : Visualisation des K plus proches voisins", 
                     "Test 2 : Impact de la PCA (Vitesse vs Dimension)"])

# --- 2. LOGIQUE DU MODE 1 : RECHERCHE VISUELLE DES VOISINS ---
if "Test 1" in mode:
    # On applique une PCA légère à 20 dimensions pour accélérer et stabiliser la métrique
    pca = PCA(n_components=20)
    X_train_pca = pca.fit_transform(X)
    img_test_pca = pca.transform(image_test.reshape(1, -1))
    
    # Entraînement du KNN
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train_pca, y)
    
    # Trouver les indices et distances des K voisins les plus proches
    distances, indices = knn.kneighbors(img_test_pca)
    prediction = knn.predict(img_test_pca)[0]

    with col_visu:
        st.header("📊 Analyse du Voisinage Local")
        
        # Affichage de la cible
        col_target, col_pred = st.columns(2)
        with col_target:
            fig_t, ax_t = plt.subplots(figsize=(2, 2))
            ax_t.imshow(image_test.reshape(28, 28), cmap='gray')
            ax_t.set_title(f"Image Test (Vrai: {label_vrai})")
            ax_t.axis('off')
            st.pyplot(fig_t)
            
        with col_pred:
            if prediction == label_vrai:
                st.success(f"🎯 Prédiction KNN : **{prediction}** (Correct !)")
            else:
                st.error(f"❌ Prédiction KNN : **{prediction}** (Erreur)")
            st.info(f"Le modèle a analysé les {K} images les plus proches dans l'espace mathématique pour voter.")

        st.markdown("---")
        st.write(f"### 👥 Les {K} plus proches voisins trouvés dans le dataset :")
        
        # Affichage des K voisins côte à côte
        fig_v, axes = plt.subplots(1, K, figsize=(3 * K, 3))
        if K == 1:
            axes = [axes] # Rendre itérable si K=1
            
        for i, idx_voisin in enumerate(indices[0]):
            axes[i].imshow(X[idx_voisin].reshape(28, 28), cmap='plasma')
            axes[i].set_title(f"Voisin {i+1}\nClasse: {y[idx_voisin]}\nDist: {distances[0][i]:.2f}", fontsize=10)
            axes[i].axis('off')
            
        st.pyplot(fig_v)

# --- 3. LOGIQUE DU MODE 2 : MALÉDICTION DE LA DIMENSIONNALITÉ ---
else:
    with col_visu:
        st.header("⚡ Benchmarking : Espace Brut (784D) vs Espace PCA (20D)")
        
        # Sous-échantillonnage pour éviter que le Streamlit Cloud ne crash à cause de la lenteur du 784D
        X_sub, y_sub = X[:2000], y[:2000]
        
        # --- CAS 1 : SANS PCA (784 Dimensions) ---
        start_time = time.time()
        knn_brut = KNeighborsClassifier(n_neighbors=K)
        knn_brut.fit(X_sub, y_sub)
        acc_brut = knn_brut.score(X_sub, y_sub)
        time_brut = time.time() - start_time
        
        # --- CAS 2 : AVEC PCA (20 Dimensions) ---
        start_time = time.time()
        pca_bench = PCA(n_components=20)
        X_sub_pca = pca_bench.fit_transform(X_sub)
        knn_pca = KNeighborsClassifier(n_neighbors=K)
        knn_pca.fit(X_sub_pca, y_sub)
        acc_pca = knn_pca.score(X_sub_pca, y_sub)
        time_pca = time.time() - start_time
        
        # Affichage du tableau comparatif
        st.write("### 📋 Tableau de comparaison des performances")
        st.table({
            "Métrique": ["Nombre de Dimensions", "Précision (Accuracy)", "Temps d'exécution (s)"],
            "KNN Brut (Sans PCA)": ["784", f"{acc_brut*100:.2f}%", f"{time_brut:.4f} s"],
            "KNN + PCA (Optimisé)": ["20", f"{acc_pca*100:.2f}%", f"{time_pca:.4f} s"]
        })
        
        st.success("💡 **Conclusion de l'expert :** La PCA réduit drastiquement le temps de calcul tout en conservant (voire améliorant) la précision, car elle supprime le bruit de fond qui fausse la distance euclidienne en haute dimension.")
