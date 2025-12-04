import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud
import imagesize
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# CONFIG
# -----------------------------------
IMG_DIR = "./Flipkart/Images/"
EMB_PATH = "./clip_embeddings.npy"
LABELS_PATH = "./clip_labels.npy"
DESCRIPTION_PATH = "./produits_clip.csv"

# -----------------------------------
# CHARGEMENT DES DONNÉES
# -----------------------------------
@st.cache_resource
def load_data():
    embeddings = np.load(EMB_PATH)
    labels = np.load(LABELS_PATH)
    description = pd.read_csv(DESCRIPTION_PATH)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings, labels, description

embeddings, labels, description = load_data()

# -----------------------------------
# FONCTIONS UTILES
# -----------------------------------
def cv2_read_rgb(path):
    img = cv2.imread(path)
    if img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess_uploaded_file(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def encode_image_clip(image_rgb, model, processor, device):
    import torch  # <-- IMPORTANT, sinon torch est importé globalement
    dummy_text = [""]
    inputs = processor(text=dummy_text, images=[image_rgb], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    img_emb = outputs.image_embeds.cpu().numpy()
    img_emb /= np.linalg.norm(img_emb, axis=-1, keepdims=True)
    return img_emb[0]

# -----------------------------------
# UI
# -----------------------------------
st.set_page_config(page_title="POC modèle CLIP", layout="wide")
tabs = st.tabs(["Analyse exploratoire", "Prédiction CLIP"])
tab_eda, tab_search = tabs

# ---------------------------
# Page 1 : Analyse exploratoire des données
# ---------------------------
with tab_eda:
    # Sidebar: about + shared category filter
    categories_list = sorted(description['category'].fillna('N/A').unique())
    filter_options = ["Toutes les catégories"] + categories_list

    st.sidebar.header("À propos")
    st.sidebar.info("""
    Cette application permet restituer les résultats d'un POC 
    sur la multi-modalité pour classifier automatiquement des produits de consommation.
    - En première page nous retrouvons une analyse exploratoire des données textuelles et visuelles.
    - En seconde page, une interface de classification des produits via le modèle CLIP est disponible afin de tester celui-ci.
    """)

    st.sidebar.markdown("---")
    selected = st.sidebar.selectbox("Filtrer par catégorie", filter_options, index=0)

    st.title("📊 Analyse exploratoire des données")
    st.markdown("Nous retrouvons ici les statistiques de base de notre jeu de données ainsi que quelques exemples de textes et d'images associées.")
    st.markdown("---")

    # Statistiques de base
    total = len(description)
    
    # Top row: TL = general stats, TR = histogram categories (or subcategories if filtered)
    top_left, top_right = st.columns([2, 4])

    with top_left:
        st.subheader("Statistiques générales")
        st.write(f"- Nombre d'images et descriptions référencées : **{total}**")

        # Liste des catégories
        categories = sorted(description["category"].unique())

        # Afficher chaque catégorie sur une nouvelle ligne pour une meilleure lisibilité
        categories_md = "\n".join(f"    * {c}" for c in categories)
        st.markdown("- Liste des catégories de produits :\n\n" + categories_md)

    # Ensure n_words exists
    if 'n_words' not in description.columns:
        description['n_words'] = description['product_description'].fillna('').astype(str).map(lambda s: len(s.split()))

    # Prepare filtered dataframe once
    if selected == "Toutes les catégories":
        df_filtered = description.copy()
    else:
        df_filtered = description[description['category'] == selected].copy()

    with top_right:
        if selected == "Toutes les catégories":
            cat_counts = description['category'].fillna('N/A').value_counts().reset_index()
            cat_counts.columns = ['category', 'count']
            cat_counts = cat_counts.sort_values('count', ascending=False)
            fig = px.bar(cat_counts, x='category', y='count',
                         labels={'category': 'Catégorie', 'count': 'Nombre de produits'},
                         title='Nombre de produits par catégorie')
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            sub_counts = df_filtered['subcategory'].fillna('N/A').value_counts().reset_index()
            sub_counts.columns = ['subcategory', 'count']
            sub_counts = sub_counts.sort_values('count', ascending=False)
            fig = px.bar(sub_counts, x='subcategory', y='count',
                         labels={'subcategory': 'Sous-catégorie', 'count': 'Nombre de produits'},
                         title=f'Nombre de produits pour la catégorie : {selected}')
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # Deuxième partie sur les statistiques textuelles
    st.subheader("Descriptions des produits : analyses textuelles")

    # Bottom row: BL = boxplot n_words (3/5), BR = wordcloud (2/5)
    bot_left, bot_right = st.columns([3, 2])

    with bot_left:
        if selected == "Toutes les catégories":
            df_box = description.copy()
            df_box['category'] = df_box['category'].fillna('N/A').astype(str)
            box_fig = px.box(df_box, x='n_words', y='category', points='outliers',
                             labels={'category': 'Catégorie', 'n_words': 'Nombre de mots'},
                             title="Dispersion du nombre de mots par catégorie")
            box_fig.update_layout(margin=dict(l=20, r=10, t=50, b=120), height=450)
            box_fig.update_xaxes(tickangle=-45)
            st.plotly_chart(box_fig, use_container_width=True)
        else:
            df_box = df_filtered.copy()
            df_box['subcategory'] = df_box['subcategory'].fillna('N/A').astype(str)
            box_fig = px.box(df_box, x='n_words', y='subcategory', points='outliers',
                             labels={'subcategory': 'Sous-catégorie', 'n_words': 'Nombre de mots'},
                             title=f"Dispersion du nombre de mots — {selected}")
            box_fig.update_layout(margin=dict(l=20, r=10, t=50, b=120), height=450)
            box_fig.update_xaxes(tickangle=-45)
            st.plotly_chart(box_fig, use_container_width=True)

    with bot_right:
        text_corpus = ' '.join(df_filtered['product_description'].dropna().astype(str).tolist())
        if len(text_corpus.strip()) == 0:
            st.info('Pas de texte disponible pour générer un wordcloud.')
        else:
            try:
                wc = WordCloud(width=450, height=300, background_color='white', collocations=False).generate(text_corpus)
                img = wc.to_image()
                st.image(img, use_container_width=True)
            except Exception as e:
                # fallback: show top words as bar chart
                tokens = [w.lower() for w in text_corpus.split() if len(w) > 2]
                common = Counter(tokens).most_common(20)
                if len(common) == 0:
                    st.info('Aucun mot fréquent à afficher.')
                else:
                    df_wc = pd.DataFrame(common, columns=['word', 'count'])
                    fig_wc = px.bar(df_wc, x='word', y='count', title='Mots fréquents (fallback)'),
                    st.plotly_chart(fig_wc, use_container_width=True)

    st.markdown("---")

    # Description des images
    st.subheader("Description des images : dimensions")

    # Récupérer les dimensions en utilisant la colonne `image` présente dans `description`
    liste_largeur = []
    liste_hauteur = []
    images_col = description.get('image') if 'image' in description.columns else None
    if images_col is None:
        st.error("La colonne 'image' est absente de `produits_clip.csv`. Impossible de récupérer les dimensions.")
        images_list = []
    else:
        images_list = images_col.fillna('').astype(str).tolist()

    for img_name in images_list:
        chemin_complet = os.path.join(IMG_DIR, img_name)
        try:
            w, h = imagesize.get(chemin_complet)
        except Exception:
            w, h = (np.nan, np.nan)
        liste_largeur.append(w)
        liste_hauteur.append(h)

    # Construire un dataframe aligné avec `description` (inclut subcategory si présent)
    dimensions_category = pd.DataFrame({
        'category': description['category'].fillna('N/A').astype(str).values,
        'subcategory': description['subcategory'].fillna('N/A').astype(str).values if 'subcategory' in description.columns else ['N/A'] * len(liste_largeur),
        'Largeur': liste_largeur,
        'Hauteur': liste_hauteur
    })

    # Convertir en numérique et nettoyer
    dimensions_category['Largeur'] = pd.to_numeric(dimensions_category['Largeur'], errors='coerce')
    dimensions_category['Hauteur'] = pd.to_numeric(dimensions_category['Hauteur'], errors='coerce')

    # Appliquer le filtre de la sidebar pour rendre les graphiques dynamiques
    if selected == "Toutes les catégories":
        dim_plot_df = dimensions_category.copy()
    else:
        dim_plot_df = dimensions_category[dimensions_category['category'] == selected].copy()

    # Deux colonnes identiques pour la dispersion largeur / hauteur
    img_left, img_right = st.columns(2)

    # Largeur
    df_w = dim_plot_df.dropna(subset=['Largeur'])
    with img_left:
        if df_w.empty:
            st.info('Aucune largeur disponible pour les images (vérifiez IMG_DIR et les noms de fichiers).')
            # Diagnostic: montrer les fichiers présents dans IMG_DIR et les résultats de imagesize.get
            try:
                files = sorted(os.listdir(IMG_DIR))
            except Exception:
                files = []
            sample_files = files[:50]
            diag_rows = []
            for fname in sample_files:
                path = os.path.join(IMG_DIR, fname)
                exists = os.path.exists(path)
                w = h = None
                try:
                    w, h = imagesize.get(path)
                except Exception:
                    w = h = None
                diag_rows.append({'file': fname, 'path': path, 'exists': exists, 'width': w, 'height': h})
            if len(diag_rows) > 0:
                st.markdown('**Diagnostic — fichiers dans IMG_DIR (extrait)**')
                st.dataframe(pd.DataFrame(diag_rows))
            else:
                st.write('Aucun fichier trouvé dans IMG_DIR.')
            # Diagnostic: vérifier les chemins construits depuis labels (extrait)
            lab_rows = []
            for i in range(min(50, len(labels))):
                lab_name = str(labels[i])
                path = os.path.join(IMG_DIR, lab_name)
                exists = os.path.exists(path)
                w = h = None
                try:
                    w, h = imagesize.get(path)
                except Exception:
                    w = h = None
                lab_rows.append({'label_index': i, 'label': lab_name, 'path': path, 'exists': exists, 'width': w, 'height': h})
            st.markdown('**Diagnostic — chemins construits depuis `labels` (extrait)**')
            st.dataframe(pd.DataFrame(lab_rows))
        else:
            if selected == "Toutes les catégories":
                y_field = 'category'
                title = 'Dispersion des largeurs par catégorie'
            else:
                y_field = 'subcategory' if 'subcategory' in dim_plot_df.columns else 'category'
                title = f'Dispersion des largeurs — {selected}'
            label_map = {'category': 'Catégorie', 'subcategory': 'Sous-catégorie'}
            fig_w = px.box(df_w, x='Largeur', y=y_field, points='outliers',
                           labels={y_field: label_map.get(y_field, y_field), 'Largeur': 'Largeur (px)'},
                           title=title)
            fig_w.update_layout(margin=dict(l=20, r=10, t=50, b=120), height=450)
            fig_w.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_w, use_container_width=True)

    # Hauteur
    df_h = dim_plot_df.dropna(subset=['Hauteur'])
    with img_right:
        if df_h.empty:
            st.info('Aucune hauteur disponible pour les images (vérifiez IMG_DIR et les noms de fichiers).')
        else:
            if selected == "Toutes les catégories":
                y_field = 'category'
                title = 'Dispersion des hauteurs par catégorie'
            else:
                y_field = 'subcategory' if 'subcategory' in dim_plot_df.columns else 'category'
                title = f'Dispersion des hauteurs — {selected}'
            label_map = {'category': 'Catégorie', 'subcategory': 'Sous-catégorie'}
            fig_h = px.box(df_h, x='Hauteur', y=y_field, points='outliers',
                           labels={y_field: label_map.get(y_field, y_field), 'Hauteur': 'Hauteur (px)'},
                           title=title)
            fig_h.update_layout(margin=dict(l=20, r=10, t=50, b=120), height=450)
            fig_h.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("---")
    
    # Exemples d'images et de textes (filtrés par la sidebar)
    st.subheader("Exemples d'images et de descriptions")
    df_examples = df_filtered if 'df_filtered' in locals() else description.copy()
    # limiter le nombre d'exemples affichés
    n_examples = min(4, len(df_examples))
    if n_examples == 0:
        st.info('Aucun item à afficher pour le filtre sélectionné.')
    else:
        cols = st.columns(4)
        for idx in range(n_examples):
            row = df_examples.iloc[idx]
            img_name = row.get('image', '')
            img_path = os.path.join(IMG_DIR, img_name) if img_name else None
            img_cv = cv2_read_rgb(img_path) if img_path else None
            with cols[idx % 4]:
                if img_cv is not None:
                    st.image(img_cv, caption=str(img_name), width=180)
                else:
                    st.write(f'Image introuvable: {img_name}')
                desc = row.get('product_description', '')
                if isinstance(desc, str) and len(desc) > 0:
                    snippet = desc.replace('\n', ' ')[:250]
                    st.caption(snippet + ('...' if len(desc) > 250 else ''))

    st.markdown("---")

# ---------------------------
# Page 2 : Prédiction CLIP
# ---------------------------
# -----------------------------------
# CLIP (chargé uniquement à la demande)
# -----------------------------------
@st.cache_resource
def load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    MODEL_DIR = "models/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(
        MODEL_DIR,
        local_files_only=True   # ← indispensable sur Streamlit Cloud !
    ).to(device)

    processor = CLIPProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    return model, processor, device

# -----------------------------------
# CLASSIFIEUR RF (caché correctement)
# -----------------------------------
@st.cache_resource
def load_classifier(embeddings, labels_or_categories):
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels_or_categories)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(embeddings, y_encoded)
    return clf, le

with tab_search:

    st.title("🔎 Recherche et Prédiction d’Images avec CLIP")
    st.markdown("Testez le modèle CLIP en choisissant une image du dataset.")

    try:
        current_category = selected
    except NameError:
        current_category = "Toutes les catégories"

    dataset_df = description if current_category == "Toutes les catégories" else description[description["category"] == current_category]
    dataset_images = dataset_df.get('image', pd.Series([])).fillna('').astype(str).tolist()

    selected_dataset_image = st.selectbox("Choisir une image du dataset", [""] + dataset_images)

    if selected_dataset_image:

        sel_path = os.path.join(IMG_DIR, selected_dataset_image)
        sel_img = cv2_read_rgb(sel_path)

        # Index dans le CSV
        matches = description.index[description['image'].fillna('').astype(str) == selected_dataset_image].tolist()
        idx = matches[0] if matches else None

        if sel_img is not None:
            st.image(sel_img, caption=selected_dataset_image, width=300)
            # Afficher la vraie catégorie / sous-catégorie associée si disponible
            if idx is not None:
                true_cat = description.at[idx, 'category'] if 'category' in description.columns else None
                true_sub = description.at[idx, 'subcategory'] if 'subcategory' in description.columns else None
                info_text = ""
                if true_cat is not None:
                    info_text += f"Vraie catégorie : {true_cat}"
                if info_text:
                    st.info(info_text)

            # Chargement CLIP uniquement maintenant
            model, processor, device = load_clip()

            

            # Chargement classifieur
            clf, le = load_classifier(embeddings, labels)

            if st.button("Prédire la catégorie"):
                if idx is None:
                    st.warning("Impossible de retrouver l’entrée dans produits_clip.csv.")
                else:
                    emb_vec = embeddings[idx].reshape(1, -1)
                    pred_enc = clf.predict(emb_vec)[0]
                    proba = clf.predict_proba(emb_vec).max()
                    pred_label = le.inverse_transform([pred_enc])[0]
                    st.success(f"🎯 Prédiction : {pred_label} — Confiance : {proba:.2%}")

        else:
            st.warning("Image introuvable dans IMG_DIR.")
