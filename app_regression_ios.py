"""
Atelier de régression — interface style iOS (CustomTkinter).

Fonctionnalités (identiques à la version Tkinter, habillage modernisé) :
  1. Charger un CSV.
  2. Cocher les colonnes d'entrée X et de sortie y.
  3. Aperçu des 10 premières lignes.
  4. Menu déroulant à 4 modèles + évaluation en validation croisée (R²).

Dépendances : customtkinter, pandas, scikit-learn, xgboost, optuna
Installation : pip install customtkinter pandas scikit-learn xgboost optuna
Lancement    : python app_regression_ios.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import customtkinter as ctk
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline


# --- Palette iOS (clair, sombre) -------------------------------------------
BG       = ("#F2F2F7", "#000000")   # fond groupé iOS
CARD     = ("#FFFFFF", "#1C1C1E")   # cartes
TXT      = ("#1C1C1E", "#FFFFFF")   # texte principal
SUBTXT   = ("#8E8E93", "#8E8E93")   # texte secondaire (gris iOS)
BLUE     = ("#007AFF", "#0A84FF")   # bleu système iOS
BLUE_HOV = ("#0063CC", "#0974E0")
GREEN    = ("#34C759", "#30D158")   # vert système iOS

RADIUS = 16


# === MODÈLES DE CALCUL ======================================================
# C'est ICI que sont définis les modèles de régression proposés à l'utilisateur.
# Les 3 premiers sont des pipelines scikit-learn : standardisation (StandardScaler)
# puis l'algorithme de régression proprement dit. Le 4e (XGB_LABEL) ne vient pas
# de ce dictionnaire : il est traité à part dans train_model() car il s'appuie
# sur optimize_xgboost() (optimize_xgboost.py), qui fait son propre split
# train/test, sa validation croisée et son réglage fin Optuna.
XGB_LABEL = "Gradient Boosting (XGBoost auto-optimisé)"


def build_models():
    """Construit le dictionnaire des fabriques de modèles scikit-learn proposés dans le menu déroulant.

    Utilité :
        Centralise la définition des 3 modèles scikit-learn de l'application (le 4e,
        XGBoost, est traité à part, voir XGB_LABEL). Chaque modèle est une pipeline
        StandardScaler + algorithme de régression. Les imports scikit-learn sont faits
        à l'intérieur de la fonction (imports paresseux) pour que la fenêtre puisse
        s'ouvrir même si scikit-learn n'est pas installé.

    Entrée :
        Aucune.

    Sortie :
        dict[str, Callable[[], sklearn.pipeline.Pipeline]] : associe le nom affiché du
        modèle (str, utilisé aussi comme clé dans MODEL_NAMES et MODEL_EXPLANATIONS) à
        une fonction sans argument qui, une fois appelée, renvoie une nouvelle instance
        non entraînée de la pipeline scikit-learn correspondante.
    """
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GridSearchCV

    def build_optimized_rf():
        # Le Random Forest n'a mathématiquement pas besoin de StandardScaler.
        # On instancie la base du modèle.
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # GRILLE ALLÉGÉE : L'appli évalue déjà le modèle avec une validation croisée externe.
        # Si on met 36 combinaisons ici, l'interface va figer trop longtemps (nested CV).
        # On réduit les choix pour rester réactif (2x2x3 = 12 combinaisons).
        param_grid = {
            'n_estimators': [ 200,600,800],      
            'max_depth': [8, 12, 16],             
            'min_samples_leaf': [2, 3, 5]
        }
        
        # On enveloppe le Random Forest dans un GridSearchCV avec cv=3 (plus rapide)
        # n_jobs=-1 est vital pour que tous les cœurs du CPU travaillent en même temps
        return GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, n_jobs=-1)
    
    def build_optimized_poly():

        pipeline = Pipeline([("scaler", StandardScaler()), ('poly_features', PolynomialFeatures(include_bias=False)), ('ridge', Ridge())])

        param_grid = {
            'poly_features__degree': [1,2,3,4,5],
            'ridge__alpha' : [0.001, 0.01, 0.1, 1.0]
        }

        grid_search = GridSearchCV(
        estimator = pipeline,
        param_grid = param_grid,
        cv = 5,
        scoring = 'neg_mean_squared_error',
        n_jobs = -1
        )
        return grid_search

    return {
        "Régression linéaire": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "Régression Polynomiale": build_optimized_poly,
        "Forêt aléatoire": build_optimized_rf,
    }

MODEL_NAMES = ["Régression linéaire", "Régression Polynomiale", "Forêt aléatoire", XGB_LABEL]


# Explications vulgarisées affichées dans la pop-up "Explication du modèle".
MODEL_EXPLANATIONS = {
    "Régression linéaire": {
        "fonctionnement": (
            "Trace la droite (ou le plan) qui passe \"au plus près\" de tous les "
            "points, en minimisant l'écart moyen entre les valeurs prédites et "
            "les valeurs réelles."
        ),
        "quand": (
            "Quand la relation entre X et y est à peu près linéaire, pour un "
            "premier modèle simple et rapide à interpréter. Fonctionne dès "
            "~20-30 lignes de données."
        ),
        "limites": (
            "Ne capture pas les relations non linéaires. Sensible aux valeurs "
            "aberrantes et aux variables d'entrée trop corrélées entre elles."
        ),
    },
    "Régression Polynomiale": {
        "fonctionnement": (
            "En réalité une régression linéaire \"régularisée\" (Ridge) : elle "
            "trace elle aussi une droite, mais pénalise les coefficients trop "
            "grands pour éviter que le modèle ne colle trop aux données "
            "d'entraînement."
        ),
        "quand": (
            "Quand on a plusieurs variables d'entrée potentiellement corrélées, "
            "ou peu de données par rapport au nombre de colonnes X. Idéalement "
            "dès ~30 lignes, et plus de lignes que de colonnes."
        ),
        "limites": (
            "Reste un modèle linéaire : ne capture pas les relations non "
            "linéaires. Moins directement interprétable qu'une régression "
            "linéaire simple."
        ),
    },
    "Forêt aléatoire": {
        "fonctionnement": (
            "Fait \"voter\" des centaines d'arbres de décision, chacun entraîné "
            "sur un échantillon légèrement différent des données, puis moyenne "
            "leurs prédictions."
        ),
        "quand": (
            "Quand la relation entre X et y est complexe ou non linéaire, et "
            "qu'on ne veut pas se soucier de normaliser les données. Prévoir "
            "au moins ~100 lignes pour un résultat fiable."
        ),
        "limites": (
            "Moins interprétable (boîte noire), plus lent qu'un modèle "
            "linéaire, et extrapole mal en dehors de la plage de valeurs vues "
            "à l'entraînement."
        ),
    },
    XGB_LABEL: {
        "fonctionnement": (
            "Construit une succession d'arbres de décision où chaque nouvel "
            "arbre corrige les erreurs des précédents (\"boosting\"), avec un "
            "réglage automatique des réglages internes (Optuna)."
        ),
        "quand": (
            "Quand on cherche la meilleure performance possible sur une "
            "relation complexe, et qu'on dispose d'un jeu de données de taille "
            "moyenne à grande (idéalement 200 lignes ou plus)."
        ),
        "limites": (
            "Le plus lent à entraîner (optimisation des hyperparamètres). "
            "Boîte noire, et risque de sur-apprentissage sur un petit jeu de "
            "données."
        ),
    },
}


# Explication vulgarisée de la case "Regrouper par essai", affichée dans la
# pop-up "Explication du modèle" quelle que soit le modèle sélectionné (ce
# réglage s'applique aux 4 modèles, pas à un modèle en particulier).
GROUP_EXPLANATION = {
    "fonctionnement": (
        "Plusieurs lignes du fichier peuvent provenir du même « essai » "
        "(la même expérience). Cochée, cette case interdit qu'un essai soit "
        "à la fois utilisé pour entraîner le modèle et pour l'évaluer : "
        "l'évaluation se fait sur un ou plusieurs essais que le modèle n'a "
        "JAMAIS vus pendant l'entraînement — comme s'il devait prédire un "
        "essai réalisé demain."
    ),
    "quand": (
        "Laisse-la cochée si tu veux savoir si le modèle sait généraliser à "
        "une nouvelle expérience : c'est la mesure la plus honnête. Ne la "
        "décoche que pour un test rapide, ou si ton usage réel consiste à "
        "compléter un essai déjà partiellement observé (dans ce cas précis, "
        "le modèle a le droit d'avoir déjà vu d'autres lignes du même essai)."
    ),
    "limites": (
        "Décochée : les lignes d'un même essai se ressemblant beaucoup entre "
        "elles, le modèle peut se retrouver à réviser avec les réponses de "
        "l'examen. Le score affiché est alors optimiste et ne dit rien de sa "
        "capacité à généraliser à une expérience réellement nouvelle.\n\n"
        "Cochée : si le fichier ne contient que peu d'essais distincts (par "
        "exemple 6), l'essai (ou les 2-3 essais) gardé de côté pour "
        "l'évaluation peut, par hasard, être assez différent des autres. Un "
        "score très mauvais (R² très négatif) ne signifie alors pas "
        "forcément que le modèle est inutilisable, mais qu'il n'a pas encore "
        "assez d'essais variés pour apprendre à généraliser correctement. "
        "Plus il y a d'essais distincts dans le fichier, plus cette mesure "
        "devient fiable."
    ),
}


def _grouped_eval_settings(n_groups_distinct):
    """Calcule les réglages d'évaluation (taille de test, nombre de plis) adaptés au nombre d'essais distincts.

    Utilité :
        Avec peu d'essais distincts, isoler un seul essai à la fois en test (ou en
        pli de validation croisée) rend le score très dépendant du cas particulier
        de cet essai. Cette fonction élargit alors la part réservée à
        l'évaluation (moins de plis pour la CV groupée, part de test plus grande
        pour un split unique), au prix de moins de données d'entraînement.
        Utilisée par les 4 modèles (les 3 pipelines scikit-learn ET XGBoost).

    Entrée :
        n_groups_distinct (int) : nombre d'essais distincts présents dans les
            données (ex. `len(np.unique(groups))`).

    Sortie :
        dict : avec deux clés :
          - "test_size" (float, entre 0 et 1) : fraction des données à réserver
            au test pour un split unique (utilisé par XGBoost).
          - "max_folds" (int) : nombre maximal de plis à utiliser pour une
            validation croisée groupée (GroupKFold, utilisé par les modèles
            scikit-learn).
    """
    if n_groups_distinct <= 10:
        return {"test_size": 0.4, "max_folds": 3}
    return {"test_size": 0.2, "max_folds": 5}


class _XGBBoosterPredictor:
    """Enveloppe un Booster XGBoost brut pour lui donner une interface `.predict(X)` identique aux estimateurs scikit-learn.

    Utilité :
        `optimize_xgboost()` renvoie un `xgboost.Booster` natif (pas un estimateur
        scikit-learn), qui attend un `xgb.DMatrix` et n'a pas de méthode `.predict`
        prenant directement un tableau. Cette classe adapte ce Booster pour qu'il
        puisse être stocké dans `self.trained_model` et exporté/réutilisé exactement
        comme les modèles scikit-learn des 3 autres options.
    """

    def __init__(self, booster):
        """Stocke le Booster XGBoost entraîné à envelopper.

        Utilité :
            Constructeur : conserve une référence au Booster pour les appels
            ultérieurs à `predict()`.

        Entrée :
            booster (xgboost.Booster) : modèle XGBoost déjà entraîné, tel que
                renvoyé par `optimize_xgboost()["model"]`.

        Sortie :
            None.
        """
        self.booster = booster

    def predict(self, X):
        """Prédit la sortie y pour de nouvelles observations X.

        Utilité :
            Reproduit l'interface `.predict(X)` des estimateurs scikit-learn, en
            convertissant X au format `DMatrix` attendu par XGBoost avant de
            déléguer la prédiction au Booster interne.

        Entrée :
            X (array-like de forme (n_échantillons, n_features), ex. numpy.ndarray
                ou pandas.DataFrame) : valeurs des colonnes d'entrée pour lesquelles
                prédire y, dans le même ordre de colonnes que lors de l'entraînement.

        Sortie :
            numpy.ndarray de forme (n_échantillons,) : valeurs prédites de y.
        """
        import xgboost as xgb
        return self.booster.predict(xgb.DMatrix(X))


class RegressionApp(ctk.CTk):
        
    def __init__(self):
        """Construit la fenêtre principale de l'application et initialise son état.

        Utilité :
            Configure le thème CustomTkinter, la fenêtre (titre, taille), initialise
            en mémoire les attributs qui stockent l'état de la session utilisateur
            (données chargées, colonnes cochées, modèle entraîné), puis construit
            tous les blocs de l'interface (en-tête, carte fichier, aperçu, sélections,
            actions).

        Entrée :
            Aucune.

        Sortie :
            None. Attributs d'instance notables créés :
              - self.df (pandas.DataFrame | None) : intégralité du CSV chargé par
                l'utilisateur, tel quel, en mémoire (RAM) uniquement. Rien n'est
                persisté sur disque ni envoyé ailleurs : les données ne vivent que le
                temps de la session et sont perdues à la fermeture de l'application
                ou au rechargement d'un nouveau fichier (voir load_csv()).
              - self.input_vars / self.output_vars (list[tuple[str, ctk.BooleanVar]]) :
                pour chaque colonne du CSV, association (nom de colonne, case à
                cocher) indiquant si elle est sélectionnée comme entrée X / sortie y.
              - self.trained_model (objet avec .predict(X) | None) : dernier modèle
                entraîné sur l'ensemble des données (exportable).
              - self.trained_model_meta (dict | None) : métadonnées associées
                (colonnes X/y, nom du modèle).
              - self.groups_col (str | None) : nom de la colonne de regroupement
                ('essai') détectée automatiquement dans le CSV.
              - self.group_by_essai_var (ctk.BooleanVar) : état de la case à cocher
                « Regrouper par essai », qui contrôle si le split train/test respecte
                ou non les essais.
        """
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.title("Atelier de régression")
        self.geometry("1060x760")
        self.minsize(900, 640)
        self.configure(fg_color=BG)

        self.df = None
        self.input_vars = []
        self.output_vars = []
        self.trained_model = None
        self.trained_model_meta = None
        self.groups_col = None
        self.group_by_essai_var = ctk.BooleanVar(value=True)

        self._build_header()
        self._build_file_card()
        self._build_preview_card()
        self._build_selection_cards()
        self._build_action_card()

        # mise en grille extensible
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=3)   # aperçu
        self.grid_rowconfigure(3, weight=2)   # sélections

    # ------------------------------------------------------------------ UI ---
    def _build_header(self):
        """Construit la barre d'en-tête : titre de l'application et bascule clair/sombre.

        Utilité :
            Affiche le titre "Atelier de régression" et un sélecteur segmenté
            ("Clair" / "Sombre") relié à `_toggle_mode()` pour changer de thème.

        Entrée :
            Aucune (utilise `self` pour rattacher les widgets à la fenêtre).

        Sortie :
            None. Crée l'attribut `self.mode_switch` (ctk.CTkSegmentedButton).
        """
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 8))
        head.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            head, text="Atelier de régression",
            font=ctk.CTkFont(size=26, weight="bold"), text_color=TXT,
        ).grid(row=0, column=0, sticky="w")

        self.mode_switch = ctk.CTkSegmentedButton(
            head, values=["Clair", "Sombre"], command=self._toggle_mode,
            selected_color=BLUE, selected_hover_color=BLUE_HOV,
        )
        self.mode_switch.set("Clair")
        self.mode_switch.grid(row=0, column=1, sticky="e")

    def _card(self, row, title=None, weight_row=None):
        """Crée une "carte" (bloc arrondi de style iOS) placée dans la grille principale, avec un titre optionnel.

        Utilité :
            Fabrique réutilisée par les différentes sections de l'interface
            (fichier, aperçu, actions) pour obtenir un style visuel cohérent.

        Entrée :
            row (int) : numéro de ligne de la grille principale (`self.grid`) où
                placer la carte.
            title (str | None) : si fourni, texte affiché en petit et en gras en
                haut de la carte (ex. "APERÇU — 10 PREMIÈRES LIGNES"). Aucun titre
                affiché si None (valeur par défaut).
            weight_row : non utilisé actuellement (paramètre réservé).

        Sortie :
            ctk.CTkFrame : le cadre de la carte créée, déjà placé dans la grille,
            dans lequel l'appelant peut ensuite ajouter ses propres widgets.
        """
        card = ctk.CTkFrame(self, fg_color=CARD, corner_radius=RADIUS)
        card.grid(row=row, column=0, sticky="nsew", padx=24, pady=8)
        if title:
            ctk.CTkLabel(
                card, text=title, font=ctk.CTkFont(size=13, weight="bold"),
                text_color=SUBTXT,
            ).pack(anchor="w", padx=18, pady=(14, 0))
        return card

    def _build_file_card(self):
        """Construit la carte contenant le bouton de chargement du CSV et le bouton d'explication du modèle.

        Utilité :
            Affiche le bouton "Charger un CSV" (relié à `load_csv()`), un libellé
            indiquant le fichier actuellement chargé, et le bouton "Explication du
            modèle" (relié à `show_model_explanation()`).

        Entrée :
            Aucune (utilise `self`).

        Sortie :
            None. Crée l'attribut `self.file_label` (ctk.CTkLabel).
        """
        card = self._card(row=1)
        bar = ctk.CTkFrame(card, fg_color="transparent")
        bar.pack(fill="x", padx=18, pady=16)

        ctk.CTkButton(
            bar, text="  Charger un CSV", command=self.load_csv,
            corner_radius=12, height=40, fg_color=BLUE, hover_color=BLUE_HOV,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(side="left")

        self.file_label = ctk.CTkLabel(
            bar, text="Aucun fichier chargé.", text_color=SUBTXT,
            font=ctk.CTkFont(size=13),
        )
        self.file_label.pack(side="left", padx=16)

        ctk.CTkButton(
            bar, text="Explication du modèle", command=self.show_model_explanation,
            corner_radius=12, height=40, fg_color=BLUE, hover_color=BLUE_HOV,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(side="right")

        # NOUVEAU : Le bouton ACP déplacé ici
        self.pca_button = ctk.CTkButton(
            bar, text="Analyse ACP", command=self.show_pca,
            corner_radius=12, height=40, fg_color=BLUE, hover_color=BLUE_HOV,
            font=ctk.CTkFont(size=14, weight="bold"), state="disabled",
        )
        self.pca_button.pack(side="right", padx=(0, 10))

    def _build_preview_card(self):
        """Construit la carte contenant le tableau d'aperçu des 10 premières lignes du CSV.

        Utilité :
            Met en place le `ttk.Treeview` (avec ses barres de défilement) qui
            affichera plus tard un extrait des données via `_populate_preview()`.

        Entrée :
            Aucune (utilise `self`).

        Sortie :
            None. Crée l'attribut `self.tree` (ttk.Treeview).
        """
        card = self._card(row=2, title="APERÇU — 10 PREMIÈRES LIGNES")
        wrap = ctk.CTkFrame(card, fg_color="transparent")
        wrap.pack(fill="both", expand=True, padx=14, pady=(8, 16))

        self.tree = ttk.Treeview(wrap, show="headings")
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(wrap, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)
        self._style_treeview()

    def _build_selection_cards(self):
        """Construit les deux cartes défilantes de sélection des colonnes d'entrée X et de sortie y.

        Utilité :
            Prépare les conteneurs (`self.input_frame`, `self.output_frame`) dans
            lesquels `_populate_checkboxes()` insérera dynamiquement une case à
            cocher par colonne du CSV chargé.

        Entrée :
            Aucune (utilise `self`).

        Sortie :
            None. Crée les attributs `self.input_frame` et `self.output_frame`
            (ctk.CTkScrollableFrame).
        """
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=3, column=0, sticky="nsew", padx=24, pady=8)
        row.grid_columnconfigure((0, 1), weight=1)
        row.grid_rowconfigure(0, weight=1)

        self.input_frame = ctk.CTkScrollableFrame(
            row, label_text="Entrées  X", corner_radius=RADIUS, fg_color=CARD,
            label_fg_color=CARD, label_text_color=SUBTXT,
        )
        self.input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.output_frame = ctk.CTkScrollableFrame(
            row, label_text="Sorties  y", corner_radius=RADIUS, fg_color=CARD,
            label_fg_color=CARD, label_text_color=SUBTXT,
        )
        self.output_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

    def _build_action_card(self):
        """Construit la carte d'action : choix du modèle, case "Regrouper par essai", boutons Entraîner/Exporter et libellé de résultat.

        Utilité :
            Rassemble tous les contrôles nécessaires pour lancer l'entraînement
            (`train_model()`) et exporter le modèle obtenu (`export_model()`), ainsi
            que le libellé où s'affichent les métriques de résultat (R², RMSE).

        Entrée :
            Aucune (utilise `self`).

        Sortie :
            None. Crée les attributs `self.model_var` (ctk.StringVar),
            `self.group_checkbox` (ctk.CTkCheckBox), `self.result_label`
            (ctk.CTkLabel) et `self.export_button` (ctk.CTkButton).
        """
        card = self._card(row=4)
        bar = ctk.CTkFrame(card, fg_color="transparent")
        bar.pack(fill="x", padx=18, pady=16)

        ctk.CTkLabel(
            bar, text="Modèle", text_color=TXT, font=ctk.CTkFont(size=14)
        ).pack(side="left")

        self.model_var = ctk.StringVar(value=MODEL_NAMES[0])
        ctk.CTkOptionMenu(
            bar, values=MODEL_NAMES, variable=self.model_var,
            corner_radius=12, height=40, width=220,
            fg_color=BLUE, button_color=BLUE, button_hover_color=BLUE_HOV,
            font=ctk.CTkFont(size=14),
        ).pack(side="left", padx=12)

        # Case à cocher : respecter ou non le regroupement par essai lors du split
        # train/test et de la validation croisée. Décochée, chaque ligne est traitée
        # indépendamment (plus de données de test, mais risque de fuite si plusieurs
        # lignes d'un même essai se retrouvent à la fois en train et en test).
        self.group_checkbox = ctk.CTkCheckBox(
            bar, text="Regrouper par essai", variable=self.group_by_essai_var,
            fg_color=BLUE, hover_color=BLUE_HOV, corner_radius=6,
            text_color=TXT, font=ctk.CTkFont(size=13), state="disabled",
        )
        self.group_checkbox.pack(side="left", padx=12)

        ctk.CTkButton(
            bar, text="Entraîner / Évaluer", command=self.train_model,
            corner_radius=12, height=40, fg_color=GREEN, hover_color=GREEN,
            text_color="#FFFFFF", font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(side="left", padx=4)

        self.result_label = ctk.CTkLabel(
            bar, text="", text_color=GREEN, font=ctk.CTkFont(size=15, weight="bold")
        )
        self.result_label.pack(side="left", padx=16)

        self.export_button = ctk.CTkButton(
            bar, text="Exporter le modèle", command=self.export_model,
            corner_radius=12, height=40, fg_color=BLUE, hover_color=BLUE_HOV,
            font=ctk.CTkFont(size=14, weight="bold"), state="disabled",
        )
        self.export_button.pack(side="right")

        # NOUVEAU : Bouton pour afficher les coefficients
        self.coef_button = ctk.CTkButton(
            bar, text="Voir les coefficients", command=self.show_coefficients,
            corner_radius=12, height=40, fg_color=BLUE, hover_color=BLUE_HOV,
            font=ctk.CTkFont(size=14, weight="bold"), state="disabled",
        )
        self.coef_button.pack(side="right", padx=(0, 10))


    # --------------------------------------------------------------- thèmes ---
    def _toggle_mode(self, value):
        """Bascule l'application entre thème clair et thème sombre.

        Utilité :
            Callback appelé automatiquement par le sélecteur segmenté "Clair"/"Sombre"
            de l'en-tête (voir `_build_header`) à chaque changement de sélection.
            Applique le nouveau thème CustomTkinter puis restyle le tableau d'aperçu
            (ttk.Treeview, non couvert par le thème CustomTkinter).

        Entrée :
            value (str) : valeur sélectionnée dans le sélecteur segmenté, soit
                "Clair", soit "Sombre".

        Sortie :
            None.
        """
        ctk.set_appearance_mode("dark" if value == "Sombre" else "light")
        self._style_treeview()

    def _style_treeview(self):
        """Applique les couleurs du thème courant (clair/sombre) au tableau d'aperçu ttk.Treeview.

        Utilité :
            Le widget `ttk.Treeview` n'est pas géré par CustomTkinter et ne change
            donc pas automatiquement d'apparence avec `ctk.set_appearance_mode()` ;
            cette méthode le restyle manuellement (fond, texte, en-têtes, sélection)
            pour qu'il reste cohérent avec le reste de l'interface.

        Entrée :
            Aucune (lit le thème courant via `ctk.get_appearance_mode()`).

        Sortie :
            None.
        """
        dark = ctk.get_appearance_mode() == "Dark"
        bg   = "#1C1C1E" if dark else "#FFFFFF"
        fg   = "#FFFFFF" if dark else "#1C1C1E"
        head = "#2C2C2E" if dark else "#F2F2F7"
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Treeview", background=bg, foreground=fg,
                        fieldbackground=bg, rowheight=30, borderwidth=0)
        style.configure("Treeview.Heading", background=head, foreground=fg,
                        relief="flat", font=("", 11, "bold"))
        style.map("Treeview", background=[("selected", "#007AFF")],
                  foreground=[("selected", "#FFFFFF")])

    # -------------------------------------------------------------- actions ---
    def load_csv(self):
        """Ouvre un sélecteur de fichier, charge le CSV choisi en mémoire et met à jour toute l'interface en conséquence.

        Utilité :
            Point d'entrée des données utilisateur dans l'application. Ouvre un
            sélecteur de fichier natif (aucun upload réseau, tout se passe en
            local), lit le CSV choisi dans un `DataFrame` pandas (stocké dans
            `self.df`, qui reste l'unique copie des données utilisées pour l'aperçu
            et l'entraînement), détecte automatiquement une éventuelle colonne de
            regroupement ('essai'), puis rafraîchit le libellé de fichier, les
          cases à cocher X/y, l'aperçu, l'état de la case "Regrouper par essai" et
            réinitialise le dernier modèle entraîné.

        Entrée :
            Aucune (l'utilisateur choisit le chemin via la boîte de dialogue native
            qui s'ouvre à l'appel). Le fichier sélectionné doit être un `.csv`
            lisible par `pandas.read_csv` (séparateur auto-détecté, encodage
            `utf-8-sig` pour tolérer un BOM ajouté par Excel).

        Sortie :
            None. Effets de bord principaux :
              - self.df (pandas.DataFrame) : données chargées.
              - self.groups_col (str | None) : nom de la colonne 'essai' détectée,
                le cas échéant.
              - self.file_label, self.tree, self.input_frame/self.output_frame,
                self.group_checkbox, self.result_label, self.export_button : widgets
                mis à jour pour refléter le nouveau fichier chargé.
              - self.trained_model / self.trained_model_meta : remis à None (tout
                modèle entraîné sur un fichier précédent devient invalide).
        """
        path = filedialog.askopenfilename(
            title="Choisir un fichier CSV",
            filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")],
        )
        if not path:
            return
        try:
            # encoding="utf-8-sig" : retire le BOM UTF-8 en tête de fichier s'il est
            # présent (Excel en ajoute un). Sans ça, la 1re colonne se retrouve
            # nommée "﻿essai" au lieu de "essai" : la détection de la colonne
            # de regroupement ci-dessous échoue silencieusement (self.groups_col
            # reste None), ce qui casse a la fois l'exclusion de "essai" des cases
            # a cocher et le split train/test par groupe (fuite de donnees -> R²
            # artificiellement gonflé).
            df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        except Exception as exc:
            messagebox.showerror("Erreur de lecture", f"Lecture impossible :\n{exc}")
            return

        self.df = df

        # Cette colonne n'est ni une entrée X ni une sortie y : elle est retirée
        # des cases à cocher et utilisée directement comme `groups` à l'entraînement.
        group_candidates = [c for c in df.columns if str(c).strip().lower() == "essai"]
        self.groups_col = group_candidates[0] if group_candidates else None

        label = f"{path.split('/')[-1]}  ·  {df.shape[0]} lignes × {df.shape[1]} colonnes"
        if self.groups_col:
            label += f"  ·  regroupement auto par « {self.groups_col} »"
        self.file_label.configure(text=label)

        # La case n'a de sens que si une colonne de regroupement a été détectée.
        if self.groups_col:
            self.group_checkbox.configure(state="normal")
            self.group_by_essai_var.set(True)
        else:
            self.group_checkbox.configure(state="disabled")
            self.group_by_essai_var.set(False)

        checkbox_cols = [c for c in df.columns if c != self.groups_col]
        self._populate_checkboxes(checkbox_cols)

        self._populate_preview(df)
        self.result_label.configure(text="")
        self.trained_model = None
        self.trained_model_meta = None
        self.export_button.configure(state="disabled")
        self.export_button.configure(state="disabled")
        self.coef_button.configure(state="disabled")

    def show_model_explanation(self):
        """Ouvre une pop-up expliquant le modèle actuellement sélectionné et la case "Regrouper par essai".

        Utilité :
            Callback du bouton "Explication du modèle". Affiche, pour le modèle
            choisi dans le menu déroulant, son fonctionnement, quand l'utiliser et
            ses limites (issus de `MODEL_EXPLANATIONS`), suivis de l'explication de
            la case "Regrouper par essai" (issue de `GROUP_EXPLANATION`), toujours
            affichée quel que soit le modèle car ce réglage s'applique aux 4
            modèles.

        Entrée :
            Aucune (lit le modèle sélectionné via `self.model_var`).

        Sortie :
            None. Ouvre une fenêtre `ctk.CTkToplevel` modale ; aucune valeur
            renvoyée.
        """
        model_name = self.model_var.get() if hasattr(self, "model_var") else ""
        if not model_name:
            messagebox.showwarning("Aucun modèle", "Sélectionne d'abord un modèle.")
            return

        popup = ctk.CTkToplevel(self)
        popup.title(f"Explication — {model_name}")
        popup.geometry("560x640")
        popup.configure(fg_color=BG)
        popup.transient(self)
        popup.grab_set()

        ctk.CTkLabel(
            popup, text=model_name, font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TXT, wraplength=500, justify="left",
        ).pack(anchor="w", padx=20, pady=(20, 12))

        # Zone déroulante : le contenu (explication du modèle + explication de
        # la case "Regrouper par essai") dépasse la hauteur fixe de la pop-up.
        scroll = ctk.CTkScrollableFrame(popup, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=4, pady=(0, 8))

        explanation = MODEL_EXPLANATIONS.get(model_name)
        sections = (
            [
                ("Fonctionnement du modèle", explanation["fonctionnement"]),
                ("Quand l'utiliser", explanation["quand"]),
                ("Limites du modèle", explanation["limites"]),
            ]
            if explanation
            else [("Explication", "Explication à venir.")]
        )
        # La case "Regrouper par essai" s'applique aux 4 modèles (ce n'est pas
        # un réglage propre à un modèle), donc son explication est toujours
        # ajoutée, quel que soit le modèle sélectionné dans le menu déroulant.
        sections += [
            ("Case « Regrouper par essai »", GROUP_EXPLANATION["fonctionnement"]),
            ("Quand la cocher / la décocher", GROUP_EXPLANATION["quand"]),
            ("Limites de ce réglage", GROUP_EXPLANATION["limites"]),
        ]

        for title, body in sections:
            ctk.CTkLabel(
                scroll, text=title, font=ctk.CTkFont(size=14, weight="bold"),
                text_color=BLUE, wraplength=500, justify="left",
            ).pack(anchor="w", padx=16, pady=(4, 2))
            ctk.CTkLabel(
                scroll, text=body, text_color=SUBTXT,
                font=ctk.CTkFont(size=13), wraplength=500, justify="left",
            ).pack(anchor="w", padx=16, pady=(0, 10))

        ctk.CTkButton(
            popup, text="Fermer", command=popup.destroy,
            corner_radius=12, height=36, fg_color=BLUE, hover_color=BLUE_HOV,
        ).pack(anchor="e", padx=20, pady=(0, 20))

    def _populate_checkboxes(self, columns):
        """Reconstruit les cases à cocher de sélection des colonnes X et y à partir des colonnes du CSV chargé.

        Utilité :
            Vide et reconstruit les deux listes de cases à cocher (entrées X,
            sorties y) après le chargement d'un nouveau fichier. Ces cases à cocher
            déterminent quelles colonnes du CSV seront effectivement envoyées au
            modèle lors de l'entraînement (`train_model()`).

        Entrée :
            columns (list[str] ou liste d'objets convertibles en str via `str()`) :
                noms des colonnes à proposer (typiquement toutes les colonnes du
                CSV sauf la colonne de regroupement 'essai', déjà exclue par
                l'appelant).

        Sortie :
            None. Réinitialise les attributs `self.input_vars` et
            `self.output_vars` (list[tuple[str, ctk.BooleanVar]]), un tuple par
            colonne, et recrée les widgets correspondants dans `self.input_frame`
            / `self.output_frame`.
        """
        for frame in (self.input_frame, self.output_frame):
            for child in frame.winfo_children():
                child.destroy()
        self.input_vars, self.output_vars = [], []

        for col in columns:
            v_in = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(
                self.input_frame, text=str(col), variable=v_in,
                fg_color=BLUE, hover_color=BLUE_HOV, corner_radius=6,
                text_color=TXT, font=ctk.CTkFont(size=13),
                command=self._refresh_preview_selection,
            ).pack(anchor="w", padx=14, pady=4)
            self.input_vars.append((col, v_in))

            v_out = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(
                self.output_frame, text=str(col), variable=v_out,
                fg_color=BLUE, hover_color=BLUE_HOV, corner_radius=6,
                text_color=TXT, font=ctk.CTkFont(size=13),
                command=self._refresh_preview_selection,
            ).pack(anchor="w", padx=14, pady=4)
            self.output_vars.append((col, v_out))

    def _refresh_preview_selection(self):
        """Callback des cases X/y : ne garde dans l'aperçu que les colonnes actuellement cochées.

        Utilité :
            Recalcule à chaque coche/décoche l'ensemble des colonnes sélectionnées
            (entrées X ∪ sorties y) et redessine `self.tree` avec uniquement ces
            colonnes, dans l'ordre du CSV d'origine. Si plus aucune case n'est
            cochée, l'aperçu redevient vide (aucune colonne cochée = rien à montrer).

        Entrée :
            Aucune (lit `self.df`, `self.input_vars` et `self.output_vars`).

        Sortie :
            None. Reconstruit `self.tree` via `_populate_preview()`.
        """
        if getattr(self, "df", None) is None:
            return
        selected = set(self._checked(self.input_vars)) | set(self._checked(self.output_vars))
        cols = [c for c in self.df.columns if c in selected]
        self._populate_preview(self.df[cols])

    def _populate_preview(self, df):
        """Affiche les 10 premières lignes de `df` dans le tableau d'aperçu (`self.tree`).

        Utilité :
            Rafraîchit le `ttk.Treeview` d'aperçu après le chargement d'un CSV.
            N'affiche que `df.head(10)` pour rester léger ; `self.df` en mémoire
            contient lui la totalité du jeu de données.

        Entrée :
            df (pandas.DataFrame) : jeu de données complet dont on affiche un
                extrait (typiquement `self.df`, tel que chargé par `load_csv()`).

        Sortie :
            None. Vide et reconstruit le contenu (colonnes et lignes) de
            `self.tree`.
        """
        self.tree.delete(*self.tree.get_children())
        cols = list(df.columns.astype(str))
        self.tree["columns"] = cols
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=130, anchor="center", stretch=False)
        for _, r in df.head(10).iterrows():
            self.tree.insert("", tk.END, values=list(r))

    def _checked(self, var_list):
        """Filtre une liste de (colonne, case à cocher) pour ne garder que les colonnes cochées.

        Utilité :
            Fonction utilitaire commune pour lire l'état de `self.input_vars` et
            `self.output_vars` et en extraire la sélection courante de l'utilisateur.

        Entrée :
            var_list (list[tuple[str, ctk.BooleanVar]]) : liste de paires
                (nom de colonne, case à cocher associée), typiquement
                `self.input_vars` ou `self.output_vars`.

        Sortie :
            list[str] : noms des colonnes dont la case est actuellement cochée,
            dans le même ordre que `var_list`.
        """
        return [col for col, v in var_list if v.get()]

    def train_model(self):
        """Valide la sélection X/y de l'utilisateur, prépare les données puis lance l'entraînement du modèle choisi.

        Utilité :
            Point d'entrée du pipeline de calcul, déclenché par le bouton
            "Entraîner / Évaluer". Vérifie qu'un CSV est chargé et qu'au moins une
            colonne X et une colonne y sont cochées (sans chevauchement), convertit
            les colonnes sélectionnées en numérique en supprimant les lignes
            incomplètes, aligne la colonne de regroupement 'essai' si la case
            "Regrouper par essai" est cochée, puis délègue l'entraînement à
            `_train_xgboost()` (pour XGB_LABEL) ou `_train_sklearn()` (pour les 3
            autres modèles).

        Entrée :
            Aucune. Lit l'état de l'interface :
              - self.df (pandas.DataFrame) : données chargées via `load_csv()`.
              - self.input_vars / self.output_vars : colonnes cochées comme X / y.
              - self.model_var (ctk.StringVar) : nom du modèle sélectionné (doit
                être l'une des valeurs de MODEL_NAMES).
              - self.groups_col / self.group_by_essai_var : colonne 'essai' et état
                de la case "Regrouper par essai".

        Sortie :
            None. En cas de sélection invalide ou de données inexploitables,
            affiche un message d'avertissement/erreur (`messagebox`) et s'arrête
            sans entraîner de modèle. Sinon, délègue à `_train_sklearn()` ou
            `_train_xgboost()`, qui mettent à jour `self.trained_model`,
            `self.trained_model_meta`, `self.result_label` et `self.export_button`.
        """
        if self.df is None:
            messagebox.showwarning("Aucune donnée", "Charge d'abord un fichier CSV.")
            return

        inputs = self._checked(self.input_vars)
        outputs = self._checked(self.output_vars)

        if not inputs:
            messagebox.showwarning("Sélection", "Coche au moins une entrée X.")
            return
        if not outputs:
            messagebox.showwarning("Sélection", "Coche au moins une sortie y.")
            return
        overlap = set(inputs) & set(outputs)
        if overlap:
            messagebox.showwarning(
                "Sélection", f"Colonne(s) à la fois X et y : {', '.join(overlap)}."
            )
            return

        model_name = self.model_var.get()

        # Extraction du sous-ensemble utile des données utilisateur (colonnes X + y
        # cochées), conversion en numérique et suppression des lignes incomplètes.
        data = self.df[inputs + outputs].apply(pd.to_numeric, errors="coerce").dropna()
        if data.empty:
            messagebox.showerror(
                "Données",
                "Aucune ligne exploitable (colonnes non numériques ou valeurs manquantes).",
            )
            return

        # Colonne de regroupement ('essai') alignée sur les lignes conservées :
        # on retire aussi les lignes où elle est manquante, sinon un groupe NaN
        # se retrouverait mélangé entre train et test. Ignoré si la case
        # "Regrouper par essai" est décochée : chaque ligne est alors traitée
        # indépendamment (plus de données de test, mais un même essai peut se
        # retrouver à la fois en train et en test).
        groups = None
        if self.groups_col and self.group_by_essai_var.get():
            groups_series = self.df.loc[data.index, self.groups_col]
            valid = groups_series.notna()
            data = data[valid]
            groups_series = groups_series[valid]
            groups = groups_series.values

        X = data[inputs].values
        y = data[outputs].values

        n = X.shape[0]
        if n < 5:
            messagebox.showwarning("Échantillon", f"Trop peu d'observations ({n}).")
            return

        self.result_label.configure(text="Calcul…", text_color=SUBTXT)
        self.export_button.configure(state="disabled")
        self.export_button.configure(state="disabled")
        self.coef_button.configure(state="disabled")
        self.pca_button.configure(state="disabled")
        self.trained_model = None
        self.trained_model_meta = None
        self.update_idletasks()

        if model_name == XGB_LABEL:
            self._train_xgboost(inputs, outputs, X, y, n, groups)
        else:
            self._train_sklearn(model_name, inputs, outputs, X, y, n, groups)

    def _train_sklearn(self, model_name, inputs, outputs, X, y, n, groups):
        """Entraîne et évalue par validation croisée l'un des 3 modèles scikit-learn (linéaire, Ridge, forêt aléatoire).

        Utilité :
            Appelée par `train_model()` pour tout modèle sauf XGB_LABEL. Instancie
            le modèle choisi via `build_models()` (enveloppé dans un
            `MultiOutputRegressor` si plusieurs colonnes y sont sélectionnées),
            l'évalue par validation croisée en k plis (GroupKFold si une colonne de
            regroupement 'essai' est active, sinon KFold classique), puis le
            ré-entraîne une dernière fois sur l'intégralité des données pour obtenir
            le modèle final exportable.

        Entrée :
            model_name (str) : nom du modèle à entraîner, doit être une clé du
                dictionnaire renvoyé par `build_models()` (ex. "Régression linéaire").
            inputs (list[str]) : noms des colonnes X sélectionnées.
            outputs (list[str]) : noms des colonnes y sélectionnées.
            X (numpy.ndarray de forme (n, n_features)) : valeurs numériques des
                colonnes d'entrée, sans valeurs manquantes.
            y (numpy.ndarray de forme (n,) ou (n, n_outputs)) : valeurs numériques
                des colonnes de sortie, sans valeurs manquantes.
            n (int) : nombre d'observations (lignes) dans X/y, utilisé pour borner
                le nombre de plis de validation croisée.
            groups (numpy.ndarray de forme (n,) | None) : identifiant d'essai pour
                chaque ligne, ou None si le regroupement est désactivé/absent.

        Sortie :
            None. En cas d'erreur (dépendance manquante ou échec d'entraînement),
            affiche un message d'erreur et laisse `self.trained_model` à None.
            En cas de succès :
              - self.result_label : mis à jour avec le R² et le RMSE moyens (± écart-type).
              - self.trained_model : estimateur scikit-learn entraîné sur toutes les données.
              - self.trained_model_meta (dict) : {"model_name", "inputs", "outputs"}.
              - self.export_button : activé.
        """
        try:
            from sklearn.model_selection import cross_validate, KFold, GroupKFold
            from sklearn.multioutput import MultiOutputRegressor
        except ImportError:
            messagebox.showerror(
                "Dépendance manquante",
                "scikit-learn n'est pas installé.\n\npip install scikit-learn",
            )
            return

        if y.shape[1] == 1:
            y = y.ravel()

        # Instanciation du modèle choisi dans le menu déroulant (dictionnaire
        # défini dans build_models() en haut du fichier). Si plusieurs colonnes
        # de sortie y sont cochées, on enveloppe le modèle dans un
        # MultiOutputRegressor pour gérer la régression multi-sorties.
        estimator = build_models()[model_name]()
        if getattr(y, "ndim", 1) == 2 and y.shape[1] > 1:
            estimator = MultiOutputRegressor(estimator)

        try:
            # Validation croisée en k plis (k=5 max, ou moins si peu de données) :
            # le modèle est ré-entraîné et évalué k fois sur des découpages
            # différents des données, puis les scores R² sont moyennés. Ceci ne
            # sert qu'à MESURER la qualité du modèle (chaque modèle entraîné
            # pendant la CV est jeté ensuite). Si une colonne de regroupement est
            # détectée ('essai'), les plis respectent les groupes (GroupKFold) pour
            # qu'un même essai ne se retrouve jamais scindé entre deux plis. Avec
            # peu d'essais distincts, on réduit aussi le nombre de plis (voir
            # _grouped_eval_settings) pour que chaque pli de test en couvre
            # plusieurs plutôt qu'un seul, moins sensible au cas particulier
            # d'un essai isolé.
            scoring = ["r2", "neg_root_mean_squared_error"]
            if groups is not None:
                max_folds = _grouped_eval_settings(len(np.unique(groups)))["max_folds"]
                k = min(max_folds, n, len(np.unique(groups)))
                cv = GroupKFold(n_splits=k)
                cv_results = cross_validate(estimator, X, y, cv=cv, groups=groups, scoring=scoring)
            else:
                k = min(5, n)
                cv = KFold(n_splits=k, shuffle=True, random_state=0)
                cv_results = cross_validate(estimator, X, y, cv=cv, scoring=scoring)
            r2_scores = cv_results["test_r2"]
            rmse_scores = -cv_results["test_neg_root_mean_squared_error"]

            # Modèle final exportable : ré-entraîné une dernière fois sur
            # l'intégralité des données (X, y), pour tirer parti de toutes les
            # observations disponibles avant l'export.
            estimator.fit(X, y)
        except Exception as exc:  # noqa: BLE001
            self.result_label.configure(text="")
            messagebox.showerror("Erreur d'entraînement", str(exc))
            return

        self.result_label.configure(
            text=(
                f"R² (CV {k}-plis) = {r2_scores.mean():.3f} ± {r2_scores.std():.3f}  ·  "
                f"RMSE = {rmse_scores.mean():.3f} ± {rmse_scores.std():.3f}"
            ),
            text_color=GREEN,
        )

        self.trained_model = estimator
        self.trained_model_meta = {
            "model_name": model_name,
            "inputs": inputs,
            "outputs": outputs,
        }
        self.export_button.configure(state="normal")
        self.export_button.configure(state="normal")
        self.pca_button.configure(state="normal")
        if model_name in ["Régression linéaire", "Régression Polynomiale"]:
            self.coef_button.configure(state="normal")

    def _train_xgboost(self, inputs, outputs, X, y, n, groups):
        """Entraîne et évalue le modèle XGBoost auto-optimisé (XGB_LABEL) via `optimize_xgboost()`.

        Utilité :
            Appelée par `train_model()` uniquement pour XGB_LABEL. Ne gère qu'une
            seule colonne de sortie y à la fois. Délègue tout le travail (split
            train/test respectant les essais via GroupShuffleSplit, validation
            croisée interne pour le réglage des hyperparamètres, optimisation
            Optuna, calcul des métriques finales) à `optimize_xgboost()`
            (optimize_xgboost.py), avec un budget Optuna réduit (20 essais) pour
            rester réactif dans l'interface.

        Entrée :
            inputs (list[str]) : noms des colonnes X sélectionnées.
            outputs (list[str]) : noms des colonnes y sélectionnées ; doit
                contenir exactement une colonne (sinon un avertissement est
                affiché et l'entraînement est annulé).
            X (numpy.ndarray de forme (n, n_features)) : valeurs numériques des
                colonnes d'entrée, sans valeurs manquantes.
            y (numpy.ndarray de forme (n, 1)) : valeurs numériques de la colonne de
                sortie, sans valeurs manquantes (aplati en `y.ravel()` avant appel).
            n (int) : nombre d'observations (lignes) dans X/y, utilisé pour borner
                le nombre de plis de validation croisée interne (`nfold`).
            groups (numpy.ndarray de forme (n,) | None) : identifiant d'essai pour
                chaque ligne, transmis à `optimize_xgboost()` pour le split
                train/test ; None si le regroupement est désactivé/absent.

        Sortie :
            None. En cas d'erreur (dépendance manquante, sélection y invalide ou
            échec d'entraînement), affiche un message d'erreur/avertissement et
            laisse `self.trained_model` à None. En cas de succès :
              - self.result_label : mis à jour avec R² (test), RMSE (test) et le
                nombre d'arbres du modèle final.
              - self.trained_model (_XGBBoosterPredictor) : Booster XGBoost entraîné,
                enveloppé pour exposer une interface `.predict(X)` scikit-learn.
              - self.trained_model_meta (dict) : {"model_name", "inputs", "outputs",
                "best_params"} (hyperparamètres retenus par Optuna).
              - self.export_button : activé.
        """
        if y.shape[1] > 1:
            self.result_label.configure(text="")
            messagebox.showwarning(
                "Sélection",
                f"{XGB_LABEL} ne gère qu'une seule colonne de sortie y à la fois.\n"
                "Décoche les autres sorties ou choisis un autre modèle.",
            )
            return

        try:
            from optimize_xgboost import optimize_xgboost
        except ImportError:
            messagebox.showerror(
                "Dépendance manquante",
                "xgboost et optuna sont nécessaires.\n\npip install xgboost optuna",
            )
            return

        # `groups` (colonne 'essai'), si fourni, est transmis tel quel à
        # optimize_xgboost qui l'utilise UNIQUEMENT pour le split train/test
        # (GroupShuffleSplit) : un même essai ne se retrouve jamais scindé
        # entre train et test, pour une mesure finale honnête. La validation
        # croisée interne (réglage des hyperparamètres, Phases 1 à 5) ne
        # respecte plus les groupes et exploite directement les lignes du
        # train ; on réduit son nombre de plis seulement si peu de lignes sont
        # disponibles, pour éviter des plis trop petits.
        nfold = min(5, max(2, n // 10))

        # Avec un split par essai, une fraction de test fixe se traduit par un
        # nombre d'essais tenus à l'écart très grossier quand il y en a peu
        # (ex. 6 essais x 0.2 -> 1 seul essai en test : le R² final dépend
        # alors entièrement du cas particulier de cet essai). On élargit la
        # part de test quand peu d'essais sont distincts (même règle que pour
        # les 3 modèles scikit-learn, voir _grouped_eval_settings), pour que
        # le test set en couvre au moins 2-3 — au prix de moins d'essais
        # disponibles pour l'entraînement.
        test_size = _grouped_eval_settings(len(np.unique(groups)))["test_size"] if groups is not None else 0.2

        self.result_label.configure(
            text="Optimisation XGBoost en cours (réglage Optuna, peut prendre un peu de temps)…",
            text_color=SUBTXT,
        )
        self.update_idletasks()
        try:
            result = optimize_xgboost(
                X, y.ravel(),
                groups=groups,
                objective="reg:squarederror",
                eval_metric="rmse",
                n_trials=20,        # budget Optuna réduit pour rester réactif dans l'UI
                nfold=nfold,
                test_size=test_size,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            self.result_label.configure(text="")
            messagebox.showerror("Erreur d'entraînement", str(exc))
            return

        metrics = result["test_metrics"]
        self.result_label.configure(
            text=(
                f"R² (test) = {metrics['r2']:.3f}  ·  RMSE = {metrics['rmse']:.3f}  ·  "
                f"{result['n_rounds']} arbres"
            ),
            text_color=GREEN,
        )

        self.trained_model = _XGBBoosterPredictor(result["model"])
        self.trained_model_meta = {
            "model_name": XGB_LABEL,
            "inputs": inputs,
            "outputs": outputs,
            "best_params": result["best_params"],
        }
        self.export_button.configure(state="normal")
        self.pca_button.configure(state="normal")

    def export_model(self):
        """Sauvegarde sur disque le dernier modèle entraîné, avec ses métadonnées, au format joblib.

        Utilité :
            Callback du bouton "Exporter le modèle" (actif seulement après un
            entraînement réussi). Ouvre un sélecteur de fichier natif puis écrit un
            fichier `.joblib` contenant le modèle final (entraîné sur toutes les
            données) et les métadonnées nécessaires pour le réutiliser (colonnes
            X/y attendues, nom du modèle, et pour XGBoost les hyperparamètres
            retenus).

        Entrée :
            Aucune. L'utilisateur choisit le chemin de destination via la boîte de
            dialogue native qui s'ouvre à l'appel. Lit `self.trained_model` (objet
            avec méthode `.predict(X)`) et `self.trained_model_meta` (dict).

        Sortie :
            None. Écrit un fichier `.joblib` sur disque contenant un dict de la
            forme `{"model": self.trained_model, **self.trained_model_meta}`
            (chargeable ensuite avec `joblib.load(path)`). Affiche une boîte de
            dialogue de succès ou d'erreur selon le résultat.
        """
        if self.trained_model is None:
            messagebox.showwarning("Aucun modèle", "Entraîne d'abord un modèle.")
            return

        try:
            import joblib
        except ImportError:
            messagebox.showerror(
                "Dépendance manquante",
                "joblib n'est pas installé.\n\npip install joblib",
            )
            return

        path = filedialog.asksaveasfilename(
            title="Exporter le modèle",
            defaultextension=".joblib",
            filetypes=[("Modèle joblib", "*.joblib"), ("Tous les fichiers", "*.*")],
        )
        if not path:
            return

        try:
            joblib.dump(
                {"model": self.trained_model, **self.trained_model_meta}, path
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erreur d'export", str(exc))
            return

        messagebox.showinfo("Export réussi", f"Modèle enregistré :\n{path}")
    
    def show_coefficients(self):
        if not self.trained_model:
            return

        model_name = self.trained_model_meta["model_name"]
        
        # Fenêtre pop-up
        popup = ctk.CTkToplevel(self)
        popup.title(f"Coefficients — {model_name}")
        popup.geometry("480x500")
        popup.configure(fg_color=BG)
        popup.transient(self)
        popup.grab_set()

        ctk.CTkLabel(
            popup, text="Coefficients du modèle",
            font=ctk.CTkFont(size=20, weight="bold"), text_color=TXT
        ).pack(pady=(20, 5))

        # Précision importante car tu utilises un StandardScaler dans tes pipelines
        ctk.CTkLabel(
            popup, text="Note : Les valeurs s'appliquent aux variables d'entrée standardisées (centrées-réduites).",
            text_color=SUBTXT, font=ctk.CTkFont(size=12), wraplength=420
        ).pack(pady=(0, 10))

        scroll = ctk.CTkScrollableFrame(popup, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=20, pady=10)

        # Gestion multi-sorties (MultiOutputRegressor) ou sortie unique
        from sklearn.multioutput import MultiOutputRegressor
        if isinstance(self.trained_model, MultiOutputRegressor):
            estimators = self.trained_model.estimators_
        else:
            estimators = [self.trained_model]

        inputs = self.trained_model_meta["inputs"]
        outputs = self.trained_model_meta["outputs"]

        for i, est in enumerate(estimators):
            out_name = outputs[i]
            ctk.CTkLabel(
                scroll, text=f"Cible : {out_name}",
                font=ctk.CTkFont(size=15, weight="bold"), text_color=BLUE
            ).pack(anchor="w", pady=(15, 5))

            # Le modèle est un Pipeline scikit-learn (StandardScaler -> Regressor)
            # On récupère la dernière étape du pipeline (le régresseur)
            regressor = est[-1]
            coefs = regressor.coef_.ravel()
            
            # Formatage de l'ordonnée à l'origine (Intercept)
            intercept = regressor.intercept_
            if isinstance(intercept, np.ndarray):
                intercept = intercept[0]

            ctk.CTkLabel(
                scroll, text=f"Ordonnée à l'origine : {intercept:.4f}",
                font=ctk.CTkFont(weight="bold"), text_color=TXT
            ).pack(anchor="w", padx=10, pady=(0, 5))

            # Affichage des coefficients pour chaque colonne d'entrée
            for col_name, coef in zip(inputs, coefs):
                ctk.CTkLabel(
                    scroll, text=f"• {col_name} : {coef:.4f}",
                    text_color=TXT
                ).pack(anchor="w", padx=10)

        ctk.CTkButton(
            popup, text="Fermer", command=popup.destroy,
            corner_radius=12, height=40, fg_color=BLUE, hover_color=BLUE_HOV,
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=20)
    
    def show_pca(self):
        if not self.trained_model_meta:
            return

        # On récupère les colonnes d'entrée utilisées par le modèle
        inputs = self.trained_model_meta["inputs"]
        
        # On extrait les données correspondantes, on convertit en numérique et on retire les NaN
        df_inputs = self.df[inputs].apply(pd.to_numeric, errors="coerce").dropna()

        if df_inputs.shape[1] < 2:
            messagebox.showwarning("ACP", "Il faut au moins 2 variables d'entrée pour afficher les graphiques de l'ACP.")
            return

        # Création de la fenêtre Pop-up
        popup = ctk.CTkToplevel(self)
        popup.title("Analyse en Composantes Principales (ACP)")
        popup.geometry("1100x600")
        popup.configure(fg_color=BG)
        popup.transient(self)
        popup.grab_set()

        # --- CALCUL DE L'ACP ---
        # Sécurité : on prend au maximum 9 composantes, ou le nombre de colonnes si inférieur
        n_comp = min(9, df_inputs.shape[1])
        
        # On standardise les données pour que l'ACP soit pertinente
        scaled_data = StandardScaler().fit_transform(df_inputs)
        
        pca = PCA(n_components=n_comp)
        pca.fit(scaled_data)

        # --- CRÉATION DES GRAPHIQUES MATPLOTLIB ---
        # Au lieu de plt.subplots(), on utilise Figure() pour l'intégrer à Tkinter
        fig = Figure(figsize=(12, 5), dpi=100)
        # Fond transparent pour coller au style de l'app
        fig.patch.set_facecolor(CARD[0] if ctk.get_appearance_mode() == "Light" else CARD[1])
        text_color = TXT[0] if ctk.get_appearance_mode() == "Light" else TXT[1]

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Graphique 1 : Variance expliquée
        ax1.plot(np.arange(1, n_comp + 1), pca.explained_variance_ratio_, marker='o', color="#007AFF")
        ax1.set_xlabel("Nombre de composantes principales", color=text_color)
        ax1.set_ylabel("Proportion de variance expliquée", color=text_color)
        ax1.set_title("Variance expliquée par chaque composante", color=text_color)
        ax1.tick_params(colors=text_color)

        # Graphique 2 : Contribution des paramètres
        pcs = pca.components_
        ax2.scatter(pcs[0], pcs[1], color="#34C759")
        
        for (x_coordinate, y_coordinate, feature_name) in zip(pcs[0], pcs[1], inputs):
            ax2.text(x_coordinate + 0.02, y_coordinate + 0.02, feature_name, color=text_color)                        
            
        ax2.set_xlabel("Contribution à la PC1", color=text_color)
        ax2.set_ylabel("Contribution à la PC2", color=text_color)
        ax2.set_title("Contribution des paramètres", color=text_color)
        ax2.tick_params(colors=text_color)
        
        # Ajout d'une grille légère et ajustement
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()

        # --- INTÉGRATION DANS TKINTER ---
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

        # Bouton fermer
        ctk.CTkButton(
            popup, text="Fermer", command=popup.destroy,
            corner_radius=12, height=40, fg_color=BLUE, hover_color=BLUE_HOV,
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(0, 20))


if __name__ == "__main__":
    RegressionApp().mainloop()
