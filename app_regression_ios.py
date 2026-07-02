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
    """Fabriques de modèles (imports paresseux : la fenêtre s'ouvre sans sklearn)."""
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline


    return {
        # Régression linéaire simple (moindres carrés ordinaires)
        "Régression linéaire": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        # Ridge = régression linéaire régularisée (le nom affiché "Polynomiale" est trompeur :
        # aucune expansion polynomiale n'est faite ici, c'est bien une régression linéaire pénalisée)
        "Régression Polynomiale":          lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        # Modèle d'ensemble à base d'arbres de décision (moyenne de 300 arbres)
        "Forêt aléatoire":     lambda: make_pipeline(StandardScaler(),
                                   RandomForestRegressor(n_estimators=300, random_state=0)),
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


class _XGBBoosterPredictor:
    """Enveloppe un Booster XGBoost pour exposer un .predict(X) identique aux estimateurs scikit-learn."""

    def __init__(self, booster):
        self.booster = booster

    def predict(self, X):
        import xgboost as xgb
        return self.booster.predict(xgb.DMatrix(X))


class RegressionApp(ctk.CTk):
        
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.title("Atelier de régression")
        self.geometry("1060x760")
        self.minsize(900, 640)
        self.configure(fg_color=BG)

        # === STOCKAGE DES DONNÉES UTILISATEUR ===================================
        # self.df contient l'intégralité du CSV chargé par l'utilisateur, tel quel,
        # en mémoire (RAM) sous forme de DataFrame pandas. Rien n'est persisté sur
        # disque ni envoyé ailleurs : les données ne vivent que le temps de la
        # session et sont perdues à la fermeture de l'application ou au rechargement
        # d'un nouveau fichier (voir load_csv()).
        self.df = None
        self.input_vars = []    # liste de (colonne, BooleanVar) : colonnes cochées comme entrées X
        self.output_vars = []   # liste de (colonne, BooleanVar) : colonnes cochées comme sorties y
        self.trained_model = None       # dernier modèle entraîné sur l'ensemble des données (exportable)
        self.trained_model_meta = None  # infos associées (colonnes X/y, nom du modèle)
        self.groups_col = None  # nom de la colonne de regroupement ('essai') détectée automatiquement

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
        card = ctk.CTkFrame(self, fg_color=CARD, corner_radius=RADIUS)
        card.grid(row=row, column=0, sticky="nsew", padx=24, pady=8)
        if title:
            ctk.CTkLabel(
                card, text=title, font=ctk.CTkFont(size=13, weight="bold"),
                text_color=SUBTXT,
            ).pack(anchor="w", padx=18, pady=(14, 0))
        return card

    def _build_file_card(self):
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

    def _build_preview_card(self):
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
        ctk.set_appearance_mode("dark" if value == "Sombre" else "light")
        self._style_treeview()

    def _style_treeview(self):
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
        # Ouvre un sélecteur de fichier natif : l'utilisateur choisit un CSV sur
        # son propre disque (aucun upload réseau, tout se passe en local).
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

        # >>> Point d'entrée des données utilisateur en mémoire <<<
        # Le CSV entier est chargé dans self.df (DataFrame pandas), qui reste
        # l'unique copie des données utilisées pour l'aperçu et l'entraînement.
        self.df = df

        # Détection automatique d'une colonne de regroupement ('essai') : plusieurs
        # lignes peuvent provenir du même essai/individu ; on évite qu'un même
        # essai se retrouve à la fois en train et en test/CV (fuite de données).
        # Cette colonne n'est ni une entrée X ni une sortie y : elle est retirée
        # des cases à cocher et utilisée directement comme `groups` à l'entraînement.
        group_candidates = [c for c in df.columns if str(c).strip().lower() == "essai"]
        self.groups_col = group_candidates[0] if group_candidates else None

        label = f"{path.split('/')[-1]}  ·  {df.shape[0]} lignes × {df.shape[1]} colonnes"
        if self.groups_col:
            label += f"  ·  regroupement auto par « {self.groups_col} »"
        self.file_label.configure(text=label)

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
        # Ouvre une pop-up d'explication pour le modèle actuellement sélectionné
        # dans le menu déroulant. Le contenu est un espace réservé, à compléter
        # ultérieurement avec la véritable explication de chaque modèle.
        model_name = self.model_var.get() if hasattr(self, "model_var") else ""
        if not model_name:
            messagebox.showwarning("Aucun modèle", "Sélectionne d'abord un modèle.")
            return

        popup = ctk.CTkToplevel(self)
        popup.title(f"Explication — {model_name}")
        popup.geometry("520x480")
        popup.configure(fg_color=BG)
        popup.transient(self)
        popup.grab_set()

        ctk.CTkLabel(
            popup, text=model_name, font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TXT, wraplength=460, justify="left",
        ).pack(anchor="w", padx=20, pady=(20, 12))

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

        for title, body in sections:
            ctk.CTkLabel(
                popup, text=title, font=ctk.CTkFont(size=14, weight="bold"),
                text_color=BLUE, wraplength=460, justify="left",
            ).pack(anchor="w", padx=20, pady=(4, 2))
            ctk.CTkLabel(
                popup, text=body, text_color=SUBTXT,
                font=ctk.CTkFont(size=13), wraplength=460, justify="left",
            ).pack(anchor="w", padx=20, pady=(0, 10))

        ctk.CTkButton(
            popup, text="Fermer", command=popup.destroy,
            corner_radius=12, height=36, fg_color=BLUE, hover_color=BLUE_HOV,
        ).pack(anchor="e", padx=20, pady=(0, 20))

    def _populate_checkboxes(self, columns):
        # Reconstruit la liste des colonnes disponibles (X et y) à partir des
        # colonnes du CSV chargé dans self.df ; ces cases à cocher déterminent
        # quelles données utilisateur seront effectivement envoyées au modèle.
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
            ).pack(anchor="w", padx=14, pady=4)
            self.input_vars.append((col, v_in))

            v_out = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(
                self.output_frame, text=str(col), variable=v_out,
                fg_color=BLUE, hover_color=BLUE_HOV, corner_radius=6,
                text_color=TXT, font=ctk.CTkFont(size=13),
            ).pack(anchor="w", padx=14, pady=4)
            self.output_vars.append((col, v_out))

    def _populate_preview(self, df):
        # Affiche uniquement les 10 premières lignes des données utilisateur
        # (df.head(10)) dans le tableau ; self.df en mémoire contient lui la
        # totalité du jeu de données.
        self.tree.delete(*self.tree.get_children())
        cols = list(df.columns.astype(str))
        self.tree["columns"] = cols
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=130, anchor="center", stretch=False)
        for _, r in df.head(10).iterrows():
            self.tree.insert("", tk.END, values=list(r))

    def _checked(self, var_list):
        return [col for col, v in var_list if v.get()]

    def train_model(self):
        # === CALCUL / ENTRAÎNEMENT DU MODÈLE ====================================
        # Toute cette méthode constitue le pipeline de calcul : elle part des
        # données utilisateur (self.df), sélectionne les colonnes X/y choisies
        # par l'utilisateur, construit le modèle puis l'évalue. Pour XGB_LABEL,
        # tout le travail (split, CV, réglage Optuna, métriques) est délégué à
        # optimize_xgboost() (optimize_xgboost.py) ; pour les 3 autres modèles,
        # on garde le pipeline scikit-learn + validation croisée existant.
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
        # se retrouverait mélangé entre train et test.
        groups = None
        if self.groups_col:
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
        self.trained_model = None
        self.trained_model_meta = None
        self.update_idletasks()

        if model_name == XGB_LABEL:
            self._train_xgboost(inputs, outputs, X, y, n, groups)
        else:
            self._train_sklearn(model_name, inputs, outputs, X, y, n, groups)

    def _train_sklearn(self, model_name, inputs, outputs, X, y, n, groups):
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
            # qu'un même essai ne se retrouve jamais scindé entre deux plis.
            scoring = ["r2", "neg_root_mean_squared_error"]
            if groups is not None:
                k = min(5, n, len(np.unique(groups)))
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
        if model_name in ["Régression linéaire", "Régression Polynomiale"]:
            self.coef_button.configure(state="normal")

    def _train_xgboost(self, inputs, outputs, X, y, n, groups):
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

        # Peu de données (ou peu de groupes distincts) : on réduit le nombre de
        # plis de validation croisée (par défaut 5 dans optimize_xgboost) pour
        # éviter des plis trop petits. `groups` (colonne 'essai') est transmis
        # tel quel à optimize_xgboost, qui l'utilise pour le split train/test
        # (GroupShuffleSplit) et la CV (GroupKFold) : un même essai ne se
        # retrouve jamais scindé entre train et test.
        effective_n = len(np.unique(groups)) if groups is not None else n
        nfold = min(5, max(2, effective_n // 3))

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

    def export_model(self):
        # Sauvegarde sur disque du modèle final (entraîné sur toutes les
        # données) au format joblib, avec les métadonnées nécessaires pour le
        # réutiliser (colonnes X/y attendues, nom du modèle).
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


if __name__ == "__main__":
    RegressionApp().mainloop()
