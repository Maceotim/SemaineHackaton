"""
Atelier de régression — interface style iOS (CustomTkinter).

Fonctionnalités (identiques à la version Tkinter, habillage modernisé) :
  1. Charger un CSV.
  2. Cocher les colonnes d'entrée X et de sortie y.
  3. Aperçu des 10 premières lignes.
  4. Menu déroulant à 4 modèles + évaluation en validation croisée (R²).

Dépendances : customtkinter, pandas, scikit-learn
Installation : pip install customtkinter pandas scikit-learn
Lancement    : python app_regression_ios.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import customtkinter as ctk
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
# C'est ICI que sont définis les 4 modèles de régression proposés à l'utilisateur.
# Chaque modèle est un pipeline scikit-learn : standardisation (StandardScaler)
# puis l'algorithme de régression proprement dit. C'est ce dictionnaire qui est
# utilisé plus bas dans train_model() pour instancier le modèle choisi.

class MaRegressionLineaire(BaseEstimator, RegressorMixin):
        
    def __init__(self):
        self.modele = LinearRegression()

    def fit(self, X, y):
        self.modele.fit(X, y)
        return self

    def predict(self, X):
        predictions = self.modele.predict(X)
        return predictions

def build_models():
    """Fabriques de modèles (imports paresseux : la fenêtre s'ouvre sans sklearn)."""
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline


    return {
        # Régression linéaire simple (moindres carrés ordinaires)
        "Régression linéaire": lambda: make_pipeline(StandardScaler(), MaRegressionLineaire()),
        # Ridge = régression linéaire régularisée (le nom affiché "Polynomiale" est trompeur :
        # aucune expansion polynomiale n'est faite ici, c'est bien une régression linéaire pénalisée)
        "Régression Polynomiale":          lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        # Modèle d'ensemble à base d'arbres de décision (moyenne de 300 arbres)
        "Forêt aléatoire":     lambda: make_pipeline(StandardScaler(),
                                   RandomForestRegressor(n_estimators=300, random_state=0)),
        # Boosting de gradient par histogramme (algorithme type LightGBM)
        "Gradient Boosting":   lambda: make_pipeline(StandardScaler(),
                                   HistGradientBoostingRegressor(random_state=0)),
    }


MODEL_NAMES = ["Régression linéaire", "Régression Polynomiale", "Forêt aléatoire", "Gradient Boosting"]


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
            df = pd.read_csv(path, sep=None, engine="python")   # détection auto du séparateur
        except Exception as exc:
            messagebox.showerror("Erreur de lecture", f"Lecture impossible :\n{exc}")
            return

        # >>> Point d'entrée des données utilisateur en mémoire <<<
        # Le CSV entier est chargé dans self.df (DataFrame pandas), qui reste
        # l'unique copie des données utilisées pour l'aperçu et l'entraînement.
        self.df = df
        self.file_label.configure(
            text=f"{path.split('/')[-1]}  ·  {df.shape[0]} lignes × {df.shape[1]} colonnes"
        )
        self._populate_checkboxes(df.columns)

        self._populate_preview(df)
        self.result_label.configure(text="")
        self.trained_model = None
        self.trained_model_meta = None
        self.export_button.configure(state="disabled")

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
        # par l'utilisateur, construit le modèle (voir build_models() plus haut)
        # puis l'évalue par validation croisée.
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

        try:
            from sklearn.model_selection import cross_val_score, KFold
            from sklearn.multioutput import MultiOutputRegressor
        except ImportError:
            messagebox.showerror(
                "Dépendance manquante",
                "scikit-learn n'est pas installé.\n\npip install scikit-learn",
            )
            return

        # Extraction du sous-ensemble utile des données utilisateur (colonnes X + y
        # cochées), conversion en numérique et suppression des lignes incomplètes.
        data = self.df[inputs + outputs].apply(pd.to_numeric, errors="coerce").dropna()
        if data.empty:
            messagebox.showerror(
                "Données",
                "Aucune ligne exploitable (colonnes non numériques ou valeurs manquantes).",
            )
            return

        X = data[inputs].values
        y = data[outputs].values
        if y.shape[1] == 1:
            y = y.ravel()

        n = X.shape[0]
        if n < 5:
            messagebox.showwarning("Échantillon", f"Trop peu d'observations ({n}).")
            return

        # Instanciation du modèle choisi dans le menu déroulant (dictionnaire
        # défini dans build_models() en haut du fichier). Si plusieurs colonnes
        # de sortie y sont cochées, on enveloppe le modèle dans un
        # MultiOutputRegressor pour gérer la régression multi-sorties.
        estimator = build_models()[self.model_var.get()]()
        if getattr(y, "ndim", 1) == 2 and y.shape[1] > 1:
            estimator = MultiOutputRegressor(estimator)

        self.result_label.configure(text="Calcul…", text_color=SUBTXT)
        self.export_button.configure(state="disabled")
        self.trained_model = None
        self.trained_model_meta = None
        self.update_idletasks()
        try:
            # Validation croisée en k plis (k=5 max, ou moins si peu de données) :
            # le modèle est ré-entraîné et évalué k fois sur des découpages
            # différents des données, puis les scores R² sont moyennés. Ceci ne
            # sert qu'à MESURER la qualité du modèle (chaque modèle entraîné
            # pendant la CV est jeté ensuite).
            k = min(5, n)
            cv = KFold(n_splits=k, shuffle=True, random_state=0)
            scores = cross_val_score(estimator, X, y, cv=cv, scoring="r2")

            # Modèle final exportable : ré-entraîné une dernière fois sur
            # l'intégralité des données (X, y), pour tirer parti de toutes les
            # observations disponibles avant l'export.
            estimator.fit(X, y)
        except Exception as exc:  # noqa: BLE001
            self.result_label.configure(text="")
            messagebox.showerror("Erreur d'entraînement", str(exc))
            return

        self.result_label.configure(
            text=f"R² (CV {k}-plis) = {scores.mean():.3f}  ±  {scores.std():.3f}",
            text_color=GREEN,
        )

        self.trained_model = estimator
        self.trained_model_meta = {
            "model_name": self.model_var.get(),
            "inputs": inputs,
            "outputs": outputs,
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


if __name__ == "__main__":
    RegressionApp().mainloop()
