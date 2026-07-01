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



# --- Palette iOS (clair, sombre) -------------------------------------------
BG       = ("#F2F2F7", "#000000")   # fond groupé iOS
CARD     = ("#FFFFFF", "#1C1C1E")   # cartes
TXT      = ("#1C1C1E", "#FFFFFF")   # texte principal
SUBTXT   = ("#8E8E93", "#8E8E93")   # texte secondaire (gris iOS)
BLUE     = ("#007AFF", "#0A84FF")   # bleu système iOS
BLUE_HOV = ("#0063CC", "#0974E0")
GREEN    = ("#34C759", "#30D158")   # vert système iOS

RADIUS = 16


def build_models():
    """Fabriques de modèles (imports paresseux : la fenêtre s'ouvre sans sklearn)."""
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    return {
        "Régression linéaire": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "Régression Polynomiale":          lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Forêt aléatoire":     lambda: make_pipeline(StandardScaler(),
                                   RandomForestRegressor(n_estimators=300, random_state=0)),
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

        self.df = None
        self.input_vars = []    # liste de (colonne, BooleanVar)
        self.output_vars = []

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

        self.df = df
        self.file_label.configure(
            text=f"{path.split('/')[-1]}  ·  {df.shape[0]} lignes × {df.shape[1]} colonnes"
        )
        self._populate_checkboxes(df.columns)
        
        self._populate_preview(df)
        self.result_label.configure(text="")

    def _populate_checkboxes(self, columns):
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

        estimator = build_models()[self.model_var.get()]()
        if getattr(y, "ndim", 1) == 2 and y.shape[1] > 1:
            estimator = MultiOutputRegressor(estimator)

        self.result_label.configure(text="Calcul…", text_color=SUBTXT)
        self.update_idletasks()
        try:
            k = min(5, n)
            cv = KFold(n_splits=k, shuffle=True, random_state=0)
            scores = cross_val_score(estimator, X, y, cv=cv, scoring="r2")
        except Exception as exc:  # noqa: BLE001
            self.result_label.configure(text="")
            messagebox.showerror("Erreur d'entraînement", str(exc))
            return

        self.result_label.configure(
            text=f"R² (CV {k}-plis) = {scores.mean():.3f}  ±  {scores.std():.3f}",
            text_color=GREEN,
        )


if __name__ == "__main__":
    RegressionApp().mainloop()
