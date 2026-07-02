import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import (
    train_test_split, GroupShuffleSplit, KFold, StratifiedKFold,
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, log_loss, roc_auc_score,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_xgboost(
    X,
    y,
    groups=None,                     # id de groupe (ex. 'essai') : evite les fuites train/test et CV
    objective="binary:logistic",   # tache : 'reg:squarederror', 'binary:logistic', 'multi:softprob'...
    eval_metric="logloss",         # metrique pilotant early stopping + selection (coherente avec objective)
    num_class=None,                 
    test_size=0.2,                  
    n_trials=150,                   # budget de la recherche fine Optuna (Phase 5)
    nfold=5,                        # nb de plis de validation croisee
    eta_start=0.3,                 # rythme d'apprentissage pendant le reglage (Phases 0-4)
    eta_final=0.01,                 # rythme d'apprentissage final, plus prudent (Phases 5-6)
    early_stopping_rounds=50,       # arret auto de l'ajout d'arbres quand ca stagne
    seed=42,
    verbose=True,
):
    """
    Optimise un modele XGBoost de bout en bout.

    Si `groups` est fourni (ex. un identifiant d'essai quand plusieurs lignes
    proviennent du meme essai/individu), le split train/test respecte les
    groupes : toutes les lignes d'un meme groupe restent ensemble, cote train
    OU cote test, jamais les deux. Cela evite qu'un essai deja vu en train se
    retrouve en test, ce qui gonflerait artificiellement le score final.
    La validation croisee interne (Phases 1 a 5, choix des hyperparametres)
    n'utilise en revanche PAS `groups` : elle decoupe le train set ligne par
    ligne (KFold/StratifiedKFold), pour beneficier de plus de plis et de plus
    de donnees par pli, surtout utile quand il y a peu de groupes distincts.
    Cela ne concerne que le reglage des hyperparametres ; la metrique finale
    rapportee (test_metrics) reste calculee sur le test set groupe.

    Retourne un dict :
        'model'        : modele XGBoost final entraine (xgb.Booster)
        'best_params'  : hyperparametres retenus
        'n_rounds'     : nombre d'arbres du modele final
        'cv_score'     : meilleure note de validation croisee (Phase 5)
        'test_score'   : note de la metrique 'eval_metric' sur le test set (Phase 6)
        'test_metrics' : dict de metriques calculees sur le test set
                         - regression      : {'rmse', 'mae', 'r2'}
                         - classif. binaire : {'accuracy', 'logloss', 'auc'}
                         - multi-classes    : {'accuracy'}
        'study'        : objet Optuna complet (inspection / graphiques)
    """

    # --- Detection automatique du type de tache -----------------------------
    is_regression = objective.startswith("reg:")
    is_binary = objective.startswith("binary:")
    is_multi = objective.startswith("multi:")
    stratified = not is_regression  # stratification : classification seulement

    maximize_metrics = {"auc", "aucpr", "map", "ndcg"}
    maximize = eval_metric in maximize_metrics  # sens d'optimisation de la metrique

    # --- Separation train / test (le test est isole jusqu'a la Phase 6) -----
    # `groups` ne sert QU'a cette separation-la : on veut une mesure honnete
    # de generalisation a un essai jamais vu (le test set final, evalue une
    # seule fois en Phase 6, ne doit jamais contenir un essai deja vu en train).
    if groups is not None:
        groups = np.asarray(groups)
        train_idx, test_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            .split(X, y, groups)
        )
        X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_test = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
        y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        y_test = y.iloc[test_idx] if hasattr(y, "iloc") else y[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y if stratified else None,
            random_state=seed,
        )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # --- Parametres communs a toutes les evaluations ------------------------
    fixed = {"objective": objective, "eval_metric": eval_metric, "seed": seed, "tree_method": "hist",
"nthread": -1}
    if is_multi:
        if num_class is None:
            raise ValueError("num_class doit etre fourni pour objective='multi:softprob'.")
        fixed["num_class"] = num_class

    # --- Plis de CV (reglage des hyperparametres, Phases 1 a 5 uniquement) --
    # Deliberement PAS groupes par essai, meme si `groups` est fourni : cette CV
    # ne sert qu'a comparer des jeux d'hyperparametres entre eux, pas a mesurer
    # la performance finale (celle-ci vient du test set, lui reste groupe et
    # jamais touche avant la Phase 6). En l'ungroupant, on exploite chaque ligne
    # du train individuellement -> des plis plus nombreux et plus stables,
    # surtout utile quand il y a peu d'essais distincts (CV par groupe alors
    # trop bruitee, ex. 2 plis pour 6 essais). Le cout : un score de CV
    # legerement optimiste (des lignes d'un meme essai peuvent se retrouver de
    # part et d'autre d'un pli), qui influence seulement le choix des
    # hyperparametres, pas la metrique finale rapportee.
    if stratified:
        splitter = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=nfold, shuffle=True, random_state=seed)
    cv_folds = list(splitter.split(X_train, y_train))

    # --- Outil interne : la "note" d'une configuration (validation croisee) --
    def cv_score(params, num_boost_round=5000):
        cv_kwargs = dict(
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            maximize=maximize,
            seed=seed,
            verbose_eval=False,
        )
        if cv_folds is not None:
            cv_kwargs["folds"] = cv_folds
        else:
            cv_kwargs["nfold"] = nfold
            cv_kwargs["stratified"] = stratified
        cv_results = xgb.cv({**fixed, **params}, dtrain, **cv_kwargs)
        best_round = cv_results.shape[0]
        best_score = cv_results.iloc[-1, 2]   # test-metric-mean
        best_std = cv_results.iloc[-1, 3]     # test-metric-std
        return best_score, best_std, best_round

    def better(a, b):
        return a > b if maximize else a < b
    worst = -np.inf if maximize else np.inf

    # PHASE 0 — Rythme d'apprentissage + baseline

    p = {
        "eta": eta_start,
        "max_depth": 6, "min_child_weight": 1,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "gamma": 0, "lambda": 1, "alpha": 0,
    }
    score, std, n_rounds = cv_score(p)

    # PHASE 1 — Taille de chaque eleve (regles ENSEMBLE car couples)

    best = worst; combo = None
    for depth in [3, 4, 5, 6, 7, 8]:
        for min_cw in [1, 3, 5, 7]:
            s, _, _ = cv_score({**p, "max_depth": depth, "min_child_weight": min_cw})
            if better(s, best):
                best, combo = s, (depth, min_cw)
    p["max_depth"], p["min_child_weight"] = combo

    # PHASE 2 — Seuil de motivation pour couper une branche (gamma)

    best = worst; g_opt = 0
    for gamma in [0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        s, _, _ = cv_score({**p, "gamma": gamma})
        if better(s, best):
            best, g_opt = s, gamma
    p["gamma"] = g_opt


    # PHASE 3 — Empecher les eleves de tous penser pareil

    best = worst; combo = None
    for sub in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for col in [0.6, 0.7, 0.8, 0.9, 1.0]:
            s, _, _ = cv_score({**p, "subsample": sub, "colsample_bytree": col})
            if better(s, best):
                best, combo = s, (sub, col)
    p["subsample"], p["colsample_bytree"] = combo


    # PHASE 4 — Penaliser les corrections trop violentes (lambda, alpha)

    best = worst; combo = None
    for lam in [0, 0.1, 1, 5, 10, 50]:
        for alp in [0, 0.1, 1, 5, 10]:
            s, _, _ = cv_score({**p, "lambda": lam, "alpha": alp})
            if better(s, best):
                best, combo = s, (lam, alp)
    p["lambda"], p["alpha"] = combo


    # PHASE 5 — Affinage fin automatique (Optuna) AUTOUR de la bonne zone

    d_opt, mcw_opt = p["max_depth"], p["min_child_weight"]
    g_opt = p["gamma"]
    sub_opt, col_opt = p["subsample"], p["colsample_bytree"]
    lam_opt, alp_opt = p["lambda"], p["alpha"]

    def objective_fn(trial):
        params = {
            "eta": eta_final,
            "max_depth": trial.suggest_int("max_depth", max(3, d_opt - 2), d_opt + 2),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", max(0.5, mcw_opt - 2), mcw_opt + 3),
            "gamma": trial.suggest_float("gamma", max(0.0, g_opt - 0.2), g_opt + 0.3),
            "subsample": trial.suggest_float(
                "subsample", max(0.5, sub_opt - 0.15), min(1.0, sub_opt + 0.15)),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", max(0.5, col_opt - 0.15), min(1.0, col_opt + 0.15)),
            "lambda": trial.suggest_float("lambda", 1e-2, lam_opt * 5 + 1, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, alp_opt * 5 + 1, log=True),
        }
        s, _, rounds = cv_score(params, num_boost_round=10000)
        trial.set_user_attr("n_rounds", rounds)
        return s

    study = optuna.create_study(
        direction="maximize" if maximize else "minimize",
        sampler=TPESampler(seed=seed),
    )
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=verbose)

    best_params = {**fixed, **study.best_params, "eta": eta_final}
    final_n_rounds = study.best_trial.user_attrs["n_rounds"]

    # PHASE 6 — Examen final honnete + calcul des metriques sur le test set
    # Le modele final est entraine sur TOUT le train, puis evalue UNE SEULE
    # FOIS sur le test set jamais touche. On calcule ici les metriques
    # metier adaptees au type de tache.

    model = xgb.train(best_params, dtrain, num_boost_round=final_n_rounds)
    raw_preds = model.predict(dtest)  # regression : valeurs ; binaire : proba ; multi : matrice proba

    # Note de la metrique d'optimisation (via l'evaluateur integre XGBoost)
    test_eval_str = model.eval(dtest)
    test_score = float(test_eval_str.split(":")[-1])

    # Metriques metier selon la tache
    if is_regression:
        rmse = float(np.sqrt(mean_squared_error(y_test, raw_preds)))
        mae = float(mean_absolute_error(y_test, raw_preds))
        r2 = float(r2_score(y_test, raw_preds))
        test_metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    elif is_binary:
        proba = raw_preds
        labels = (proba >= 0.5).astype(int)
        acc = float(accuracy_score(y_test, labels))
        ll = float(log_loss(y_test, proba, labels=[0, 1]))
        try:
            auc = float(roc_auc_score(y_test, proba))
        except ValueError:
            auc = float("nan")  # une seule classe presente dans y_test
        test_metrics = {"accuracy": acc, "logloss": ll, "auc": auc}

    else:  # multi-classes
        labels = np.argmax(raw_preds, axis=1)
        acc = float(accuracy_score(y_test, labels))
        test_metrics = {"accuracy": acc}

    return {
        "model": model,
        "best_params": best_params,
        "n_rounds": final_n_rounds,
        "cv_score": study.best_value,
        "test_score": test_score,
        "test_metrics": test_metrics,
        "study": study,
    }
