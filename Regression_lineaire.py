from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#Régression linéaire en supprimant les lignes NA

colonnes_utiles = ["P1_reel", "P2", "Lt", "Lq_reel", "P4", "P5", "V", "E2", "P6_reel", "E1"] #On garde seulement les colonnes utiles pour la régression
masque = df_normalize.columns.isin(colonnes_utiles)
df2 = df_normalize.loc[:, masque]

df_normalizedrop = df2.dropna() #On supprime toutes les lignes où il y a des valeurs manquantes
df_normalizedrop.head() 
print(f"Nombre de lignes conservées pour l'entraînement : {len(df_normalizedrop)}") #On vérifie qu'il reste suffisament de donnés pour la régression
X = df_normalizedrop.drop(columns = ["E1"]) #On divise l'input ou l'output
Y = df_normalizedrop["E1"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2) #Séparation entre les données d'entrainement et les données test

modele = LinearRegression() #Initialisation de la régression
modele.fit(X_train, y_train) #Entrainement du modèle

test = modele.predict(X_test) #test du modèle

erreur = mean_absolute_error(y_test, test) #On calcule l'erreur absolue du modèle

print(f"Le modèle se trompe en moyenne de : {erreur}")

#Régression linéaire N°2 en supprimant les lignes NA (calcul direct moyenne)

df_normalize = df_normalize.apply(pd.to_numeric, errors='coerce') #Transformation du tableau en valeurs numériques

colonnes_a_moyenner = ['E2.1', 'E2.2', 'E2.3', 'E2.4'] #On fait la moyenne des 4 colonnes pour entrainer le modèle dessus
df_normalize['E2_Moyenne'] = df_normalize[colonnes_a_moyenner].mean(axis=1)

colonnes_utiles = ["P1_reel", "P2", "Lt", "Lq_reel", "P4", "P5", "V", "E2", "P6_reel", "E1", "E2_Moyenne"] #On garde seulement les colonnes utiles pour la régression
df_reduit = df_normalize[colonnes_utiles]

df_propre = df_reduit.dropna() #On supprime toutes les lignes où il y a des valeurs manquantes
print(f"Nombre de lignes conservées pour l'entraînement : {len(df_propre)}") #On vérifie qu'il reste suffisament de donnés pour la régression

X = df_propre.drop(columns=['E2_Moyenne']) #On divise l'input ou l'output
Y = df_propre['E2_Moyenne']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #Séparation entre les données d'entrainement et les données test

modele = LinearRegression() #Initialisation de la régression
modele.fit(X_train, y_train) #Entrainement du modèle

predictions = modele.predict(X_test) #test du modèle
erreur = mean_absolute_error(y_test, predictions) #On calcule l'erreur absolue du modèle

print(f"Le modèle se trompe en moyenne de : {erreur}")

#Régression linéaire N°3 en supprimant les lignes NA

features = ["P1_reel", "P2", "Lt", "Lq_reel", "P4", "P5", "V", "E2", "P6_reel"] #On garde seulement les colonnes utiles pour la régression
cibles = ["S_moy", "sigma_S"]
colonnes_utiles = features + cibles

df_normalize = df_normalize.apply(pd.to_numeric, errors='coerce') #Transformation du tableau en valeurs numériques

df_propre = df_normalize[colonnes_utiles].dropna() #On supprime toutes les lignes où il y a des valeurs manquantes
print(f"Nombre de lignes conservées pour l'entraînement : {len(df_propre)}")

X = df_propre[features] #On divise l'input ou l'output
Y = df_propre[cibles]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #Séparation entre les données d'entrainement et les données test

modele = LinearRegression() #Initialisation de la régression
modele.fit(X_train, y_train) #Entrainement du modèle
test_pred = modele.predict(X_test) #test du modèle

erreurs = mean_absolute_error(y_test, test_pred, multioutput='raw_values') #On calcule l'erreur absolue du modèle

print(f"Le modèle se trompe en moyenne de : {erreurs[0]:.4f} sur la moyenne (S_moy)")
print(f"Le modèle se trompe en moyenne de : {erreurs[1]:.4f} sur l'écart type (sigma_S)")