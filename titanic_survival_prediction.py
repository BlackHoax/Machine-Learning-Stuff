import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# On charge notre dataframe à partir du fichier titanic.csv
df = pd.read_csv('/home/leuzly/Bureau/titanic.csv')

# On recupère notre target dans y
y = df['Survived'].values

# On change les strings du la série Sexe en booléens
df['Male'] = df['Sex'] == 'male'

# On recupére nos features
X = df[['Pclass', 'Male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values

# On charge le modele de régression logistique
model = LogisticRegression()

# On entraine notre modele
model.fit(X, y)

# On affiche son score
score = model.score(X,y)
print(f"Le score est de {score*100:.2f}%")


print("[+] Aurez vous survécu au naufrage du Titanic ? Remplissez ce formulaire pour vérifier ! [+]")

# Recupération des informations
try:
    pclass = int(input("[ Vous voyagez dans quelle classe ? (1 à 3) : "))
    male = int(input("[ Vous êtes un Homme (1) ou une Femme(2) ? : "))    
    age = int(input("[ Votre Age: "))
    sb_sp = int(input("[ Nombre de partenaires à bord (Epoux(se), Petit(es) ami(es)): "))
    pr_ch = int(input("[ Nombre de Parents ou d'Enfants à bord: "))
    fare = float(input("[ Prix du billet (en $): "))
except ValueError:
    print("Entrer des valeurs numériques !")

# Le sexe sera traité en tant que booléen
if male == 1:
    male = True
else:
    male = False


person_feature = [pclass, male, age, sb_sp, pr_ch, fare]

# On effectue la prédiction
if model.predict([person_feature]) == 1:
    print("-"*60)
    print("Bravo, Vous aurez survécu au naufrage du Titanic !")
else:
    print("-"*60)
    print("Dommage, Vous n'aurez pas survécu au naufrage du Titanic")

