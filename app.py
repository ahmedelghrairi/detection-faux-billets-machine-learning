"""
ONCFM - Detection automatique de faux billets
=============================================

Deux modes d'utilisation :

  1) Fichier CSV     python app.py chemin/vers/fichier.csv
  2) Saisie manuelle python app.py            (sans argument)

Le script affiche pour chaque billet la probabilite qu'il soit vrai,
puis le verdict VRAI / FAUX.
"""

import sys
from pathlib import Path

import pandas as pd
import joblib

# Le modele est cherche A COTE de ce script, pas dans le dossier courant.
# Le script fonctionne donc meme s'il est lance depuis un autre repertoire.
DOSSIER = Path(__file__).parent
CHEMIN_MODELE = DOSSIER / "modele_billets.joblib"


def charger_modele():
    """Charge le pipeline, le seuil et la liste des colonnes attendues."""
    if not CHEMIN_MODELE.exists():
        print(f"ERREUR : modele introuvable ({CHEMIN_MODELE})")
        print("Le fichier modele_billets.joblib doit se trouver a cote de app.py.")
        sys.exit(1)

    modele = joblib.load(CHEMIN_MODELE)
    return modele["pipeline"], modele["seuil"], modele["colonnes"]


def lire_fichier(chemin, colonnes):
    """Lit un CSV et renvoie (mesures, identifiants). Gere les formats variables."""
    chemin = Path(chemin)

    if not chemin.exists():
        print(f"ERREUR : fichier introuvable ({chemin.resolve()})")
        sys.exit(1)

    # sep=None : pandas detecte seul le separateur (virgule, point-virgule, tabulation)
    try:
        donnees = pd.read_csv(chemin, sep=None, engine="python")
    except Exception as e:
        print(f"ERREUR : lecture impossible ({e})")
        sys.exit(1)

    # Defense 4 : une mesure obligatoire manque
    absentes = [c for c in colonnes if c not in donnees.columns]
    if absentes:
        print("ERREUR : colonnes manquantes dans le fichier -> " + ", ".join(absentes))
        print("Colonnes attendues : " + ", ".join(colonnes))
        print("Colonnes trouvees  : " + ", ".join(donnees.columns))
        sys.exit(1)

    if len(donnees) == 0:
        print("ERREUR : le fichier ne contient aucune ligne.")
        sys.exit(1)

    # Defense 2 : pas de colonne id -> on en fabrique une
    if "id" in donnees.columns:
        identifiants = donnees["id"].astype(str)
    else:
        identifiants = pd.Series([f"billet_{i}" for i in range(1, len(donnees) + 1)])
        print("Note : aucune colonne 'id', identifiants generes automatiquement.")

    # Defense 3 : colonnes en trop ignorees, et ordre impose par le modele
    mesures = donnees[colonnes].copy()

    # Defense 7 : virgule decimale (export Excel francais) -> conversion en nombre
    for col in colonnes:
        if not pd.api.types.is_numeric_dtype(mesures[col]):
            mesures[col] = (mesures[col].astype(str)
                                        .str.replace(",", ".", regex=False)
                                        .str.strip())
            mesures[col] = pd.to_numeric(mesures[col], errors="coerce")

    if mesures.isna().all(axis=None):
        print("ERREUR : aucune mesure numerique exploitable dans le fichier.")
        sys.exit(1)

    # Defense 5 : cases vides -> signalees, mais le pipeline sait les completer
    nb_vides = int(mesures.isna().sum().sum())
    if nb_vides > 0:
        print(f"Note : {nb_vides} mesure(s) manquante(s), estimee(s) par le modele.")

    return mesures, identifiants


def saisie_manuelle(colonnes):
    """Demande les 6 mesures une par une au clavier."""
    print("Saisie des mesures du billet (en mm).")
    print("Utilisez le point comme separateur decimal, exemple : 112.5")
    print()

    valeurs = {}
    for nom in colonnes:
        while True:
            saisie = input(f"  {nom} : ").strip().replace(",", ".")
            try:
                valeurs[nom] = float(saisie)
                break
            except ValueError:
                print("    Valeur invalide, entrez un nombre.")

    mesures = pd.DataFrame([valeurs], columns=colonnes)
    identifiants = pd.Series(["billet_saisi"])
    return mesures, identifiants


def predire(pipeline, seuil, mesures, identifiants):
    """Calcule les probabilites et applique le seuil de decision."""
    proba = pipeline.predict_proba(mesures)[:, 1]

    return pd.DataFrame({
        "id": identifiants.values,
        "probabilite_vrai": proba.round(4),
        "verdict": ["VRAI" if p >= seuil else "FAUX" for p in proba],
    })


def afficher(resultats, seuil):
    """Affiche le tableau des resultats et un resume."""
    print()
    print("=" * 46)
    print(f"RESULTATS  (seuil de decision : {seuil})")
    print("=" * 46)
    print(f"{'id':<16}{'proba vrai':>14}{'verdict':>14}")
    print("-" * 46)

    for _, ligne in resultats.iterrows():
        print(f"{ligne['id']:<16}{ligne['probabilite_vrai']:>14.4f}{ligne['verdict']:>14}")

    print("-" * 46)
    nb_faux = int((resultats["verdict"] == "FAUX").sum())
    nb_vrais = int((resultats["verdict"] == "VRAI").sum())
    print(f"{len(resultats)} billet(s) analyse(s) : {nb_vrais} vrai(s), {nb_faux} faux")
    print()


def main():
    pipeline, seuil, colonnes = charger_modele()

    if len(sys.argv) > 1:
        # Mode fichier : le chemin est passe en argument
        mesures, identifiants = lire_fichier(sys.argv[1], colonnes)
    else:
        # Mode manuel : aucun argument fourni
        mesures, identifiants = saisie_manuelle(colonnes)

    resultats = predire(pipeline, seuil, mesures, identifiants)
    afficher(resultats, seuil)


if __name__ == "__main__":
    main()
