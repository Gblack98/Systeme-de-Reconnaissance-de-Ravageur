# Système de Reconnaissance de Ravageurs Agricoles

## Description
Cette application Streamlit permet de détecter et de classifier les ravageurs agricoles à partir d'images ou de vidéos. Elle utilise un modèle de deep learning entraîné pour identifier différents types de ravageurs.

## Fonctionnalités
- **Détection d'images** : Téléversez une image pour identifier le ravageur présent.
- **Détection de vidéos** : Téléversez une vidéo pour analyser chaque frame et détecter les ravageurs.
- **Affichage des résultats** : Les résultats incluent le type de ravageur détecté et le niveau de confiance de la prédiction.

## Pourquoi certains fichiers sont sur Google Drive ?
Le modèle de deep learning (`modele_agricultural_pests.h5`) et d'autres fichiers volumineux ne sont pas inclus dans ce dépôt GitHub en raison des limitations de taille de fichier de GitHub (100 MB par fichier). Ces fichiers sont hébergés sur Google Drive pour des raisons pratiques :
- **Taille du modèle** : Le modèle est trop volumineux pour être stocké directement sur GitHub.
- **Facilité d'accès** : Google Drive permet un téléchargement facile et rapide des fichiers volumineux.

### Lien vers les fichiers sur Google Drive
- [Télécharger le modèle](https://drive.google.com/file/d/10QhOSIebVwsgasKHar-bLnwtyReEAWXL/view?usp=sharing)

## Comment exécuter l'application
1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/Systeme-de-Reconnaissance-de-Ravageur.git