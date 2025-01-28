import os
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from my_function import calculate_percentages, assign_classes, save_results, calculate_polygon_surface

# Chemins des fichiers
sample_filename = '/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ.shp'
classification_filename = '/home/onyxia/work/results/data/classif/carte_essences_echelle_pixel.tif'
out_shapefile = '/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ_stand.shp'
out_matrix = '/home/onyxia/work/results/figure/matrice_confusion_peuplements.png'
output_csv_path = '/home/onyxia/work/results/data/classif/pourcentages_classes_peuplement.csv'

# Charger les données vectorielles des polygones
polygon_gdf = gpd.read_file(sample_filename)

# Calculer la surface des polygones en hectares
polygon_gdf = calculate_polygon_surface(polygon_gdf)

# Calculer les pourcentages des classes et sauvegarder les résultats
sample_data, percentages_df = calculate_percentages(
    sample_filename, classification_filename, output_csv_path
)

# Assigner les classes prédites selon les règles
sample_data = assign_classes(sample_data, percentages_df)

# Vérifier la présence des colonnes nécessaires
if 'Code_Objet' not in sample_data.columns or 'code_predit' not in sample_data.columns:
    raise ValueError("Les colonnes 'Code_Objet' ou 'code_predit' sont manquantes dans sample_data.")

# Remplacer 'Inconnu' par 0 dans les colonnes Code_Objet et code_predit
sample_data['Code_Objet'] = sample_data['Code_Objet'].replace('Inconnu', 0)
sample_data['code_predit'] = sample_data['code_predit'].replace('Inconnu', 0)

# Convertir les colonnes en entiers
sample_data['Code_Objet'] = pd.to_numeric(sample_data['Code_Objet'], errors='coerce').fillna(0).astype(int)
sample_data['code_predit'] = pd.to_numeric(sample_data['code_predit'], errors='coerce').fillna(0).astype(int)

# Renommer la colonne code_predit pour éviter des conflits
sample_data.rename(columns={'code_predit': 'code_pred'}, inplace=True)

# Sauvegarder les résultats dans un shapefile
save_results(sample_data, out_shapefile)

# Produire la matrice de confusion
y_true = sample_data['Code_Objet'].values
y_pred = sample_data['code_pred'].values

# Vérifier les formes des tableaux
print(f"Type de y_true : {type(y_true)}, forme : {y_true.shape}")
print(f"Type de y_pred : {type(y_pred)}, forme : {y_pred.shape}")

# S'assurer que les tableaux sont unidimensionnels
y_true = np.ravel(y_true)
y_pred = np.ravel(y_pred)

# Convertir en entiers
y_true = y_true.astype(int)
y_pred = y_pred.astype(int)

# Synchroniser les longueurs si nécessaire
if len(y_true) != len(y_pred):
    print(f"Différence de longueur détectée : y_true = {len(y_true)}, y_pred = {len(y_pred)}")
    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]
    print(f"Synchronisation effectuée : y_true = {len(y_true)}, y_pred = {len(y_pred)}")

# Vérifier les classes présentes
labels = [0, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27, 28, 29]
print(f"Classes trouvées dans y_true : {np.unique(y_true)}")
print(f"Classes trouvées dans y_pred : {np.unique(y_pred)}")

# Calculer la matrice de confusion
try:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Matrice de confusion calculée avec succès.")
except ValueError as e:
    print(f"Erreur lors du calcul de la matrice de confusion : {e}")
    raise

# Afficher et sauvegarder la matrice de confusion
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice de confusion')
plt.colorbar()

# Ajouter des étiquettes sur les axes
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')

# Ajuster la mise en page et sauvegarder
plt.tight_layout()
plt.savefig(out_matrix)
plt.show()

print("Processus terminé avec succès !")
print(f"- Résultats enregistrés dans : {out_shapefile}")
print(f"- Matrice de confusion sauvegardée dans : {out_matrix}")
