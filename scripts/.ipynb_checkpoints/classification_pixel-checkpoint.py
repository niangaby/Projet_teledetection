import sys
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import itertools  # Ajoutez cette ligne au début du fichier
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal, osr
from matplotlib import rcParams
import itertools

from my_function import report_from_dict_to_df, plot_cm, get_row_col_from_file, get_samples_from_roi, get_xy_from_file, get_projection, get_image_bounds, get_pixel_size, get_origin_coordinates, get_image_dimension, xy_to_rowcol, open_raster , load_img_as_array

#load_img_as_array
gdal.UseExceptions()  # Active les exceptions GDAL

# personal libraries
#import classification as cla
#import read_and_write as rw
#import plots
#from my_function import report_from_dict_to_df


from osgeo import gdal
# Activer les exceptions GDAL
gdal.UseExceptions()
# Chemin de l'image
filename_mask = '/home/onyxia/work/results/data/img_pretraitees/masque_foret.tif'
#filename = '/home/onyxia/work/results/data/img_pretraitees/Serie_temp_S2_allbands.tif'
# Ouvrir l'image
data_set = open_raster(filename_mask)
if data_set is None:
    raise Exception("Impossible d'ouvrir l'image. Vérifiez le chemin du fichier.")

# Obtenir les dimensions de l'image
nb_lignes, nb_col, nb_band = get_image_dimension(data_set)

# Obtenir les coordonnées de l'origine
origin_x, origin_y = get_origin_coordinates(data_set)

# Obtenir la taille des pixels
psize_x, psize_y = get_pixel_size(data_set)

# Calculer les bornes de l'image
xmin, xmax, ymin, ymax = get_image_bounds(data_set, origin_x, origin_y, psize_x, psize_y, nb_col, nb_lignes)

# Obtenir le système de projection
projection = get_projection(data_set)

# Fermer l'image
data_set = None

##rasterisation
# Définir les chemins des fichiers
in_vector = '/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ.shp'
out_image = '/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ_rasterized.tif'

# Caractéristiques extraites de l'image de référence (masque_foret.tif)
sptial_resolution = 10.0  # Taille des pixels (10 mètres)
xmin = 501127.9697        # Borne minimale en X
ymin = 6240664.0236       # Borne minimale en Y
xmax = 609757.9697        # Borne maximale en X
ymax = 6314464.0236       # Borne maximale en Y
field_name = 'Code_Pixel'        # Champ contenant les étiquettes numériques des classes
projection = 'EPSG:2154'  # Système de projection Lambert-93 (EPSG:2154)

# define command pattern to fill with paremeters
cmd_pattern = ("gdal_rasterize -a {field_name} "
               "-tr {sptial_resolution} {sptial_resolution} "
               "-te {xmin} {ymin} {xmax} {ymax} -ot Byte -of GTiff "
               "{in_vector} {out_image}")

# fill the string with the parameter thanks to format function
cmd = cmd_pattern.format(in_vector=in_vector, xmin=xmin, ymin=ymin, xmax=xmax,
                         ymax=ymax, out_image=out_image, field_name=field_name,
                         sptial_resolution=sptial_resolution)

# Exécuter la commande
print("Exécution de la commande :")
print(cmd)
os.system(cmd)

print(f"Rasterisation terminée. Le fichier de sortie est : {out_image}")

# 1 --- parametres
id_filename = '/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ_rasterized.tif'
sample_filename = '/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ.shp'
image_filename = '/home/onyxia/work/results/data/img_pretraitees/Serie_temp_S2_allbands.tif'

# Sample parameters
nb_iter = 30  # Nombre de répétitions de la validation croisée
nb_folds = 5  # Nombre de folds
is_point = True  # Les échantillons sont des points (shapefile)
field_name = 'Code_Pixel'  # Champ contenant les étiquettes dans le shapefile

# 1 --- Paramètres de sortie
suffix = '_CV{}folds_stratified_group_x{}times'.format(nb_folds, nb_iter)

# Dossier de sortie
out_folder = '/home/onyxia/work/results'  # Modifier le dossier de sortie
os.makedirs(out_folder, exist_ok=True)  # Crée le dossier de sortie s'il n'existe pas

# Chemins de sortie corrigés pour l'image de classification, la matrice de confusion et les métriques
out_classif = os.path.join(out_folder, 'classif', 'carte_essences_echelle_pixel.tif')  # Renommer l'image de classification
out_matrix = os.path.join(out_folder, 'figure', '_matrice{}.png'.format(suffix))
out_qualite = os.path.join(out_folder, 'figure', '_qualites{}.png'.format(suffix))

# 2 --- extract samples
if not is_point:
    X, Y, t = get_samples_from_roi(image_filename, sample_filename)
    _, groups, _ = get_samples_from_roi(image_filename, id_filename)
else:
    # Extraire les coordonnées des points du shapefile
    gdf = gpd.read_file(sample_filename)
    list_row, list_col = get_row_col_from_file(sample_filename, image_filename)

    # Extraire les valeurs des pixels de l'image
    image = load_img_as_array(image_filename)
    X = image[(list_row, list_col)]  # Caractéristiques (bandes spectrales)

    # Extraire les étiquettes (classes) du shapefile
    Y = gdf.loc[:, field_name].values
    Y = Y.ravel()  # Convertit Y en un tableau 1D

    # Remplacer les classes "inconnu" par 0 et s'assurer que tout est numérique
    Y = np.array([0 if str(label).lower() == 'inconnu' else label for label in Y], dtype=np.int32)
    # Affichage des valeurs avant conversion
    print("Valeurs de Code_Pixel avant conversion :", np.unique(Y))

    # Extraire les groupes (identifiants des polygones)
    groups = gdf.loc[:, 'ID'].values  # le champ des groupes

# 3 --- Répéter la validation croisée 30 fois
list_cm = []
list_accuracy = []
list_report = []
groups = np.squeeze(groups)  # Assure que les groupes sont au format 1D

for _ in range(nb_iter):
    # Initialiser StratifiedGroupKFold
    kf = StratifiedGroupKFold(n_splits=nb_folds, shuffle=True, random_state=42)

    # Validation croisée
    for train, test in kf.split(X, Y, groups=groups):
        X_train, X_test = X[train], X[test]
        Y_train, Y_test = Y[train], Y[test]

        # 4 --- Entraînement du modèle
        clf = RandomForestClassifier(
            max_depth=50,           # Profondeur maximale des arbres
            oob_score=True,         # Active le score OOB
            max_samples=0.75,       # Utilise 75% des échantillons pour chaque arbre
            class_weight='balanced', # Ajuste les poids des classes
            random_state=42         # Pour la reproductibilité
        )
        clf.fit(X_train, Y_train)

        # 5 --- Test du modèle
        Y_predict = clf.predict(X_test)

        # Calcul des métriques de qualité
        list_cm.append(confusion_matrix(Y_test, Y_predict))
        list_accuracy.append(accuracy_score(Y_test, Y_predict))
        report = classification_report(Y_test, Y_predict, labels=np.unique(Y_predict), output_dict=True, zero_division=0)
        list_report.append(report_from_dict_to_df(report))

# 6 --- Calcul des moyennes et écarts-types
array_cm = np.array(list_cm)
mean_cm = array_cm.mean(axis=0)

array_accuracy = np.array(list_accuracy)
mean_accuracy = array_accuracy.mean()
std_accuracy = array_accuracy.std()

array_report = np.array(list_report)
mean_report = array_report.mean(axis=0)
std_report = array_report.std(axis=0)
a_report = list_report[0]
mean_df_report = pd.DataFrame(mean_report, index=a_report.index, columns=a_report.columns)
std_df_report = pd.DataFrame(std_report, index=a_report.index, columns=a_report.columns)

# 7 --- Affichage des résultats
# Matrice de confusion
plot_cm(mean_cm, np.unique(Y_predict))
plt.savefig(out_matrix, bbox_inches='tight')

# Métriques de qualité
fig, ax = plt.subplots(figsize=(10, 7))
ax = mean_df_report.T.plot.bar(ax=ax, yerr=std_df_report.T, zorder=2)
ax.set_ylim(0.5, 1)
_ = ax.text(1.5, 0.95, 'OA : {:.2f} +- {:.2f}'.format(mean_accuracy, std_accuracy), fontsize=14)
ax.set_title('Class quality estimation')

# Personnalisation du graphique
ax.set_facecolor('ivory')
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis='x', colors='darkslategrey', labelsize=14)
ax.tick_params(axis='y', colors='darkslategrey', labelsize=14)
ax.yaxis.grid(which='major', color='darkgoldenrod', linestyle='--', linewidth=0.5, zorder=1)
ax.yaxis.grid(which='minor', color='darkgoldenrod', linestyle='-.', linewidth=0.3, zorder=1)
plt.savefig(out_qualite, bbox_inches='tight')

# 8 --- Entraîner le modèle final sur l'ensemble des données
clf_final = RandomForestClassifier(
    max_depth=50,           # Profondeur maximale des arbres
    oob_score=True,         # Active le score OOB
    max_samples=0.75,       # Utilise 75% des échantillons pour chaque arbre
    class_weight='balanced', # Ajuste les poids des classes
    random_state=42         # Pour la reproductibilité
)
clf_final.fit(X, Y)  # Entraînement sur l'ensemble des données

# 9 --- Prédire les classes pour l'ensemble de l'image
# Charger l'image entière
image = load_img_as_array(image_filename)
rows, cols, bands = image.shape
image_reshaped = image.reshape(rows * cols, bands)  # Redimensionner pour la prédiction

# Prédire les classes
classification_result = clf_final.predict(image_reshaped)
classification_result = classification_result.astype(np.uint8)  # Convertir en uint8
classification_result = classification_result.reshape(rows, cols)  # Remettre en forme 2D

# 10 --- Enregistrer l'image de classification
# Obtenir les informations géoréférencées de l'image d'entrée
src_ds = gdal.Open(image_filename)
geo_transform = src_ds.GetGeoTransform()
projection = src_ds.GetProjection()

# Créer un fichier TIFF pour l'image classifiée
driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create(out_classif, cols, rows, 1, gdal.GDT_Byte)
out_ds.SetGeoTransform(geo_transform)
out_ds.SetProjection(projection)

# Écrire les données classifiées dans le fichier
out_band = out_ds.GetRasterBand(1)
out_band.WriteArray(classification_result)
out_band.SetNoDataValue(0)  # Définir une valeur NoData
out_band.FlushCache()

# Libérer les ressources
out_band = None
out_ds = None
src_ds = None

print("L'image de classification a été enregistrée avec succès dans :", out_classif)
