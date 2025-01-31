import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
import os

import numpy as np
import matplotlib.pyplot as plt
import os
from my_function import load_raster_sign, rasterize_shapefile_sign

#  Dossier contenant le fichier NDVI 
ndvi_path = "/home/onyxia/work/results/data/img_pretraitees/Serie_temp_S2_ndvi.tif"
shapefile_path = "/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ.shp"  #  shapefile contenant les classes
field_name = "Code_Pixel"  # Le champ qui définit les classes dans le shapefile

# Classes à analyser
classes_to_analyze = [12, 13, 14, 23, 24, 25]  # Codes des classes en gras

# Mappage des codes aux noms des classes (correction du nom de la classe)
class_names = {
    12: "Chêne",
    13: "Robinier",
    14: "Peupleraie",
    23: "Douglas",
    24: "Pin lariceo ou pin noir",  
    25: "Pin maritime"
}

# Charger le raster NDVI
ndvi_data, ndvi_nodata = load_raster_sign(ndvi_path)

# Rasteriser le shapefile pour obtenir les classes
rasterized_classes = rasterize_shapefile_sign(shapefile_path, ndvi_path, field_name)

# Calcul des moyennes et écarts-types par classe
class_means = {}
class_stddevs = {}

for classe in classes_to_analyze:
    # Obtenir les indices correspondant à la classe
    mask = rasterized_classes == classe
    values_ndvi = ndvi_data[mask]
    values_ndvi = values_ndvi[values_ndvi != ndvi_nodata]  # Exclure les valeurs no-data
    
    if len(values_ndvi) > 0:
        class_means[classe] = np.mean(values_ndvi)
        class_stddevs[classe] = np.std(values_ndvi)

# Tracer les moyennes et écarts-types par classe
fig, ax1 = plt.subplots(figsize=(12, 8))  # Augmentation de la taille du graphique pour une meilleure lisibilité

# Tracer les barres avec les noms des classes
classes = list(class_means.keys())
means = list(class_means.values())
stddevs = list(class_stddevs.values())

# Tracer les barres avec des barres d'erreur pour les écarts-types
bars = ax1.bar(classes, means, yerr=stddevs, capsize=10, color='skyblue', edgecolor='black', alpha=0.7)

# Ajouter les noms des classes en dessous des barres
ax1.set_xticks(classes)
ax1.set_xticklabels([class_names[classe] for classe in classes], rotation=45, ha="right", fontsize=12)

# Ajouter des étiquettes pour les moyennes et écarts-types
for i, classe in enumerate(classes):
    ax1.text(classes[i], means[i] + 0.01, f"{means[i]:.2f}", ha='center', va='bottom', fontsize=10)
    ax1.text(classes[i], means[i] - 0.02, f"± {stddevs[i]:.2f}", ha='center', va='top', fontsize=10)

ax1.set_xlabel('Classes', fontsize=14)
ax1.set_ylabel('Moyenne NDVI', fontsize=14)
ax1.set_title('Moyenne et écart-type du NDVI par classe', fontsize=16)

# Créer un second axe pour les valeurs de NDVI à droite
ax2 = ax1.twinx()
ax2.set_ylabel('Moyenne NDVI', fontsize=14)  
ax2.set_yticks(ax1.get_yticks())  
ax2.set_ylim(ax1.get_ylim())  

# Ajuster les marges pour que tout soit bien visible
plt.tight_layout()

# Sauvegarder le graphique dans le dossier 'results/figure'
output_path = "/home/onyxia/work/results/figure/temp_mean_ndvi.png"
plt.savefig(output_path)

# Afficher le graphique
plt.show()

# Afficher le message de confirmation
print(f"Le graphique a été enregistré avec succès dans le chemin : {output_path}")

