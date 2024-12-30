import geopandas as gpd
import os
from my_function import (
    plot_bar_polygons_per_class, 
    plot_bar_pixels_per_class, 
    plot_violin_pixels_per_polygon_by_class
)


# Chemins des fichiers
input_shapefile = '/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ.shp'
output_dir = '/home/onyxia/work/results/figure/'

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Dossier de sortie vérifié/créé : {output_dir}")
# Chargement des données
gdf = gpd.read_file(input_shapefile)

# Vérification et calcul de NB_PIX si manquant
if 'NB_PIX' not in gdf.columns:
    print("⚠️ Colonne 'NB_PIX' manquante. Calcul en cours...")
    if not gdf.crs.is_projected:
        raise ValueError("Le CRS doit être projeté (en mètres) pour calculer les surfaces correctement.")
    
    pixel_area = 100  # Aire d'un pixel en m² (10m x 10m)
    gdf['Area'] = gdf['geometry'].area  # Calcul de la surface
    gdf['NB_PIX'] = (gdf['Area'] / pixel_area).astype(int)  # Calcul du nombre de pixels
    print("✅ Colonne 'NB_PIX' ajoutée avec succès.")

 # Afficher un aperçu des données
print(gdf.head())

# Vérifier les valeurs uniques de Code_Pixel
print("Valeurs uniques de 'Code_Pixel':", gdf['Code_Pixel'].unique())

# Vérifier si NB_PIX contient des valeurs nulles ou négatives
print("Statistiques de 'NB_PIX':")
print(gdf['NB_PIX'].describe())
print("Valeurs nulles dans 'NB_PIX':", gdf['NB_PIX'].isnull().sum())
print("Valeurs négatives dans 'NB_PIX':", (gdf['NB_PIX'] < 0).sum())
   

# Vérification des colonnes nécessaires
required_columns = ['Code_Pixel', 'NB_PIX']
missing_columns = [col for col in required_columns if col not in gdf.columns]
if missing_columns:
    raise ValueError(f"Les colonnes manquantes sont : {missing_columns}. Vérifiez vos données.")
# Exclure les polygones avec la classe 'Inconnu'
gdf = gdf[gdf['Code_Pixel'] != 'Inconnu']
# Liste des classes valides (à ajuster selon vos données réelles)
classes_valides = ['11', '12', '13', '14', '21', '22', '23', '24', '25']

# Filtrer les polygones avec des classes valides
gdf = gdf[gdf['Code_Pixel'].isin(classes_valides)]

# Vérification après filtrage
print("✅ Données après exclusion des classes non valides :")
print(gdf['Code_Pixel'].unique())
print(gdf[['Code_Pixel', 'NB_PIX']].head())

# Génération des graphiques avec choix interactif (Plotly) ou statique (Matplotlib)
use_interactive = True  # Changez en False pour des graphiques statiques

# 1. Diagramme en bâtons : Nombre de polygones par classe
plot_bar_polygons_per_class(
    gdf, 
    f"{output_dir}diag_baton_nb_poly_by_class.{'html' if use_interactive else 'png'}", 
    interactive=use_interactive
)

# 2. Diagramme en bâtons : Nombre de pixels par classe
plot_bar_pixels_per_class(
    gdf, 
    f"{output_dir}diag_baton_nb_pix_by_class.{'html' if use_interactive else 'png'}", 
    interactive=use_interactive
)

# 3. Violin Plot : Distribution des pixels par polygone par classe
plot_violin_pixels_per_polygon_by_class(
    gdf, 
    f"{output_dir}violin_plot_nb_pix_by_poly_by_class.{'html' if use_interactive else 'png'}", 
    interactive=use_interactive
)

print("✅ Violin plot du nombre de pixels par polygone, par classe généré.")

print("🎯 Analyse terminée. Les graphiques sont disponibles dans le dossier 'results/figure/'.")
