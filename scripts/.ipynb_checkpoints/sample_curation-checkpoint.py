import geopandas as gpd
from my_function import filter_classes, clip_to_extent, save_vector_file

# Chemins des fichiers
input_shapefile = '/home/onyxia/work/data/project/FORMATION_VEGETALE.shp'
emprise_shapefile = '/home/onyxia/work/data/project/emprise_etude.shp'
output_shapefile = '/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ.shp'

# Chargement des fichiers
gdf = gpd.read_file(input_shapefile)
gdf_emprise = gpd.read_file(emprise_shapefile)

# Harmonisation des CRS
if gdf.crs != gdf_emprise.crs:
    gdf = gdf.to_crs(gdf_emprise.crs)

# Filtrage par emprise
gdf_filtered = gdf[gdf.intersects(gdf_emprise.union_all())].copy()

# Étape 2 : Clipage précis avec gpd.clip
gdf_clipped = gpd.clip(gdf_filtered, gdf_emprise)


# Mapping complet pour Classif Pixel
mapping_pixel = {
    # Classe 11 : Autres feuillus
    'FF1-49-49': {'Code_Pixel': 11, 'Nom_Pixel': 'Autres feuillus'},
    'FF1-09-09': {'Code_Pixel': 11, 'Nom_Pixel': 'Autres feuillus'},
    'FF1-10-10': {'Code_Pixel': 11, 'Nom_Pixel': 'Autres feuillus'},
    
    # Classe 12 : Chêne
    'FF1G01-01': {'Code_Pixel': 12, 'Nom_Pixel': 'Chêne'},
    
    # Classe 13 : Robinier
    'FF1-14-14': {'Code_Pixel': 13, 'Nom_Pixel': 'Robinier'},
    
    # Classe 14 : Peuplier
    'FP': {'Code_Pixel': 14, 'Nom_Pixel': 'Peuplier'},
    
    # Classe 15 : Mélange de feuillus
    'FF1-00-00': {'Code_Pixel': 15, 'Nom_Pixel': 'Mélange de feuillus'},
    
    # Classe 16 : Feuillus en îlots
    'FF1-00': {'Code_Pixel': 16, 'Nom_Pixel': 'Feuillus en îlots'},
    
    # Classe 21 : Autres conifères autre que pin
    'FF2G61-61': {'Code_Pixel': 21, 'Nom_Pixel': 'Autres conifères autre que pin'},
    'FF2-91-91': {'Code_Pixel': 21, 'Nom_Pixel': 'Autres conifères autre que pin'},
    'FF2-90-90': {'Code_Pixel': 21, 'Nom_Pixel': 'Autres conifères autre que pin'},
    'FF2-63-63': {'Code_Pixel': 21, 'Nom_Pixel': 'Autres conifères autre que pin'},
    
    # Classe 22 : Autres Pin
    'FF2-52-52': {'Code_Pixel': 22, 'Nom_Pixel': 'Autres Pin'},
    'FF2-80-80': {'Code_Pixel': 22, 'Nom_Pixel': 'Autres Pin'},
    'FF2-81-81': {'Code_Pixel': 22, 'Nom_Pixel': 'Autres Pin'},
    
    # Classe 23 : Douglas
    'FF2-64-64': {'Code_Pixel': 23, 'Nom_Pixel': 'Douglas'},
    
    # Classe 24 : Pin laricio ou pin noir
    'FF2G53-53': {'Code_Pixel': 24, 'Nom_Pixel': 'Pin laricio ou pin noir'},
    
    # Classe 25 : Pin maritime
    'FF2-51-51': {'Code_Pixel': 25, 'Nom_Pixel': 'Pin maritime'},
    
    # Classe 26 : Mélange conifères
    'FF2-00-00': {'Code_Pixel': 26, 'Nom_Pixel': 'Mélange conifères'},
    
    # Classe 27 : Conifères en îlots
    'FF2-00': {'Code_Pixel': 27, 'Nom_Pixel': 'Conifères en îlots'},
    
    # Classe 28 : Mélange conifères prépondérants et feuillus
    'FF32': {'Code_Pixel': 28, 'Nom_Pixel': 'Mélange de conifères prépondérants et feuillus'},
    
    # Classe 29 : Mélange de feuillus prépondérants et conifères
    'FF31': {'Code_Pixel': 29, 'Nom_Pixel': 'Mélange de feuillus prépondérants et conifères'},
}

# Mapping complet pour Classif Objet
mapping_objet = {
    # Classe 11 : Autres feuillus
    'FF1-49-49': {'Code_Objet': 11, 'Nom_Objet': 'Autres feuillus'},
    'FF1-09-09': {'Code_Objet': 11, 'Nom_Objet': 'Autres feuillus'},
    'FF1-10-10': {'Code_Objet': 11, 'Nom_Objet': 'Autres feuillus'},
    
    # Classe 12 : Chêne
    'FF1G01-01': {'Code_Objet': 12, 'Nom_Objet': 'Chêne'},
    
    # Classe 13 : Robinier
    'FF1-14-14': {'Code_Objet': 13, 'Nom_Objet': 'Robinier'},
    
    # Classe 14 : Peuplier
    'FP': {'Code_Objet': 14, 'Nom_Objet': 'Peuplier'},
    
    # Classe 15 : Mélange de feuillus
    'FF1-00-00': {'Code_Objet': 15, 'Nom_Objet': 'Mélange de feuillus'},
    
    # Classe 16 : Feuillus en îlots
    'FF1-00': {'Code_Objet': 16, 'Nom_Objet': 'Feuillus en îlots'},
    
    # Classe 21 : Autres conifères autre que pin
    'FF2G61-61': {'Code_Objet': 21, 'Nom_Objet': 'Autres conifères autre que pin'},
    'FF2-91-91': {'Code_Objet': 21, 'Nom_Objet': 'Autres conifères autre que pin'},
    'FF2-90-90': {'Code_Objet': 21, 'Nom_Objet': 'Autres conifères autre que pin'},
    'FF2-63-63': {'Code_Objet': 21, 'Nom_Objet': 'Autres conifères autre que pin'},
    
    # Classe 22 : Autres Pin
    'FF2-52-52': {'Code_Objet': 22, 'Nom_Objet': 'Autres Pin'},
    'FF2-80-80': {'Code_Objet': 22, 'Nom_Objet': 'Autres Pin'},
    'FF2-81-81': {'Code_Objet': 22, 'Nom_Objet': 'Autres Pin'},
    
    # Classe 23 : Douglas
    'FF2-64-64': {'Code_Objet': 23, 'Nom_Objet': 'Douglas'},
    
    # Classe 24 : Pin laricio ou pin noir
    'FF2G53-53': {'Code_Objet': 24, 'Nom_Objet': 'Pin laricio ou pin noir'},
    
    # Classe 25 : Pin maritime
    'FF2-51-51': {'Code_Objet': 25, 'Nom_Objet': 'Pin maritime'},
    
    # Classe 26 : Mélange conifères
    'FF2-00-00': {'Code_Objet': 26, 'Nom_Objet': 'Mélange conifères'},
    
    # Classe 27 : Conifères en îlots
    'FF2-00': {'Code_Objet': 27, 'Nom_Objet': 'Conifères en îlots'},
    
    # Classe 28 : Mélange conifères prépondérants et feuillus
    'FF32': {'Code_Objet': 28, 'Nom_Objet': 'Mélange de conifères prépondérants et feuillus'},
    
    # Classe 29 : Mélange de feuillus prépondérants et conifères
    'FF31': {'Code_Objet': 29, 'Nom_Objet': 'Mélange de feuillus prépondérants et conifères'},
}


# Ajout des champs Pixel et Objet
gdf_clipped['Code_Pixel'] = gdf_clipped['CODE_TFV'].apply(
    lambda x: mapping_pixel.get(str(x), {}).get('Code_Pixel', 'Inconnu')
)
gdf_clipped['Nom_Pixel'] = gdf_clipped['CODE_TFV'].apply(
    lambda x: mapping_pixel.get(str(x), {}).get('Nom_Pixel', 'Inconnu')
)
gdf_clipped['Code_Objet'] = gdf_clipped['CODE_TFV'].apply(
    lambda x: mapping_objet.get(str(x), {}).get('Code_Objet', 'Inconnu')
)
gdf_clipped['Nom_Objet'] = gdf_clipped['CODE_TFV'].apply(
    lambda x: mapping_objet.get(str(x), {}).get('Nom_Objet', 'Inconnu')
)

# Vérification des données après agrégation
print("🔍 Aperçu des données après ajout des champs :")
print(gdf_clipped[['CODE_TFV', 'TFV', 'Code_Pixel', 'Nom_Pixel', 'Code_Objet', 'Nom_Objet']].head())

# Vérification des valeurs uniques
print("\n📊 Valeurs uniques pour Code_Pixel :", gdf_clipped['Code_Pixel'].unique())
print("📊 Valeurs uniques pour Nom_Pixel :", gdf_clipped['Nom_Pixel'].unique())
print("📊 Valeurs uniques pour Code_Objet :", gdf_clipped['Code_Objet'].unique())
print("📊 Valeurs uniques pour Nom_Objet :", gdf_clipped['Nom_Objet'].unique())

output_path = '/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ.shp'
gdf_clipped.to_file(output_path, driver='ESRI Shapefile')

print(f"✅ Fichier sauvegardé avec succès : {output_path}")
print(f"📊 Nombre de polygones sauvegardés : {len(gdf_clipped)}")
