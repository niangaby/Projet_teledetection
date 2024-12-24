import os
from my_function import (
    validate_and_create_directory,
    open_shapefile,
    filter_forest_layer,
    create_raster_from_shapefile,
    rasterize_layer
)

def build_forest_mask(formation_shp, emprise_shp, output_mask):
    """
    Crée un masque raster pour les zones de forêt.
    """
    # ✅ Valider et créer le dossier de sortie
    output_dir = os.path.dirname(output_mask)
    validate_and_create_directory(output_dir)
    
    # ✅ Ouvrir le shapefile Formation_vegetale et filtrer
    formation_ds = open_shapefile(formation_shp)
    formation_layer = formation_ds.GetLayer()
    formation_layer = filter_forest_layer(formation_layer)
    
    # ✅ Ouvrir le shapefile emprise_etude
    emprise_ds = open_shapefile(emprise_shp)
    emprise_layer = emprise_ds.GetLayer()
    spatial_ref = emprise_layer.GetSpatialRef()
    
    if spatial_ref is None:
        raise ValueError("Erreur : Impossible d'obtenir la projection depuis emprise_etude.shp.")
    
    # ✅ Créer un raster vide
    out_raster = create_raster_from_shapefile(output_mask, emprise_layer, spatial_ref)
    
    # ✅ Rasteriser la couche filtrée
    rasterize_layer(out_raster, formation_layer)
    
    print(f"✅ Masque forêt créé : {output_mask}")

# ✅ Chemins des fichiers
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
formation_shp = os.path.join(BASE_DIR, "data", "project", "FORMATION_VEGETALE.shp")
emprise_shp = os.path.join(BASE_DIR, "data", "project", "emprise_etude.shp")
output_mask = os.path.join(BASE_DIR, "results", "data", "img_pretraitees", "masque_foret.tif")

# ✅ Appel de la fonction
if __name__ == "__main__":
    build_forest_mask(formation_shp, emprise_shp, output_mask)
