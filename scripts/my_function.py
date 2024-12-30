import os
from osgeo import gdal, ogr, osr
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

def validate_and_create_directory(path):
    """
    Valide et crée un répertoire s'il n'existe pas.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📂 Dossier créé : {path}")

def open_shapefile(shapefile_path):
    """
    Ouvre un fichier shapefile avec OGR.
    """
    ds = ogr.Open(shapefile_path)
    if ds is None:
        raise FileNotFoundError(f"Erreur : Impossible d'ouvrir le fichier {shapefile_path}.")
    return ds

def filter_forest_layer(layer):
    """
    Applique un filtre pour exclure certaines classes non forestières.
    """
    excluded_classes = [
        'Formation herbacée', 'Lande', 'Forêt fermée sans couvert arboré', 
        'Forêt ouverte sans couvert arboré'
    ]
    layer.SetAttributeFilter(
        "TFV NOT IN ('" + "', '".join(excluded_classes) + "')"
    )
    return layer

def create_raster_from_shapefile(output_path, emprise_layer, spatial_ref, resolution=10):
    """
    Crée un raster vide basé sur une emprise shapefile.
    """
    emprise_extent = emprise_layer.GetExtent()
    x_res = int((emprise_extent[1] - emprise_extent[0]) / resolution)
    y_res = int((emprise_extent[3] - emprise_extent[2]) / resolution)
    
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(
        output_path,
        x_res, y_res,
        1, gdal.GDT_Byte
    )
    
    if out_raster is None:
        raise RuntimeError(f"Erreur : Impossible de créer le fichier raster {output_path}.")
    
    out_raster.SetProjection(spatial_ref.ExportToWkt())
    out_raster.SetGeoTransform((
        emprise_extent[0], resolution, 0,
        emprise_extent[3], 0, -resolution
    ))
    
    return out_raster

def rasterize_layer(raster, layer):
    """
    Rasterise une couche vectorielle dans un raster.
    """
    gdal.RasterizeLayer(
        raster,
        [1],  # Bande 1
        layer,
        burn_values=[1]
    )
    band = raster.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.FlushCache()
    raster = None
    print("✅ Rasterisation terminée.")

# sample curation

import geopandas as gpd

def clip_to_extent(gdf, extent_gdf):
    """
    Découpe un GeoDataFrame avec une emprise spécifiée.
    """
    return gdf.clip(extent_gdf)

def filter_classes(gdf):
    """
    Filtre les classes en fonction de la Figure 2.
    """
    # Classes valides
    valid_classes = {
        'Autres feuillus': 11,
        'Chêne': 12,
        'Robinier': 13,
        'Peupleraie': 14,
        'Autres conifères autre que pin': 21,
        'Autres Pin': 22,
        'Douglas': 23,
        'Pin laricio ou pin noir': 24,
        'Pin maritime': 25,
        'Feuillus en îlots': 16,
        'Mélange conifères': 26,
        'Conifères en îlots': 27,
        'Mélange de conifères prépondérants et feuillus': 28,
        'Mélange de feuillus prépondérants et conifères': 29
    }
    
    # Filtrer les classes et ajouter les attributs 'Nom' et 'Code'
    gdf_filtered = gdf[gdf['TFV'].isin(valid_classes.values())].copy()
    gdf_filtered['Nom'] = gdf_filtered['TFV'].map({v: k for k, v in valid_classes.items()})
    gdf_filtered['Code'] = gdf_filtered['TFV']
    
    print(f"✅ {len(gdf_filtered)} polygones sélectionnés.")
    return gdf_filtered

def save_vector_file(gdf, output_path):
    """
    Sauvegarde un GeoDataFrame en tant que fichier vectoriel.
    """
    gdf.to_file(output_path, driver='ESRI Shapefile')
    print(f"💾 Fichier sauvegardé : {output_path}")

 # une analyse des échantillons sélectionné


def plot_bar_polygons_per_class(gdf, output_path, interactive=False):
    """ Crée un diagramme en bâtons du nombre de polygones par classe. """
    polygon_counts = gdf['Code_Pixel'].value_counts().reset_index()
    polygon_counts.columns = ['Classe', 'Nombre de polygones']
    
    if interactive:
        fig = px.bar(
            polygon_counts, 
            x='Classe', 
            y='Nombre de polygones',
            title='Nombre de polygones par classe',
            labels={'Nombre de polygones': 'Nombre de polygones', 'Classe': 'Classe'},
            template='plotly_dark'
        )
        fig.write_html(output_path)
    else:
        plt.figure(figsize=(12, 6))
        plt.bar(polygon_counts['Classe'], polygon_counts['Nombre de polygones'], color='skyblue')
        plt.title('Nombre de polygones par classe')
        plt.xlabel('Classe')
        plt.ylabel('Nombre de polygones')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def plot_bar_pixels_per_class(gdf, output_path, interactive=False):
    """ Crée un diagramme en bâtons du nombre de pixels par classe. """
    pixel_counts = gdf.groupby('Code_Pixel')['NB_PIX'].sum().reset_index()
    pixel_counts.columns = ['Classe', 'Nombre de pixels']
    
    if interactive:
        fig = px.bar(
            pixel_counts, 
            x='Classe', 
            y='Nombre de pixels',
            title='Nombre de pixels par classe',
            labels={'Nombre de pixels': 'Nombre de pixels', 'Classe': 'Classe'},
            template='plotly_dark'
        )
        fig.write_html(output_path)
    else:
        plt.figure(figsize=(12, 6))
        plt.bar(pixel_counts['Classe'], pixel_counts['Nombre de pixels'], color='lightcoral')
        plt.title('Nombre de pixels par classe')
        plt.xlabel('Classe')
        plt.ylabel('Nombre de pixels')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def plot_violin_pixels_per_polygon_by_class(gdf, output_path, interactive=False):
    """ Crée un Violin Plot pour la distribution du nombre de pixels par polygone, par classe. """
    if interactive:
        fig = px.violin(
            gdf, 
            x='Code_Pixel', 
            y='NB_PIX', 
            box=True, 
            points='all',
            title='Distribution du nombre de pixels par polygone, par classe',
            labels={'NB_PIX': 'Nombre de pixels', 'Code_Pixel': 'Classe'},
            template='plotly_dark'
        )
        fig.write_html(output_path)
    else:
        plt.figure(figsize=(14, 8))
        classes = gdf['Code_Pixel'].unique()
        for cls in classes:
            subset = gdf[gdf['Code_Pixel'] == cls]
            plt.violinplot(subset['NB_PIX'], positions=[list(classes).index(cls)], showmeans=True)
        
        plt.title('Distribution du nombre de pixels par polygone, par classe')
        plt.xlabel('Classe')
        plt.ylabel('Nombre de pixels par polygone')
        plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

