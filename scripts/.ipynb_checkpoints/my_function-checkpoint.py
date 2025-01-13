import os
from osgeo import gdal, ogr, osr
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import rasterio
import numpy as np
#Validation et création des dossiers de sortie
def validate_and_create_directory(path):
    """
    Valide et crée un répertoire s'il n'existe pas.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📂 Dossier créé : {path}")
    else:  # 
        print(f"✅ Dossier déjà existant : {path}")  

        
#Vérification des systèmes de coordonnées
def check_and_reproject_layer(layer, target_srs):
    """
    Vérifie si la couche est dans le même système de coordonnées que celui cible.
    Si ce n'est pas le cas, reprojette la couche.
    """
    source_srs = layer.GetSpatialRef()
    if not source_srs.IsSame(target_srs):
        print("🔄 Reprojection en cours...")
        coord_trans = osr.CoordinateTransformation(source_srs, target_srs)
        for feature in layer:
            geom = feature.GetGeometryRef()
            geom.Transform(coord_trans)
        print("✅ Reprojection terminée.")
    else:
        print("✅ Les systèmes de coordonnées correspondent.")


def open_shapefile(shapefile_path):
    """
    Ouvre un fichier shapefile avec OGR.
    """
    ds = ogr.Open(shapefile_path)
    if ds is None:
        raise FileNotFoundError(f"Erreur : Impossible d'ouvrir le fichier {shapefile_path}.")
    return ds
    
#Filtrage des formations
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
    # Vérification du nombre d'entités après filtrage
    print(f"✅ Nombre de polygones après filtrage : {layer.GetFeatureCount()}")
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

#calcul du NDVI et l'application du masque.

def calculer_ndvi(image_red, image_nir):
    # Calcul du NDVI : (NIR - Rouge) / (NIR + Rouge)
    return (image_nir - image_red) / (image_nir + image_red)

def appliquer_mask(input_image, mask_image, output_image, nodata_value=-9999):
    with rasterio.open(input_image) as src:
        image_data = src.read(1)
        nodata_value = src.nodata
    
    with rasterio.open(mask_image) as mask:
        mask_data = mask.read(1)
    
    # Appliquer le masque (valeurs 0 pour non forêt)
    image_data[mask_data == 0] = nodata_value

    # Sauvegarder l'image masquée
    with rasterio.open(output_image, 'w', driver='GTiff', count=1, dtype=image_data.dtype,
                       width=image_data.shape[1], height=image_data.shape[0], crs=src.crs, transform=src.transform,
                       nodata=nodata_value) as dst:
        dst.write(image_data, 1)
    
    print(f"L'image masquée a été sauvegardée sous {output_image}")

def pre_traiter_ndvi(input_folder, output_path, shapefile_path, mask_path, resolution=10):
    images = [f for f in os.listdir(input_folder) if f.endswith(".tif")]
    
    ndvi_stack = []
    
    for image in images:
        image_path = os.path.join(input_folder, image)
        
        with rasterio.open(image_path) as src:
            # Lire les bandes nécessaires (bandes 4 et 8 pour calculer le NDVI)
            band_red = src.read(4)
            band_nir = src.read(8)

            # Calculer le NDVI
            ndvi = calculer_ndvi(band_red, band_nir)

            # Appliquer le masque de forêt
            output_ndvi_path = os.path.join(output_path, f"NDVI_{image}")
            appliquer_mask(ndvi, mask_path, output_ndvi_path)
            ndvi_stack.append(output_ndvi_path)
    
    print("Série temporelle NDVI produite et masquée.")

# Exemple d'appel pour calculer NDVI
input_folder = "input_images"  # dossier contenant les images
output_path = "results/data/img_pretaitees"
shapefile_path = "emprise_etude.shp"
mask_path = "masque_foret.tif"
pre_traiter_ndvi(input_folder, output_path, shapefile_path, mask_path)


#pour pretraitemnt
#Découper l'image selon l'emprise (clipping)
def clip_image(input_image, shapefile_path, output_image):
    """
    Découpe une image raster en fonction d'un shapefile.
    """
    clip_image_to_extent(input_image, shapefile_path, output_image)
    print(f"✅ Image découpée et sauvegardée sous : {output_image}")
#Reprojection des images 

def reproject_image(input_image, output_image, target_crs):
    """
    Reprojette une image raster dans un système de coordonnées cible.
    """
    with rasterio.open(input_image) as src:
        transform, width, height = calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        with rasterio.open(output_image, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
        print(f"✅ Image projetée et sauvegardée sous : {output_image}")

#2
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling


def clip_image(image_path, shapefile_path, output_path):
    """
    Découpe une image raster selon l'emprise définie par un shapefile.
    """
    with rasterio.open(image_path) as src:
        shapefile = gpd.read_file(shapefile_path)
        shapefile = shapefile.to_crs(src.crs)  # Aligner le shapefile au système de coordonnées de l'image
        shapes = [feature.geometry for _, feature in shapefile.iterrows()]
        out_image, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)


def resample_image(input_image, output_image, target_resolution=10):
    """
    Rééchantillonne une image raster à une résolution cible.
    """
    with rasterio.open(input_image) as src:
        transform, width, height = calculate_default_transform(
            src.crs, src.crs, src.width, src.height, *src.bounds, resolution=target_resolution)
        profile = src.profile
        profile.update(transform=transform, width=width, height=height, dtype=rasterio.float32)

        with rasterio.open(output_image, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest)


def merge_bands_to_multispectral(input_bands, output_image):
    """
    Fusionne plusieurs bandes raster en une seule image multispectrale.
    """
    with rasterio.open(input_bands[0]) as src:
        profile = src.profile
        profile.update(count=len(input_bands))

        with rasterio.open(output_image, 'w', **profile) as dst:
            for i, band_path in enumerate(input_bands, start=1):
                with rasterio.open(band_path) as band:
                    dst.write(band.read(1), i)


def apply_mask(input_image, mask_image, output_image):
    """
    Applique un masque (par exemple de forêt) à une image raster.
    """
    with rasterio.open(input_image) as src:
        image_data = src.read()
        nodata_value = src.nodata

    with rasterio.open(mask_image) as mask_src:
        mask_data = mask_src.read(1)

    # Appliquer le masque (les zones non désirées deviennent NoData)
    image_data[:, mask_data == 0] = nodata_value

    with rasterio.open(output_image, 'w', driver='GTiff', count=image_data.shape[0], dtype=image_data.dtype,
                       width=image_data.shape[2], height=image_data.shape[1], crs=src.crs, transform=src.transform,
                       nodata=nodata_value) as dst:
        dst.write(image_data)





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
