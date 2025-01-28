import os
from osgeo import gdal, ogr, osr
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
import itertools  
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





#
#PRE - traitement 

from osgeo import gdal
import os
import numpy as np

def reproject_image(input_path, dst_crs, resolution, nodata_value=0, nodata_value_ndvi = -9999, clip_shapefile=None):
    """ Reprojects an image to a specified CRS and resolution, ensuring alignment. """
    if not os.path.exists(input_path):
        print(f"Erreur : Le fichier {input_path} n'existe pas.")
        return None

    warp_options = gdal.WarpOptions(
        dstSRS=dst_crs,
        xRes=resolution,
        yRes=resolution,
        dstNodata=nodata_value,
        cutlineDSName=clip_shapefile,
        cropToCutline=True,
        resampleAlg=gdal.GRA_Bilinear,
        format="MEM"
    )

    ds = gdal.Warp('', input_path, options=warp_options)
    if ds is None:
        print(f"Erreur : La reprojection a échoué pour {input_path}")
    return ds
def align_and_mask(input_path, mask_path, dst_crs, resolution, nodata_value, clip_shapefile):
    """Aligne une image avec un masque et applique ce masque pour exclure les zones non-forestières.
    Parameters:
        input_path (str): Chemin de l'image d'entrée.
        mask_path (str): Chemin du fichier de masque.
        dst_crs (str): Système de coordonnées cible (par ex. 'EPSG:2154').
        resolution (float): Résolution de la sortie en mètres.
        nodata_value (float): Valeur de NoData pour l'image de sortie.
        clip_shapefile (str): Chemin d'un shapefile pour découper l'image (optionnel).
    Returns:
        gdal.Dataset: Dataset aligné et masqué, ou None en cas d'erreur.
    """
    try:
        # Reprojection et découpage de l'image d'entrée
        print(f"Reprojection et découpage de l'image : {input_path}")
        warp_options = gdal.WarpOptions(
            dstSRS=dst_crs,
            xRes=resolution,
            yRes=resolution,
            dstNodata=nodata_value,
            cutlineDSName=clip_shapefile,
            cropToCutline=True,
            resampleAlg=gdal.GRA_Bilinear,
            format="MEM"
        )
        input_ds = gdal.Warp('', input_path, options=warp_options)

        if input_ds is None:
            print(f"Erreur : Impossible de reprojeter l'image {input_path}")
            return None

        # Reprojection et découpage du masque
        print(f"Reprojection et découpage du masque : {mask_path}")
        mask_ds = gdal.Warp('', mask_path, options=warp_options)
        if mask_ds is None:
            print(f"Erreur : Impossible de reprojeter le masque {mask_path}")
            return None

        # Vérification des dimensions
        if (input_ds.RasterXSize != mask_ds.RasterXSize) or (input_ds.RasterYSize != mask_ds.RasterYSize):
            print("Erreur : Dimensions du masque et de l'image d'entrée incompatibles.")
            print(f"Dimensions image : {input_ds.RasterXSize}x{input_ds.RasterYSize}")
            print(f"Dimensions masque : {mask_ds.RasterXSize}x{mask_ds.RasterYSize}")
            return None

        # Lecture des données
        input_band = input_ds.GetRasterBand(1).ReadAsArray()
        mask_band = mask_ds.GetRasterBand(1).ReadAsArray()

        # Définir la valeur NoData du masque
        mask_band = np.where(mask_band == nodata_value, 0, mask_band)

        # Appliquer le masque : exclure les zones non forestières (mask_band == 0)
        masked_data = np.where(mask_band == 0, nodata_value, input_band)

        # Créer le dataset de sortie en mémoire
        driver = gdal.GetDriverByName("MEM")
        output_ds = driver.Create(
            '', input_ds.RasterXSize, input_ds.RasterYSize, 1, gdal.GDT_Float32
        )
        output_ds.SetGeoTransform(input_ds.GetGeoTransform())
        output_ds.SetProjection(input_ds.GetProjection())
        out_band = output_ds.GetRasterBand(1)
        out_band.WriteArray(masked_data)
        out_band.SetNoDataValue(nodata_value)

        print("Masque appliqué avec succès.")
        return output_ds

    except Exception as e:
        print(f"Erreur lors de l'application du masque : {e}")
        return None
def apply_mask(input_dataset, mask_path):
    """Applique un masque à un dataset en mémoire et retourne le dataset masqué."""
    try:
        mask_ds = gdal.Open(mask_path)
        if mask_ds is None:
            print(f"Erreur : Impossible de charger le masque {mask_path}")
            return None

        # Ré-alignement si nécessaire
        if (
            input_dataset.RasterXSize != mask_ds.RasterXSize or
            input_dataset.RasterYSize != mask_ds.RasterYSize
        ):
            print(f"Dimensions incompatibles, réalignement...")
            warp_options = gdal.WarpOptions(
                format="MEM",
                width=input_dataset.RasterXSize,
                height=input_dataset.RasterYSize,
                resampleAlg=gdal.GRA_NearestNeighbour
            )
            mask_ds = gdal.Warp('', mask_ds, options=warp_options)

        # Lecture des données
        input_band = input_dataset.GetRasterBand(1)
        mask_band = mask_ds.GetRasterBand(1)
        input_data = input_band.ReadAsArray()
        mask_data = mask_band.ReadAsArray()

        # Vérifiez à nouveau les dimensions après réalignement
        if input_data.shape != mask_data.shape:
            print(f"Erreur : Dimensions toujours incompatibles après réalignement.")
            return None

        # Appliquer le masque
        masked_data = input_data.copy()
        masked_data[mask_data == 0] = 0

        # Créer un dataset masqué
        driver = gdal.GetDriverByName("MEM")
        masked_ds = driver.Create(
            "", input_dataset.RasterXSize, input_dataset.RasterYSize, 1, gdal.GDT_Float32
        )
        masked_ds.SetGeoTransform(input_dataset.GetGeoTransform())
        masked_ds.SetProjection(input_dataset.GetProjection())
        masked_band = masked_ds.GetRasterBand(1)
        masked_band.WriteArray(masked_data)
        masked_band.SetNoDataValue(0)

        return masked_ds

    except Exception as e:
        print(f"Erreur lors de l'application du masque : {str(e)}")
        return None

def merge_images(input_paths, output_path, dst_crs, resolution, nodata_value=0, clip_shapefile=None):
    """Merges multiple images into a single mosaic with the desired CRS, resolution, and study area defined by a shapefile.
    
    Parameters:
        input_paths (list): List of file paths to input images.
        output_path (str): Path to save the merged image.
        dst_crs (str): Destination coordinate reference system (e.g., 'EPSG:2154').
        resolution (float): Desired resolution in target CRS units (e.g., meters).
        nodata_value (int/float): NoData value to assign to the output mosaic (default is 0).
        clip_shapefile (str): Path to the shapefile used for clipping (optional).
    """
    print("Fusion des images en cours...")

    warp_options = gdal.WarpOptions(
        dstSRS=dst_crs,       # Système de coordonnées cible
        xRes=resolution,      # Résolution en X
        yRes=resolution,      # Résolution en Y
        dstNodata=nodata_value,  # Appliquer une valeur de NoData
        cutlineDSName=clip_shapefile,  # Utiliser le shapefile pour découper (optionnel)
        cropToCutline=True,   # Activer le découpage à l'emprise
        resampleAlg=gdal.GRA_Bilinear,  # Méthode de rééchantillonnage
        format="GTiff"        # Format de sortie
    )

    try:
        # Utilisation de gdal.Warp pour effectuer la mosaïque
        gdal.Warp(output_path, input_paths, options=warp_options)

        # Vérification du fichier de sortie
        ds = gdal.Open(output_path)
        if ds is None:
            print(f"Erreur : La fusion a échoué pour les images : {input_paths}")
        else:
            print(f"Image fusionnée avec succès : {output_path}")
            print(f"Dimensions de l'image : X = {ds.RasterXSize}, Y = {ds.RasterYSize}")

    except Exception as e:
        print(f"Erreur lors de la fusion : {str(e)}")


        
#classif pixel
def report_from_dict_to_df(report):
    """
    Convertit un rapport de classification (dictionnaire) en DataFrame pandas.
    """
    # Supprimer la ligne 'accuracy' si elle existe
    if 'accuracy' in report:
        report.pop('accuracy')
    
    # Convertir le dictionnaire en DataFrame
    df = pd.DataFrame(report).transpose()
    return df
    
def open_image(filename, verbose=False):
  data_set = gdal.Open(filename, gdal.GA_ReadOnly)

  if data_set is None:
      print('Impossible to open {}'.format(filename))
  elif data_set is not None and verbose:
      print('{} is open'.format(filename))

  return data_set
    
def convert_data_type_from_gdal_to_numpy(gdal_data_type):
    
    if gdal_data_type == 'Byte':
        numpy_data_type = 'uint8'
    else:
        numpy_data_type = gdal_data_type.lower()
    return numpy_data_type


def load_img_as_array(filename):

    # Get size of output array
    data_set = open_image(filename)
    nb_lignes, nb_col, nb_band = get_image_dimension(data_set)

    # Get data type
    band = data_set.GetRasterBand(1)
    gdal_data_type = gdal.GetDataTypeName(band.DataType)
    numpy_data_type = convert_data_type_from_gdal_to_numpy(gdal_data_type)

    # Initialize an empty array
    array = np.empty((nb_lignes, nb_col, nb_band), dtype=numpy_data_type)

    # Fill the array
    for idx_band in range(nb_band):
        idx_band_gdal = idx_band + 1
        array[:, :, idx_band] = data_set.GetRasterBand(idx_band_gdal).ReadAsArray()

    # close data_set
    data_set = None
    band = None
    return array

            
    
def plot_cm(cm, labels, out_filename=None,
            normalize=False,  cmap='Greens'):
    """
    Plot du confusion matrix avec precision, recall et F1-score metrics.
    ----------
    """
    # Calculate precision, recall, and F1 score
    precision = cm.diagonal() / cm.sum(axis=0) * 100
    recall = cm.diagonal() / cm.sum(axis=1) * 100
    # Éviter les divisions par zéro dans le calcul du F1-score
    f1_score = np.where(
        (precision + recall) == 0,
        0,  # F1-score à 0 si precision + recall == 0
        2 * precision * recall / (precision + recall)
    )
    # Class names and values
    class_values = list(range(len(labels)))
    dic = dict(zip(class_values, labels))
    
    # Normalization and limits
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        max_value = 100
        name = "cm_normalized"
    else:
        max_value = cm.max()
        name = "cm"
    
    # Setup figure and gridspec
    n_classes = len(labels)
    # fig_width = 3 * n_classes  # Width of the figure
    # fig_height = fig_width / 2  # Fixed height, can be adjusted as needed
    fig = plt.figure(figsize=(15, 6))
    
    # Create gridspec for subplots
    gs = fig.add_gridspec(1, 3, width_ratios=[n_classes, 3, 0.2], wspace=0.2)    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cbar_ax_percent = fig.add_subplot(gs[0, 2])
    
    # Plot confusion matrix
    cax1 = ax1.matshow(cm, vmin=0, vmax=max_value, cmap=cmap, alpha=0.75)
    
    # Plot precision and recall metrics
    metrics = np.zeros((cm.shape[0], 3))
    metrics[:, 0] = precision
    metrics[:, 1] = recall
    metrics[:, 2] = f1_score
    
    ax2.set_aspect(2 / n_classes)
    cax2 = ax2.matshow(metrics, vmin=0, vmax=100, cmap=cmap, alpha=0.75)
    
    # Colorbars and labels
    if normalize:
        cbar_percent = fig.colorbar(cax2, cax=cbar_ax_percent, orientation='vertical')
        cbar_percent.set_label('Score (%)', rotation=90, labelpad=5)
        cbar_percent.set_ticks(np.linspace(0, 100, 5))
    else:
        cbar_ax_count = cbar_ax_percent.twinx()
        cbar_percent = fig.colorbar(cax2, cax=cbar_ax_percent, orientation='vertical')
        cbar_percent.set_ticks(np.linspace(0, 100, 5))
        
        # Colorbar for counts scale (on the left)
        cbar_count = fig.colorbar(cax1, cax=cbar_ax_count, orientation='vertical')
        
        fig.canvas.draw() 
        cbar_width = cbar_percent.ax.get_window_extent().width

        # Dynamically adjust labelpad based on the colorbar width
        cbar_position = cbar_percent.ax.get_position()
        cbar_count.set_label('Pixel count', rotation=90, labelpad=-30 - cbar_position.x0 - cbar_width)
        
        cbar_percent.set_label('Score (%)', rotation=90, labelpad=30 + cbar_position.x1 - cbar_width)

        cbar_count.set_ticks(np.arange(0, max_value + 1, max_value // 4))  # Ensure max_value is an integer
        cbar_ax_count.yaxis.set_ticks_position('left')

    # Confusion matrix labels and values
    tick_marks = np.arange(n_classes)
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(labels, rotation=50)
    ax1.set_yticklabels(labels)
    
    fmt = '.1f' if normalize or isinstance(cm[0,0], float) else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax1.text(j, i, f"{cm[i, j]:{fmt}}",
                 horizontalalignment="center", color="black")

    # Precision-Recall labels and values
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['Precision', 'Recall', 'F1_score'], rotation=50)
    ax2.set_yticks([])
    
    for i in range(n_classes):
        ax2.text(0, i, f'{precision[i]:.1f}', horizontalalignment="center", color="black")
        ax2.text(1, i, f'{recall[i]:.1f}', horizontalalignment="center", color="black")
        ax2.text(2, i, f'{f1_score[i]:.1f}', horizontalalignment="center", color="black")

    ax1.set_ylabel('True labels', fontweight='bold', fontsize=14, labelpad=10)
    ax1.set_xlabel('Predicted labels', fontweight='bold', fontsize=14, labelpad=10)

    # Customize tick parameters
    for ax in [ax1, ax2]:
        ax.tick_params(bottom=False, right=False)
        ax.tick_params(left=True, top=True, length=5)
        ax.xaxis.tick_top()
    if out_filename:
        plt.savefig(out_filename, bbox_inches='tight', dpi=300)


def get_xy_from_file(point_file):
    """
    Extrait les coordonnées x et y d'un fichier shapefile.
    Gère les points et les polygones (en utilisant les centroïdes).
    """
    gdf = gpd.read_file(point_file)
    
    # Vérifier le type de géométrie
    if gdf.geometry.type[0] == 'Point':
        list_x = gdf.geometry.x.values
        list_y = gdf.geometry.y.values
    elif gdf.geometry.type[0] in ['Polygon', 'MultiPolygon']:
        # Extraire les centroïdes des polygones
        centroids = gdf.geometry.centroid
        list_x = centroids.x.values
        list_y = centroids.y.values
    else:
        raise ValueError("Le shapefile doit contenir des points ou des polygones.")
    
    return list_x, list_y
def open_raster(filename):
    data_set = gdal.Open(filename, gdal.GA_ReadOnly)
    if data_set is None:
        print('Impossible to open {}'.format(filename))
    else:
        print('{} is open'.format(filename))
    return data_set

def get_image_dimension(data_set):
    nb_col = data_set.RasterXSize
    nb_lignes = data_set.RasterYSize
    nb_band = data_set.RasterCount
    return nb_lignes, nb_col, nb_band

def get_origin_coordinates(data_set):
    geotransform = data_set.GetGeoTransform()
    origin_x, origin_y = geotransform[0], geotransform[3]
    return origin_x, origin_y

def get_pixel_size(data_set):
    geotransform = data_set.GetGeoTransform()
    psize_x, psize_y = geotransform[1], geotransform[5]
    return psize_x, psize_y

def get_image_bounds(data_set, origin_x, origin_y, psize_x, psize_y, nb_col, nb_lignes):
    """
    Calcule les bornes de l'image (xmin, xmax, ymin, ymax).
    """
    xmin = origin_x
    xmax = origin_x + nb_col * psize_x
    ymin = origin_y + nb_lignes * psize_y  # psize_y est négatif
    ymax = origin_y
    print(f"Bornes de l'image : xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
    return xmin, xmax, ymin, ymax

def get_projection(data_set):
    """
    Récupère le système de projection de l'image.
    """
    projection = data_set.GetProjection()
    print(f"Système de projection : {projection}")
    return projection

def get_samples_from_roi(raster_name, roi_name, value_to_extract=None,
                         bands=None, output_fmt='full_matrix'):
    # Get size of output array
    raster = rw.open_image(raster_name)
    nb_col, nb_row, nb_band = rw.get_image_dimension(raster)

    # Get data type
    band = raster.GetRasterBand(1)
    gdal_data_type = gdal.GetDataTypeName(band.DataType)
    numpy_data_type = rw.convert_data_type_from_gdal_to_numpy(gdal_data_type)

    # Check if is roi is raster or vector dataset
    roi = open_image(roi_name)

    if (raster.RasterXSize != roi.RasterXSize) or \
            (raster.RasterYSize != roi.RasterYSize):
        print('Images should be of the same size')
        print('Raster : {}'.format(raster_name))
        print('Roi : {}'.format(roi_name))
        exit()

    if not bands:
        bands = list(range(nb_band))
    else:
        nb_band = len(bands)

    #  Initialize the output
    ROI = roi.GetRasterBand(1).ReadAsArray()
    if value_to_extract:
        t = np.where(ROI == value_to_extract)
    else:
        t = np.nonzero(ROI)  # coord of where the samples are different than 0

    Y = ROI[t].reshape((t[0].shape[0], 1)).astype('int32')

    del ROI
    roi = None  # Close the roi file

    try:
        X = np.empty((t[0].shape[0], nb_band), dtype=numpy_data_type)
    except MemoryError:
        print('Impossible to allocate memory: roi too large')
        exit()

    # Load the data
    for i in bands:
        temp = raster.GetRasterBand(i + 1).ReadAsArray()
        X[:, i] = temp[t]
        del temp
    raster = None  # Close the raster file

    # Store data in a dictionnaries if indicated
    if output_fmt == 'by_label':
        labels = np.unique(Y)
        dict_X = {}
        dict_t = {}
        for lab in labels:
            coord = np.where(Y == lab)[0]
            dict_X[lab] = X[coord]
            dict_t[lab] = (t[0][coord], t[1][coord])

        return dict_X, Y, dict_t
    else:
        return X, Y, t,

def xy_to_rowcol(x, y, image_filename):
    # get image infos
    data_set = open_image(image_filename)
    origin_x, origin_y = get_origin_coordinates(data_set)
    psize_x, psize_y = get_pixel_size(data_set)

    # convert x y to row col
    col = int((x - origin_x) / psize_x)
    row = - int((origin_y - y) / psize_y)

    return row, col


def get_row_col_from_file(point_file, image_file):
    
    list_row = []
    list_col = []
    list_x, list_y = get_xy_from_file(point_file)
    for x, y in zip(list_x, list_y):
        row, col = xy_to_rowcol(x, y, image_file)
        list_row.append(row)
        list_col.append(col)
    return list_row, list_col

# Dictionnaire de correspondance entre codes et noms des classes
CODE_TO_NAME = {
    0: "Inconnu",
    11: "Autres feuillus",
    12: "Chêne",
    13: "Robinier",
    14: "Peupleraie",
    15: "Mélange de feuillus",
    16: "Feuillus en îlots",
    21: "Autres conifères autre que pin",
    22: "Autres Pin",
    23: "Douglas",
    24: "Pin laricio ou pin noir",
    25: "Pin maritime",
    26: "Mélange de conifères",
    27: "Conifères en îlots",
    28: "Mélange de conifères prépondérants et feuillus",
    29: "Mélange de feuillus prépondérants et conifères"
}

def calculate_polygon_surface(polygon_gdf):
    """
    Calcule la surface des polygones en hectares.
    """
    if polygon_gdf.crs.to_epsg() != 2154:
        raise ValueError("Le GeoDataFrame doit être dans le système de coordonnées EPSG:2154.")
    polygon_gdf['surface_ha'] = polygon_gdf.geometry.area / 10_000
    return polygon_gdf

import os
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats

def calculate_percentages(sample_filename, classification_filename, output_csv_path=None):
    """
    Calcule les pourcentages des classes présentes dans les polygones et les sauvegarde dans un fichier CSV.
    """
    # Charger les données vectorielles
    sample_data = gpd.read_file(sample_filename)
    sample_data['surface_ha'] = sample_data.geometry.area / 10_000  # Surface en hectares
    
    # Calcul des statistiques zonales
    stats = zonal_stats(sample_data, classification_filename, categorical=True)
    stats_df = pd.DataFrame(stats).fillna(0)  # Remplacer les NaN par 0
    
    # Calcul des pourcentages
    total_pixels = stats_df.sum(axis=1)
    percentages_df = stats_df.div(total_pixels, axis=0) * 100
    percentages_df = percentages_df.fillna(0)  # Remplacer les éventuels NaN par 0 après division
    
    # Fusionner les résultats avec les données d'origine
    sample_data = pd.concat([sample_data, percentages_df], axis=1)
    
    # Supprimer les colonnes en double
    sample_data = sample_data.loc[:, ~sample_data.columns.duplicated(keep='first')]
    # S'assurer que toutes les colonnes sont bien en format texte
    sample_data.columns = sample_data.columns.astype(str)
    
    # Suppression et recréation du fichier CSV si demandé
    if output_csv_path:
        if os.path.exists(output_csv_path):
            os.remove(output_csv_path)
        sample_data.to_csv(output_csv_path, index=False)

    print("✅ Résultats enregistrés dans le fichier CSV.")
    return sample_data, percentages_df

def assign_classes(sample_data, percentages_df):
    """
    Attribue les classes prédites en fonction des règles de décision.
    """
    # Initialisation de la colonne pour les classes prédites
    sample_data['code_predit'] = np.nan

    for index, row in percentages_df.iterrows():
        surface_ha = sample_data.at[index, 'surface_ha']  # Surface en hectares

        # Si la ligne est entièrement composée de NaN, assigner une valeur par défaut
        if row.isna().all():
            sample_data.at[index, 'code_predit'] = 0  # "Inconnu"
            continue

        # Calcul des sommes des catégories principales
        feuillus_sum = sum(row.get(code, 0) for code in [11, 12, 13, 14, 15, 16])
        coniferes_sum = sum(row.get(code, 0) for code in [21, 22, 23, 24, 25, 26, 27])

        # Règles pour les surfaces < 2 ha
        if surface_ha < 2:
            if feuillus_sum > 75:  # Feuillus en îlots
                sample_data.at[index, 'code_predit'] = 16
            elif coniferes_sum > 75:  # Conifères en îlots
                sample_data.at[index, 'code_predit'] = 27
            elif coniferes_sum > feuillus_sum:  # Mélange de conifères prépondérants et feuillus
                sample_data.at[index, 'code_predit'] = 28
            else:  # Mélange de feuillus prépondérants et conifères
                sample_data.at[index, 'code_predit'] = 29

        # Règles pour les surfaces >= 2 ha
        else:
            # Identifier la classe dominante en ignorant les NaN
            max_class = row.idxmax(skipna=True)
            max_value = row[max_class]

            if max_value > 75:  # Si une classe a une proportion > 75 %
                sample_data.at[index, 'code_predit'] = max_class
            elif feuillus_sum > 75:  # Mélange feuillus
                sample_data.at[index, 'code_predit'] = 15
            elif coniferes_sum > 75:  # Mélange conifères
                sample_data.at[index, 'code_predit'] = 26
            elif coniferes_sum > feuillus_sum:  # Mélange de conifères prépondérants et feuillus
                sample_data.at[index, 'code_predit'] = 28
            else:  # Mélange de feuillus prépondérants et conifères
                sample_data.at[index, 'code_predit'] = 29

    # Ajouter le nom associé au code prédit
    sample_data['nom_predit'] = sample_data['code_predit'].map(CODE_TO_NAME)

    return sample_data


def save_results(sample_data, out_shapefile):
    """
    Sauvegarde les résultats dans un shapefile.
    """
    # Supprimer les colonnes dupliquées avant de sauvegarder
    sample_data = sample_data.loc[:, ~sample_data.columns.duplicated(keep='first')]
    
    # Renommer la colonne 'code_predit' pour éviter des conflits
    sample_data.rename(columns={'code_predit': 'code_pred'}, inplace=True)
    
    # Sauvegarder les résultats dans le shapefile
    sample_data.to_file(out_shapefile)
