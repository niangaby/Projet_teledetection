import os
from glob import glob
from osgeo import gdal
import numpy as np
import tempfile
from my_function import align_and_mask  # Cette fonction devrait gérer la découpe et le masquage correctement

# Répertoires et fichiers
input_dir = "/home/onyxia/work/data/images"
output_dir = "/home/onyxia/work/results/data/img_pretraitees/"
os.makedirs(output_dir, exist_ok=True)

# Paramètres
dst_crs = "EPSG:2154"  # Système de coordonnées cible
resolution = 10  # Résolution en mètres
nodata_value_allbands = 0  # Valeur NoData pour les bandes fusionnées
nodata_value_ndvi = -9999  # Valeur NoData pour le NDVI
clip_shapefile = "/home/onyxia/work/data/project/emprise_etude.shp"  # Emprise de découpe
mask_path = "/home/onyxia/work/results/data/img_pretraitees/masque_foret.tif"  # Masque d'origine

output_mosaic_allbands = os.path.join(output_dir, "Serie_temp_S2_allbands.tif")  # Image finale des bandes
output_mosaic_ndvi = os.path.join(output_dir, "Serie_temp_S2_ndvi.tif")  # Image finale du NDVI

# Liste des fichiers d'entrée
input_files = glob(os.path.join(input_dir, "*.tif"))

# Traitement et fusion des bandes
masked_file_paths = []
for input_file in input_files:
    print(f"Traitement de l'image : {input_file}")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_output_file:
            temp_path = temp_output_file.name
            # Appliquer la découpe et le masque via align_and_mask
            masked_ds = align_and_mask(
                input_file, mask_path, dst_crs, resolution,
                nodata_value_allbands, clip_shapefile
            )
            
            if masked_ds is not None:
                driver = gdal.GetDriverByName("GTiff")
                output_ds = driver.Create(temp_path, masked_ds.RasterXSize, masked_ds.RasterYSize, 1, gdal.GDT_Float32)
                output_ds.SetGeoTransform(masked_ds.GetGeoTransform())
                output_ds.SetProjection(masked_ds.GetProjection())
                output_band = output_ds.GetRasterBand(1)
                output_band.WriteArray(masked_ds.ReadAsArray())
                output_band.SetNoDataValue(nodata_value_allbands)
                output_ds = None
                
                masked_file_paths.append(temp_path)
            else:
                print(f"Erreur : align_and_mask a échoué pour {input_file}")
    except Exception as e:
        print(f"Erreur pendant l'alignement et le masquage : {e}")

# Fusion des bandes (si des images ont été traitées)
if masked_file_paths:
    print("Fusion des images masquées en une seule image multi-bandes...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".vrt") as vrt_temp_file:
        vrt_path = vrt_temp_file.name
        try:
            gdal.BuildVRT(vrt_path, masked_file_paths)
            gdal.Translate(output_mosaic_allbands, vrt_path, format="GTiff", noData=nodata_value_allbands)
            print(f"Image finale des bandes créée avec succès : {output_mosaic_allbands}")
        except Exception as e:
            print(f"Erreur lors de la fusion des images : {e}")
        finally:
            os.remove(vrt_path)
else:
    print("Erreur : Aucun dataset masqué à fusionner.")

# Étape 2 : Calcul et fusion du NDVI
# Chercher les fichiers des bandes nécessaires (B4 et B8)
band4_files = glob(os.path.join(input_dir, "*_B4.tif"))  # Bande rouge
band8_files = glob(os.path.join(input_dir, "*_B8.tif"))  # Bande NIR

if not band4_files or not band8_files:
    print("Erreur : Fichiers des bandes B4 ou B8 introuvables.")
    exit(1)

# Associer les fichiers B4 et B8
paired_files = []
for b4_file in band4_files:
    base_name = os.path.basename(b4_file).replace("_B4.tif", "")
    b8_file = os.path.join(input_dir, f"{base_name}_B8.tif")
    if os.path.exists(b8_file):
        paired_files.append((b4_file, b8_file))
    else:
        print(f"Fichier correspondant B8 introuvable pour {b4_file}")

if not paired_files:
    print("Erreur : Aucune paire de fichiers B4/B8 trouvée.")
    exit(1)

# Calcul et fusion du NDVI
masked_ndvi_datasets = []
for b4_file, b8_file in paired_files:
    print(f"Calcul du NDVI pour les fichiers : {b4_file} et {b8_file}")
    try:
        # Ouvrir les fichiers des bandes
        b4_ds = gdal.Open(b4_file)
        b8_ds = gdal.Open(b8_file)

        if b4_ds is None or b8_ds is None:
            print(f"Erreur : Impossible d'ouvrir {b4_file} ou {b8_file}")
            continue

        # Lire les bandes sous forme de tableau
        b4_band = b4_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        b8_band = b8_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

        # Vérification et gestion des valeurs NoData
        b4_nodata = b4_ds.GetRasterBand(1).GetNoDataValue() or 0
        b8_nodata = b8_ds.GetRasterBand(1).GetNoDataValue() or 0

        # Création du masque des zones valides (exclure les NoData)
        valid_mask = (b4_band != b4_nodata) & (b8_band != b8_nodata) & ((b8_band + b4_band) != 0)

        # Calcul du NDVI uniquement pour les zones valides
        ndvi = np.full_like(b4_band, nodata_value_ndvi, dtype=np.float32)  # Init avec NoData
        ndvi[valid_mask] = (b8_band[valid_mask] - b4_band[valid_mask]) / (b8_band[valid_mask] + b4_band[valid_mask])

        # Limiter les valeurs NDVI entre 0 et 1
        ndvi = np.clip(ndvi, 0, 1, out=ndvi)

        # Vérifiez les min/max pour débogage
        print(f"NDVI pour {b4_file} - Min: {np.min(ndvi[valid_mask])}, Max: {np.max(ndvi[valid_mask])}")

        # Sauvegarder le NDVI dans un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_ndvi_file:
            temp_ndvi_path = temp_ndvi_file.name
            driver = gdal.GetDriverByName("GTiff")
            ndvi_ds = driver.Create(temp_ndvi_path, b4_ds.RasterXSize, b4_ds.RasterYSize, 1, gdal.GDT_Float32)
            ndvi_ds.SetGeoTransform(b4_ds.GetGeoTransform())
            ndvi_ds.SetProjection(b4_ds.GetProjection())
            ndvi_band = ndvi_ds.GetRasterBand(1)
            ndvi_band.WriteArray(ndvi)
            ndvi_band.SetNoDataValue(nodata_value_ndvi)
            ndvi_ds = None

        # Reprojection du NDVI vers EPSG:2154
        print(f"Reprojection du NDVI vers {dst_crs}...")
        warp_options = gdal.WarpOptions(dstSRS=dst_crs, format="GTiff", resampleAlg=gdal.GRA_Bilinear)
        reprojected_ndvi_path = temp_ndvi_path.replace(".tif", "_reprojected.tif")
        gdal.Warp(reprojected_ndvi_path, temp_ndvi_path, options=warp_options)

        # Appliquer découpe et masque
        print("Découpe et masquage du NDVI...")
        masked_ndvi_ds = align_and_mask(reprojected_ndvi_path, mask_path, dst_crs, resolution, nodata_value_ndvi, clip_shapefile)

        if masked_ndvi_ds is not None:
            # Sauvegarder le fichier NDVI masqué
            masked_temp_path = reprojected_ndvi_path.replace(".tif", "_masked.tif")
            gdal.GetDriverByName("GTiff").CreateCopy(masked_temp_path, masked_ndvi_ds)
            masked_ndvi_datasets.append(masked_temp_path)
            print(f"NDVI masqué et découpé avec succès pour {b4_file} et {b8_file}")
        else:
            print(f"Erreur lors du masquage du NDVI pour {b4_file} et {b8_file}")

    except Exception as e:
        print(f"Erreur lors du calcul du NDVI pour {b4_file} et {b8_file} : {e}")


# Fusion des images NDVI masquées en une seule image
if masked_ndvi_datasets:
    print("Fusion des images NDVI masquées en une seule image...")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".vrt") as vrt_temp_file:
            vrt_path_ndvi = vrt_temp_file.name
            gdal.BuildVRT(vrt_path_ndvi, masked_ndvi_datasets)
            gdal.Translate(output_mosaic_ndvi, vrt_path_ndvi, format="GTiff", outputType=gdal.GDT_Float32, noData=nodata_value_ndvi)
            print(f"Image NDVI finale créée : {output_mosaic_ndvi}")
    except Exception as e:
        print(f"Erreur lors de la fusion des NDVI : {e}")
else:
    print("Erreur : Aucun fichier NDVI masqué à fusionner.")