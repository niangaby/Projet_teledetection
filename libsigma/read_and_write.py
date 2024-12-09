# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:35:21 2017

@author: marc lang
"""

from osgeo import gdal
import numpy as np
import geopandas as gpd


def open_image(filename, verbose=False):
  """
  Open an image file with gdal

  Paremeters
  ----------
  filename : str
      Image path to open

  Return
  ------
  osgeo.gdal.Dataset
  """
  data_set = gdal.Open(filename, gdal.GA_ReadOnly)

  if data_set is None:
      print('Impossible to open {}'.format(filename))
  elif data_set is not None and verbose:
      print('{} is open'.format(filename))

  return data_set

def get_image_dimension(data_set, verbose=False):
    """
    get image dimensions

    Parameters
    ----------
    data_set : osgeo.gdal.Dataset

    Returns
    -------
    nb_lignes : int
    nb_col : int
    nb_band : int
    """

    nb_col = data_set.RasterXSize
    nb_lignes = data_set.RasterYSize
    nb_band = data_set.RasterCount
    if verbose:
        print('Number of columns :', nb_col)
        print('Number of lines :', nb_lignes)
        print('Number of bands :', nb_band)

    return nb_lignes, nb_col, nb_band


def get_origin_coordinates(data_set, verbose=False):
    """
    get origin coordinates

    Parameters
    ----------
    data_set : osgeo.gdal.Dataset

    Returns
    -------
    origin_x : float
    origin_y : float
    """
    geotransform = data_set.GetGeoTransform()
    origin_x, origin_y = geotransform[0], geotransform[3]
    if verbose:
        print('Origin = ({}, {})'.format(origin_x, origin_y))

    return origin_x, origin_y

def get_pixel_size(data_set, verbose=False):
    """
    get pixel size

    Parameters
    ----------
    data_set : osgeo.gdal.Dataset

    Returns
    -------
    psize_x : float
    psize_y : float
    """
    geotransform = data_set.GetGeoTransform()
    psize_x, psize_y = geotransform[1],geotransform[5]
    if verbose:
        print('Pixel Size = ({}, {})'.format(psize_x, psize_y))

    return psize_x, psize_y

def convert_data_type_from_gdal_to_numpy(gdal_data_type):
    """
    convert data type from gdal to numpy style

    Parameters
    ----------
    gdal_data_type : str
        Data type with gdal syntax
    Returns
    -------
    numpy_data_type : str
        Data type with numpy syntax
    """
    if gdal_data_type == 'Byte':
        numpy_data_type = 'uint8'
    else:
        numpy_data_type = gdal_data_type.lower()
    return numpy_data_type

def load_img_as_array(filename, verbose=False):
    """
    Load the whole image into an numpy array with gdal

    Paremeters
    ----------
    filename : str
        Path of the input image

    Returns
    -------
    array : numpy.ndarray
        Image as array
    """

    # Get size of output array
    data_set = open_image(filename, verbose=verbose)
    nb_lignes, nb_col, nb_band = get_image_dimension(data_set, verbose=verbose)

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

def write_image(out_filename, array, data_set=None, gdal_dtype=None,
                transform=None, projection=None, driver_name=None,
                nb_col=None, nb_ligne=None, nb_band=None):
    """
    Write a array into an image file.

    Parameters
    ----------
    out_filename : str
        Path of the output image.
    array : numpy.ndarray
        Array to write
    nb_col : int (optional)
        If not indicated, the function consider the `array` number of columns
    nb_ligne : int (optional)
        If not indicated, the function consider the `array` number of rows
    nb_band : int (optional)
        If not indicated, the function consider the `array` number of bands
    data_set : osgeo.gdal.Dataset
        `gdal_dtype`, `transform`, `projection` and `driver_name` values
        are infered from `data_set` in case there are not indicated.
    gdal_dtype : int (optional)
        Gdal data type (e.g. : gdal.GDT_Int32).
    transform : tuple (optional)
        GDAL Geotransform information same as return by
        data_set.GetGeoTransform().
    projection : str (optional)
        GDAL projetction information same as return by
        data_set.GetProjection().
    driver_name : str (optional)
        Any driver supported by GDAL. Ignored if `data_set` is indicated.
    Returns
    -------
    None
    """
    # Get information from array if the parameter is missing
    nb_col = nb_col if nb_col is not None else array.shape[1]
    nb_ligne = nb_ligne if nb_ligne is not None else array.shape[0]
    array = np.atleast_3d(array)
    nb_band = nb_band if nb_band is not None else array.shape[2]


    # Get information from data_set if provided
    transform = transform if transform is not None else data_set.GetGeoTransform()
    projection = projection if projection is not None else data_set.GetProjection()
    gdal_dtype = gdal_dtype if gdal_dtype is not None \
        else data_set.GetRasterBand(1).DataType
    driver_name = driver_name if driver_name is not None \
        else data_set.GetDriver().ShortName

    # Create DataSet
    driver = gdal.GetDriverByName(driver_name)
    output_data_set = driver.Create(out_filename, nb_col, nb_ligne, nb_band,
                                    gdal_dtype)
    output_data_set.SetGeoTransform(transform)
    output_data_set.SetProjection(projection)

    # Fill it and write image
    for idx_band in range(nb_band):
        output_band = output_data_set.GetRasterBand(idx_band + 1)
        output_band.WriteArray(array[:, :, idx_band])  # not working with a 2d array.
                                                       # this is what np.atleast_3d(array)
                                                       # was for
        output_band.FlushCache()

    del output_band
    output_data_set = None


def xy_to_rowcol(x, y, image_filename):
    """
    Convert geographic coordinates into row/col coordinates

    Paremeters
    ----------
    x : float
      x geographic coordinate
    y : float
        y geographic coordinate
    image_filename : str
        Path of the image.

    Returns
    -------
    row : int
    col : int
    """
    # get image infos
    data_set = open_image(image_filename)
    origin_x, origin_y = get_origin_coordinates(data_set)
    psize_x, psize_y = get_pixel_size(data_set)

    # convert x y to row col
    col = int((x - origin_x) / psize_x)
    row = - int((origin_y - y) / psize_y)

    return row, col


def get_xy_from_file(filename):
    """
    Get x y coordinates from a vector point file

    Parameters
    ----------
    filename : str
        Path of the vector point file

    Returns
    -------
    list_x : np.array
    list_y : np.array
    """
    gdf = gpd.read_file(filename)
    geometry = gdf.loc[:, 'geometry']
    list_x = geometry.x.values
    list_y = geometry.y.values

    return list_x, list_y

def get_row_col_from_file(point_file, image_file):
    """
    Getrow col image coordinates from a vector point file
    and image file

    Parameters
    ----------
    point_file : str
        Path of the vector point file
    image_file : str
        Path of the raster image file

    Returns
    -------
    list_row : np.array
    list_col : np.array
    """
    list_row = []
    list_col = []
    list_x, list_y = get_xy_from_file(point_file)
    for x, y in zip(list_x, list_y):
        row, col = xy_to_rowcol(x, y, image_file)
        list_row.append(row)
        list_col.append(col)
    return list_row, list_col

def get_data_for_scikit(point_file, image_file, field_name):
    """
    Get a sample matrix and a label matrix from a point vector file and an
    image.

    Parameters
    ----------
    point_file : str
        Path of the vector point file
    image_file : str
        Path of the raster image file
    field_name : str
        Field name containing the numeric label of the sample.

    Returns
    -------
     X : ndarray or dict of ndarra
        The sample matrix. A nXd matrix, where n is the number of referenced
        pixels and d is the number of variables. Each line of the matrix is a
        pixel.
    Y : ndarray
        the label of the pixel
    """

    list_row, list_col = get_row_col_from_file(point_file, image_file)
    image = load_img_as_array(image_file)
    X = image[(list_row, list_col)]

    gdf = gpd.read_file(point_file)
    Y = gdf.loc[:, field_name].values
    Y = np.atleast_2d(Y).T

    return X, Y
