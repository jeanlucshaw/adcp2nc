import xarray as xr
import numpy as np
import pandas as pd
import warnings


def binc2edge(z):
    """
    Get bin edges from bin centers.

    Bin centers can be irregularly spaced. Edges are halfway between
    one point and the next.

    Parameters
    ----------
    z : numpy.array, pandas.DatetimeIndex, pandas.Series
        Bin centers.

    Returns
    -------
    numpy.array, pandas.DatetimeIndex, pandas.Series
        Bin edges.

    See Also
    --------

       * convert.bine2center

    """
    if type(z) is pd.core.indexes.datetimes.DatetimeIndex:
        TIME = pd.Series(z)
        DT = TIME.diff()[1:].reset_index(drop=True)

        # Extend time vector
        TIME = TIME.append(TIME.take([-1])).reset_index(drop=True)

        # Make offset pandas series
        OS = pd.concat((-0.5 * pd.Series(DT.take([0])),
                        -0.5 * DT,
                        0.5 * pd.Series(DT.take([-1])))).reset_index(drop=True)

        # Make bin edge vector
        EDGES = TIME + OS
    elif type(z) is pd.core.series.Series:
        DT = z.diff()[1:].reset_index(drop=True)

        # Extend time vector
        z = z.append(z.take([-1])).reset_index(drop=True)

        # Make offset pandas series
        OS = pd.concat((-0.5 * pd.Series(DT.take([0])),
                        -0.5 * DT,
                        0.5 * pd.Series(DT.take([-1])))).reset_index(drop=True)

        # Make bin edge vector
        EDGES = z + OS
    else:
        dz = np.diff(z)
        EDGES = np.r_[z[0]-dz[0]/2, z[1:]-dz/2, z[-1] + dz[-1]/2]

    return EDGES


def bine2center(bine):
    """
    Get bin centers from bin edges.

    Bin centers can be irregularly spaced. Edges are halfway between
    one point and the next.

    Parameters
    ----------
    bine : 1D array
        Bin edges.

    Returns
    -------
    1D array
        Bin centers.

    See Also
    --------

       * convert.binc2edge
    """
    return bine[:-1] + np.diff(bine) / 2


def circular_distance(a1, a2, units='rad'):
    '''
    Function circdist usage:

        d   =   circdist(a1,a2,units='rad')

    Returns to 'd' the distance between angles a1 and a2
    expected to be radians by default, or degrees if units
    is specified to 'deg'.

    Parameters
    ----------
    a1, a2 : float
        Input angle.
    units: str
        Units of input angles ('deg', 'rad')

    Returns
    -------
    float
        Angular distance between `a1` and `a2`.

    '''
    if units == 'deg':
        a1 = np.pi*a1/180
        a2 = np.pi*a2/180

    if np.isscalar(a1) and np.isscalar(a2):
        v1 = np.array([np.cos(a1), np.sin(a1)])
        v2 = np.array([np.cos(a2), np.sin(a2)])
        dot = np.dot(v1, v2)
    elif not np.isscalar(a1) and np.isscalar(a2):
        a2 = np.tile(a2, a1.size)
        v1 = np.array([np.cos(a1), np.sin(a1)]).T
        v2 = np.array([np.cos(a2), np.sin(a2)]).T
#        dot =   np.diag( v1 @ v2.T )
        dot = (v1 * v2).sum(-1)
    else:
        v1 = np.array([np.cos(a1), np.sin(a1)]).T
        v2 = np.array([np.cos(a2), np.sin(a2)]).T
#        dot =   np.diag( v1 @ v2.T )
        dot = (v1 * v2).sum(-1)

    res = np.arccos(np.clip(dot, -1., 1.))

    if units == 'deg':
        res = 180*res/np.pi

    return res


def dayofyear2dt(days, yearbase):
    """
    Takes as input days since january first of the year specified by yearbase and
    returns a list of datetime objects.

    Convert UTC days of year since January 1, 00:00:00 of `yearbase` to datetime array.

    Parameters
    ----------
    days : 1D array
        Dates in day of year format.
    yearbase : int
        Year when the data starts.

    Returns
    -------
    1D array
        Datetime equivalent of day of year dates.

    """
    start = np.array(['%d-01-01' % yearbase], dtype='M8[us]')
    deltas = np.array([np.int32(np.floor(days * 24 * 3600))], dtype='m8[s]')
    return (start + deltas).flatten()


def rotate_frame(u, v, angle, units='rad', inplace=False):
    """
    Return 2D data in rotated frame of reference.

    Rotates values of 2D vectors whose component values
    are given by `u` and `v`. This function should be thought
    of as rotating the frame of reference anti-clockwise by
    `angle`.

    Parameters
    ----------
    u, v : array_like
        Eastward and northward vector components.
    angle : float
        Rotate the frame of reference by this value.
    units : str ('deg' or 'rad')
        Units of `angle`.
    inplace : bool
        Rotate around mean instead of origin.

    Returns
    -------
    ur, vr : array_like
        Vector components in rotated reference frame.

    """
    # Size errors
    if u.shape == v.shape:
        (sz) = u.shape
    else:
        raise ValueError("u and v must be of same size")

    # Prepare vectors
    u = u.flatten()
    v = v.flatten()

    if inplace:
        o_u, o_v = np.nanmean(u), np.nanmean(v)
        u -= o_u
        v -= o_v

    # Handle deg/rad opts and build rotation matrix
    angle = angle if units == 'rad' else np.pi * angle / 180
    B = np.array([[np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]])

    # Rotate
    ur = (B @ np.array([u, v]))[0, :]
    vr = (B @ np.array([u, v]))[1, :]

    # Reshape
    ur = np.reshape(ur, sz)
    vr = np.reshape(vr, sz)

    if inplace:
        ur += o_u
        vr += o_v

    return ur, vr


def xr_bin(dataset, dim, bins, centers=True, func=np.nanmean):
    '''
    Bin dataset along `dim`.

    Convenience wrapper for the groupby_bins xarray method. Meant for
    simply binning xarray `dataset` to the values of dimension `dim`, and
    return values at bin centers (or edges) `bins`.

    Parameters
    ----------
    dataset : xarray.Dataset or xarray.DataArray
        Dataset to operate on.
    dim: str
        Name of dimension along which to bin.
    bins: array_like
        Bin centers or edges if `centers` is False.
    centers: bool
        Parameter `bins` is the centers, otherwise it is the edges.
    func: Object
        Function used to reduce bin groups.

    Returns
    -------
    xarray.Dataset
        Dataset binned at `binc` along `dim`.
    '''
    # Bin type management
    if centers:
        edge = binc2edge(bins)
        labels = bins
    else:
        edge = bins
        labels = bine2center(bins)

    # Skip for compatibility with DataArray
    if isinstance(dataset, xr.core.dataset.Dataset):
        # Save dimension orders for each variable
        dim_dict = dict()
        for key in dataset.keys():
            dim_dict[key] = dataset[key].dims

        # Save dataset attributes
        attributes = dataset.attrs

        # Save variable attributes
        var_attributes = dict()
        for v in dataset.data_vars:
            var_attributes[v] = dataset[v].attrs

        # Save variable attributes
        coord_attributes = dict()
        for c in dataset.coords:
            coord_attributes[c] = dataset[c].attrs

    # Avoids printing mean of empty slice warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)

        # Bin reduction 
        output = (dataset.groupby_bins(dataset[dim], bins=edge, labels=labels)
                   .reduce(func, dim=dim)
                   .rename({dim+'_bins': dim}))

    # Skip for compatibility with DataArray
    if isinstance(dataset, xr.core.dataset.Dataset):
        # Restore dataset attributes
        output.attrs = attributes

        # Restore variable
        for v in output.data_vars:
            output[v].attrs = var_attributes[v]

        # Restore variable
        for c in output.coords:
            output[c].attrs = coord_attributes[c]

        # Restore dimension order to each variable
        for key, dim_tuple in dim_dict.items():
            if dim not in dim_tuple:
                output[key] = dataset[key]
            else:
                output[key] = output[key].transpose(*dim_tuple)

    return output


def xr_unique(dataset, dim):
    '''
    Remove duplicates along dimension `dim`.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to operate on.
    dim : str
        Name of dimension to operate along.

    Returns
    -------
    xarray.Dataset
        Dataset with duplicates removed along `dim`.
    '''
    _, index = np.unique(dataset[dim], return_index=True)
    return dataset.isel({dim: index})
