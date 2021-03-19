import datetime
import glob
import os
import xarray as xr
import numpy as np
from scipy.stats import circmean
from tqdm import tqdm
from rti_python.Codecs.BinaryCodec import BinaryCodec
from rti_python.Ensemble.EnsembleData import *
from .adcp import adcp_init
from .functions_ import xr_bin

__all__ = ['index_rtb_data',
           'load_rtb_binary',
           'read_rtb_ensemble',
           'read_rtb_file']


# Constants
DELIMITER = b'\x80' * 16


def index_rtb_data(file_path):
    """
    Read binary as byte stream. Find ensemble locations and sizes.

    Parameters
    ----------
    file_path : str
        File path and name.

    Returns
    -------
    1D array
        Ensemble start index values.
    1D array
        Ensemble data lengths.
    1D array
        Data as byte stream.

    """
    # Open binary
    with open(file_path, 'rb') as df:

        # Read data file
        data = df.read()
        ensemble_starts = []
        ensemble_lengths = []

        # Get payload size of first ensemble
        payloadsize = int.from_bytes(data[24: 27],'little')

        # Get individual ensemble starts and lengths
        ii = 0
        while ii < len(data) - payloadsize - 32 - 4:
            if data[ii: ii + 16] == DELIMITER:
                ensemble_starts.append(ii)
                ensemble_lengths.append(payloadsize + 32 + 4)

                # Increment by payload size, plus header plus checksum
                ii  +=  payloadsize +   32  +   4
            else:
                print("Data format bad")
                break

            # Get payload size of next ensemble
            payloadsize =   int.from_bytes(data[ii+24: ii+27],'little')

    return ensemble_starts, ensemble_lengths, data


def load_rtb_binary(files):
    """
    Read Rowetech RTB binary ADCP data to xarray.

    Parameters
    ----------
    files : str or list of str
        File name or expression designating .ENS files, or list or file names.

    Returns
    -------
    xarray.Dataset
        ADCP data as organized by mxtoolbox.read.adcp.adcp_init .

    """

    # Make list of files to read
    if isinstance(files, str):
        adcp_files = glob.glob(files)
    else:
        adcp_files = files

    # Make xarray from file list
    if len(adcp_files) > 1:
        # xarray_datasets = [read_rtb_file(f) for f in adcp_files if f[-3:]=='ENS']
        xarray_datasets = [read_rtb_file(f) for f in adcp_files]
        ds = xr.concat(xarray_datasets, dim='time')

        # Find average depths by bin
        bins = np.arange(0, ds.z.max() + ds.bin_size, ds.bin_size)
        lbb = bins - 0.5 * ds.bin_size
        ubb = bins + 0.5 * ds.bin_size
        z_bins = np.array([ds.z.where((ds.z < ub) & (ds.z > lb)).mean().values for lb, ub in zip(lbb, ubb)])
        z_bins = z_bins[np.isfinite(z_bins)]

        # Bin to these depths for uniformity
        ds = xr_bin(ds, 'z', z_bins)
    else:
        ds = read_rtb_file(*adcp_files)

    return ds


def read_rtb_file(file_path):
    """
    Read data from one RTB .ENS file into xarray.

    Parameters
    ----------
    file_path : str
        File name and path.

    Returns
    -------
    xarray.Dataset
        As organized by mxtoolbox.read.adcp.adcp_init .

    """
    # Index the ensemble starts
    idx, enl, data = index_rtb_data(file_path)

    chunk = data[idx[0]: idx[1]]
    if BinaryCodec.verify_ens_data(chunk):
        ens     =   BinaryCodec.decode_data_sets(chunk)

    # Get coordinate sizes
    ens_count = len(idx)
    bin_count = ens.EnsembleData.NumBins
    bin_size = ens.AncillaryData.BinSize

    # Initialize xarray dataset
    ds = adcp_init(bin_count, ens_count)
    time = np.empty(ens_count, dtype='datetime64[ns]')

    # Read and store ensembles
    with tqdm(total = len(idx)-1, desc="Processing "+file_path,unit=' ensembles') as pbar:
        for ii in range( len(idx) ):

            # Get data binary chunck for one ensemble
            chunk   =   data[ idx[ii]:idx[ii] + enl[ii] ]

            # Check that chunk looks ok
            if BinaryCodec.verify_ens_data(chunk):

                # Decode data variables
                ens = BinaryCodec.decode_data_sets(chunk)

                CORR = np.array(ens.Correlation.Correlation)
                AMP = np.array(ens.Amplitude.Amplitude)
                PG = np.array(ens.GoodEarth.GoodEarth)

                time[ii] = ens.EnsembleData.datetime()
                ds.u.values[:, ii] = np.array(ens.EarthVelocity.Velocities)[:, 0]
                ds.v.values[:, ii] = np.array(ens.EarthVelocity.Velocities)[:, 1]
                ds.w.values[:, ii] = np.array(ens.EarthVelocity.Velocities)[:, 2]
                ds.temp.values[ii] = ens.AncillaryData.WaterTemp
                ds.depth.values[ii] = ens.AncillaryData.TransducerDepth
                ds.heading.values[ii] = ens.AncillaryData.Heading
                ds.pitch.values[ii] = ens.AncillaryData.Pitch
                ds.roll_.values[ii] = ens.AncillaryData.Roll
                ds.corr.values[:, ii] = np.nanmean(CORR, axis=-1)
                ds.amp.values[:, ii] = np.nanmean(AMP, axis=-1)
                ds.pg.values[:, ii] = PG[:, 3]

                # Bottom track data
                if ens.IsBottomTrack:
                    ds.u_bt[ii] = np.array(ens.BottomTrack.EarthVelocity)[0]
                    ds.v_bt[ii] = np.array(ens.BottomTrack.EarthVelocity)[1]
                    ds.w_bt[ii] = np.array(ens.BottomTrack.EarthVelocity)[2]
                    ds.pg_bt[ii] = np.nanmean(ens.BottomTrack.BeamGood, axis=-1)
                    ds.corr_bt[ii] = np.nanmean(ens.BottomTrack.Correlation, axis=-1)
                    ds.range_bt[ii] = np.nanmean(ens.BottomTrack.Range, axis=-1)
            pbar.update(1)

    # Determine up/down configuration
    mroll = np.abs(180 * circmean(np.pi * ds.roll_.values / 180) / np.pi)
    if mroll >= 0 and mroll < 30:
        ds.attrs['looking'] = 'up'
    else:
        ds.attrs['looking'] = 'down'

    # Determine bin depths
    if ds.attrs['looking'] == 'up':
        z = np.asarray(ds.depth.mean()
                       - ens.AncillaryData.FirstBinRange
                       - np.arange(0, bin_count * bin_size, bin_size)).round(2)
    else:
        z = np.asarray(ds.depth.mean()
                       + ens.AncillaryData.FirstBinRange
                       + np.arange(0, bin_count * bin_size, bin_size)).round(2)

    # Roll near zero means downwards (like RDI)
    roll_ = ds.roll_.values + 180
    roll_[roll_ > 180] -= 360
    ds['roll_'].values = roll_

    # Correlation between 0 and 255 (like RDI)
    ds['corr'].values *= 255

    # Set coordinates and attributes
    z_attrs, t_attrs = ds.z.attrs, ds.time.attrs
    ds = ds.assign_coords(z=z, time=time)
    ds['z'].attrs = z_attrs
    ds['time'].attrs = t_attrs

    # Get beam angle
    if ens.EnsembleData.SerialNumber[1] in '12345678DEFGbcdefghi':
        ds.attrs['beam_angle'] = 20
    elif ens.EnsembleData.SerialNumber[1] in 'OPQRST':
        ds.attrs['beam_angle'] = 15
    elif ens.EnsembleData.SerialNumber[1] in 'IJKLMNjklmnopqrstuvwxy':
        ds.attrs['beam_angle'] = 30
    elif ens.EnsembleData.SerialNumber[1] in '9ABCUVWXYZ':
        ds.attrs['beam_angle'] = 0
    else:
        raise ValueError("Could not determine beam angle.")

    # Manage coordinates and remaining attributes
    ds.attrs['bin_size'] = bin_size
    ds.attrs['instrument_serial'] = ens.EnsembleData.SerialNumber
    ds.attrs['ping_frequency'] = ens.SystemSetup.WpSystemFreqHz

    return ds


def read_rtb_ensemble(file_path ,N=0):
    """
    Read one ensemble from a RTB .ENS file.

    Parameters
    ----------
    file_path : str
        Name and path of the RTB file.
    N : int
        Index value of the ensemble to read.

    Returns
    -------
    rti_python.Ensemble.Ensemble
        Ensemble data object.

    """
    ensemble_starts, ensemble_lengths, data = index_rtb_data(file_path)

    chunk = data[ensemble_starts[N]: ensemble_starts[N] + ensemble_lengths[N]]
    if BinaryCodec.verify_ens_data(chunk):
        ens = BinaryCodec.decode_data_sets(chunk)
    else:
        ens = []

    return ens
