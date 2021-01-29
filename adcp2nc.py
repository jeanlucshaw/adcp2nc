"""
Read and quality control binary ADCP data from command line.

Uses the CODAS library to read Teledyne RDI ADCP
files (.000, .ENX, .ENS, .LTA, etc), arranges data
into an xarray Dataset, performs QC, and saves to
netCDF in the current directory.

Rowetech files are also accepted but reading is not handled
by CODAS, and processing is much slower. Forcing processing
as upward/downward looking is not yet implemented for this
type of input. Neither are the minimum required depth or
time offset options.

It is best to inlcude as much information as possible
through the option flags as this will improve the quality
control.

This module is meant to be called from command line. A full
list of options can be displayed by calling,

.. code::

   $ adcp2nc -h

For this utility to be available at the command line, add a
file called :code:`adcp2nc` on your shell path, for example
at :code:`/usr/local/bin/` containing the following lines,

.. code::

   #!/path/to/bash
   /path/to/python /path/to/mxtoolbox/convert/adcp2nc.py "$@"

Alternatively, it can be called this way

.. code::

   $ python /path/to/adcp2nc.py -h

The code installation can be tested by running

.. code::

   $ python adcp2nc.py teledyne_workhorse.000 wh nickname
   $ python adcp2nc.py rowetech_seawatch.ens sw nickname

and then checking the netCDF files e.g. with ncdump

.. code::

   $ ncdump twh_2015-04-29_2015-12-04_rdi_adcp.nc

More attributes can be added to the output netCDF file by
creating a file called `info.adcp2nc` next to the raw ADCP
binaries. This csv file should have the format

.. code::

   attribute,value
   name_1,value_1
   name_2,value_2
   ...

See Also
--------

   * mxtoolbox.read.adcp
   * mxtoolbox.read.rtitools
   * pycurrents.adcp.rdiraw.Multiread

"""
from proc_.adcp import *
from proc_.rtitools import load_rtb_binary
from proc_.functions_ import *
from datetime import datetime
import xarray as xr
import numpy as np
import warnings
import csv
import os
import re
import argparse


__all__ = list()


# Command line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=__doc__)

    # identifies files
    parser.add_argument('files',
                        metavar='1 - files',
                        help='Expression identifying adcp files',
                        nargs='+')
    # adcp type
    parser.add_argument('adcptype',
                        metavar='2 - sonar',
                        help='''String designating type of adcp. This
                        is fed to CODAS Multiread or switches to the RTI
                        binary reader. Must be one
                        of `wh`, `os`, `bb` or `sw`''')
    # deployment nickname
    parser.add_argument('name',
                        metavar='3 - name',
                        help='''Mission, mooring, or station name to
                        prepend to the output file name.''')

    parser.add_argument('-a', '--amp-thres',
                        metavar='',
                        type=float,
                        help='Amplitude threshold (0-255). Defaults to 0.')
    parser.add_argument('-b', '--start-time',
                        metavar='',
                        type=str,
                        help='Remove data before this date (YYYYMMDDTHHMM).')
    parser.add_argument('-c', '--corr-thres',
                        metavar='',
                        type=float,
                        help='Correlation threshold (0-255). Defaults to 64.')
    parser.add_argument('-C', '--compass-adjustment',
                        metavar='',
                        type=float,
                        help='''Rotate velocities (and bottom track velocities) by
                        this angle (degrees) before adding them for motion correction.
                        Use this option to correct for compass misalignment.''')
    parser.add_argument('-d', '--depth',
                        metavar='',
                        type=float,
                        help='Water depth (scalar)')
    parser.add_argument('-D', '--force-dw',
                        action='store_true',
                        help='Force downward looking processing.')
    parser.add_argument('-e', '--end-time',
                        metavar='',
                        type=str,
                        help='Remove data after this date (YYYYMMDDTHHMM).')
    parser.add_argument('-f', '--fill-na',
                        action='store_true',
                        help='Interpolate velocity vertically and marke as changed (5).')
    parser.add_argument('-g', '--gps-file',
                        metavar='',
                        help='GPS netcdf file path and name.')
    parser.add_argument('-i', '--include-temp',
                        action='store_true',
                        help='''Include temperature. Otherwise, default behavior
                        is to save it to a difference netcdf file.''')
    parser.add_argument('-I', '--info',
                        action='store_true',
                        help='''Show information about the processed
                        velocities.''')
    parser.add_argument('-k', '--clip',
                        metavar='',
                        type=int,
                        help='Number of ensembles to clip from the end of the dataset.')
    parser.add_argument('-m', '--motion-correction',
                        metavar='',
                        help='''Motion correction mode. Defaults to no motion correction.
                        If given 'bt', will use bottom track data to correct for
                        instrument motion. If given 'gps', will use gps data to
                        correct for instrument motion but fail if no gps file is
                        provided. See vkdat2vknetcdf.py for gps file details.''')
    parser.add_argument('-o', '--t-offset',
                        metavar='',
                        type=int,
                        help='''Offset by which to correct time in hours. May for
                        example be used to move dataset from UTC to local
                        time.''')
    parser.add_argument('-p', '--pg-thres',
                        metavar='',
                        type=float,
                        help='Percentage of 4 beam threshold (0-100). Defaults to 80.')
    parser.add_argument('-P', '--pitch-thres',
                        metavar='',
                        type=float,
                        help='Pitch threshold (0-180). Defaults to 20.')
    parser.add_argument('-q', '--no-qc',
                        action='store_true',
                        help='Omit quality control.')
    parser.add_argument('-r', '--roll-thres',
                        metavar='',
                        type=float,
                        help='Roll threshold (0-180). Defaults to 20.')
    parser.add_argument('-R', '--rot-ang',
                        type=float,
                        metavar='',
                        help='''Rotate velocities clockwise (and bottom track velocities) by
                        this angle (degrees) motion correction. Use this option to place data
                        in a preferred reference frame.''')
    parser.add_argument('-s', '--sl-mode',
                        metavar='',
                        help='''Side lobe rejection mode. Default is None. If given `bt`,
                        will use range to boundary from bottom track data. If given
                        `dep` will use a constant depth but fail if depth is not
                        provided for downward looking data. If data is upward looking,
                        the average depth of the instrument is used as distance to
                        boundary.''')
    parser.add_argument('-S', '--flag-sparse',
                        action='store_true',
                        help='''Flag data beyond depths where 10/100 of the data
                        seem like bad data (4).''')
    parser.add_argument('-T', '--mindep',
                        metavar='',
                        type=float,
                        help='''Minimum instrument depth threshold. Keep only data for
                        which the instrument was below the provided depth in
                        meters.''')
    parser.add_argument('-U', '--force-up',
                        action='store_true',
                        help='Force upward looking processing.')
    parser.add_argument('-v', '--velocity-thres',
                        type=float,
                        metavar='',
                        help='''Reject velocities whose absolute value is greather than
                        this value in m/s.''')
    parser.add_argument('-z', '--zgrid',
                        metavar='',
                        help='''Bin depths to grid defined by the single column
                        depth in meters file given in argument.''')
    args = parser.parse_args()


    # Option switches
    if args.force_up and args.force_dw:
        raise ValueError("Cannot force downwards AND upwards processing")

    # Brand independent quality control defaults
    qc_defaults = dict(mode_platform_velocity=None,
                       gps_file=None,
                       corr_th=64,
                       pitch_th=20,
                       roll_th=20,
                       vel_th=2,
                       theta_2=0,
                       theta_1=0,
                       mode_sidelobes=None,
                       depth=None,
                       pg_th=90)

    # Brand dependent quality control defaults
    rti_qc_defaults = dict(amp_th=20)
    rdi_qc_defaults = dict(amp_th=0)

    # Quality control options
    user_qc_kw = {}
    if args.motion_correction:
        user_qc_kw['mode_platform_velocity'] = args.motion_correction
    if args.gps_file:
        user_qc_kw['gps_file'] = args.gps_file
    if args.corr_thres:
        user_qc_kw['corr_th'] = args.corr_thres
    if args.pg_thres:
        user_qc_kw['pg_th'] = args.pg_thres
    if args.amp_thres:
        user_qc_kw['amp_th'] = args.amp_thres
    if args.pitch_thres:
        user_qc_kw['pitch_th'] = args.pitch_thres
    if args.roll_thres:
        user_qc_kw['roll_th'] = args.roll_thres
    if args.roll_thres:
        user_qc_kw['vel_th'] = args.velocity_thres
    if args.rot_ang:
        user_qc_kw['theta_2'] = args.rot_ang
    if args.compass_adjustment:
        user_qc_kw['theta_1'] = args.compass_adjustment
    if args.sl_mode:
        user_qc_kw['mode_sidelobes'] = args.sl_mode
    if args.depth:
        user_qc_kw['depth'] = args.depth

    # Other options
    t_offset = args.t_offset / 24 if args.t_offset else 0
    clip = args.clip or 0
    min_depth = args.mindep or 0
    gridf = args.zgrid
    qc = not args.no_qc

    # Get output path
    if isinstance(args.files, list):
        abs_path = os.path.abspath(args.files[0])
    else:
        abs_path = os.path.abspath(args.files)
    path = '%s/' % os.path.dirname(abs_path)

    # Read teledyne ADCP data
    if args.adcptype in ['wh', 'bb', 'os']:
        ds = load_rdi_binary(args.files,
                             args.adcptype,
                             force_dw=args.force_dw,
                             force_up=args.force_up,
                             min_depth=min_depth)
        brand = 'RDI'
        sonar = args.adcptype
        qc_kw = {**qc_defaults, **rdi_qc_defaults, **user_qc_kw}

    # Read rowetech seawatch binaries directly
    elif args.adcptype == 'sw':
        ds = load_rtb_binary(args.files)
        brand = 'RTI'
        sonar = args.adcptype
        qc_kw = {**qc_defaults, **rti_qc_defaults, **user_qc_kw}

    # Read rowetech seawatch binaries converted to pd0
    elif args.adcptype == 'sw_pd0':
        ds = load_rdi_binary(args.files,
                             'wh',
                             force_dw=args.force_dw,
                             force_up=args.force_up,
                             min_depth=min_depth)
        brand = 'RTI'
        sonar = 'sw'
        qc_kw = {**qc_defaults, **rdi_qc_defaults, **user_qc_kw}

    # Sonar unknown
    else:
        raise ValueError('Sonar type %s not recognized' % args.adcptype)

    # Write sonar information
    ds.attrs['sonar'] = '%s %s' % (brand, sonar)

    # Quality control
    if qc:
        ds = adcp_qc(ds, **qc_kw)

    # Find maximum depth where 10% of data is good
    u_good = ds.u.where(ds.flags < 2).values
    Ng = np.asarray([np.isfinite(u_good[ii, :]).sum()
                     for ii in range(ds.z.size)])
    z_max = ds.z.values[np.argmin(np.abs(Ng.max() * 0.1 - Ng))]

    # Interpolate to z grid
    if gridf:
        # Bin to z grid
        z_grid = np.loadtxt(gridf)
        ds = xr_bin(ds, 'z', z_grid)

        # Interpolate velocities vertically and mark as changed (5)
        if args.fill_na:
            _nan_before = np.isnan(ds.u) | np.isnan(ds.v) | np.isnan(ds.w)

            ds['u'] = ds.u.interpolate_na(dim='z')
            ds['v'] = ds.v.interpolate_na(dim='z')
            ds['w'] = ds.w.interpolate_na(dim='z')
            _finite_after = np.isfinite(ds.u) | np.isfinite(ds.v) | np.isfinite(ds.w)
            ds['flags'] = xr.where(_nan_before & _finite_after, 5, ds.flags)

        # Bins outside data range flags should be missing (9)
        ds['flags'] = ds.flags.where(np.isfinite(ds.flags), 9)

    # Flag sparse data
    if args.flag_sparse:
        cond = ds.z < z_max if ds.looking == 'down' else ds.z > z_max
        ds['flags'] = ds.flags.where(cond, 4)

    # Sort by time and drop duplicates
    ds = xr_unique(ds.sortby('time'), 'time')

    # Trim if requested
    time_strt = args.start_time or ds.time.min()
    time_stop = args.end_time or ds.time.max()
    ds = ds.sel(time=slice(time_strt, time_stop))

    # Set variable range attributes
    for v in ds.data_vars:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ds[v].attrs['data_min'] = ds[v].min().values
            ds[v].attrs['data_max'] = ds[v].max().values

    # Set coordinate range attributes
    ds['z'].attrs['data_min'] = ds['z'].min().values
    ds['z'].attrs['data_max'] = ds['z'].max().values
    ds['time'].attrs['data_min'] = str(ds['time'].min().values)[:19]
    ds['time'].attrs['data_max'] = str(ds['time'].max().values)[:19]

    # Look for info.adcp2nc file and set description attributes
    if os.path.exists('%sinfo.adcp2nc' % path):
        reader = csv.DictReader(open('%sinfo.adcp2nc' % path))
        for row in reader:
            ds.attrs[row['attribute']] = row['value']

    # Set creation date attribute
    ds.attrs['history'] = 'Created: %s' % datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    # Save to netcdf
    strt = str(ds.time.values[0])[:10]
    stop = str(ds.time.values[-1])[:10]
    filename = '%s_%s_%s_%s_ADCP.nc' % (args.name, strt, stop, brand)
    ds.to_netcdf('%s%s' % (path, filename.lower()))
