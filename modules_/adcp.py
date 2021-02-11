from pycurrents.adcp.rdiraw import Multiread
import xarray as xr
import numpy as np
from .functions_ import rotate_frame, circular_distance, dayofyear2dt
import warnings
from scipy.stats import circmean


__all__ = ['adcp_init',
           'adcp_qc',
           'load_rdi_binary']


def adcp_init(depth, time):
    """
    Return empty xarray shell in standardized format.

    Initializes many attributes required or recommended by
    CF-1.8 conventions. If used by others that the author,
    change the default values of the `source` and `contact`
    attributes.

    Parameters
    ----------
    depth : int or 1D array
        Number of vertical bins or vertical bin vector.
    time : int or 1D array
        Number of time steps or time vector.

    Returns
    -------
    xarray.Dataset
        An empty dataset ready for data input.

    """
    # Take inputs as coordinate sizes or vectors
    if isinstance(depth, int):
        z = np.arange(depth)
    else:
        z = depth
    size_depth = z.size
    if isinstance(time, int):
        t = np.empty(time, dtype='datetime64[ns]')
    else:
        t = time 
    size_time = t.size

    # Main adcp data structure
    ds = xr.Dataset(
        data_vars = {'u': (['z', 'time'], np.nan * np.ones((size_depth, size_time))),
                     'v': (['z', 'time'], np.nan * np.ones((size_depth, size_time))),
                     'w': (['z', 'time'], np.nan * np.ones((size_depth, size_time))),
                     'e': (['z', 'time'], np.nan * np.ones((size_depth, size_time))),
                     'flags': (['z', 'time'], np.zeros((size_depth, size_time))),
                     'lon': (['time'], np.nan * np.ones(size_time)),
                     'lat': (['time'], np.nan * np.ones(size_time)),
                     'temp': (['time'], np.nan * np.ones(size_time)),
                     'depth': (['time'], np.nan * np.ones(size_time)),
                     'roll_': (['time'], np.nan * np.ones(size_time)),
                     'pitch': (['time'], np.nan * np.ones(size_time)),
                     'heading': (['time'], np.nan * np.ones(size_time)),
                     'uship': (['time'], np.nan * np.ones(size_time)),
                     'vship': (['time'], np.nan * np.ones(size_time)),
                     'u_bt': (['time'], np.nan * np.ones(size_time)),
                     'v_bt': (['time'], np.nan * np.ones(size_time)),
                     'w_bt': (['time'], np.nan * np.ones(size_time)),
                     'amp': (['z', 'time'], np.nan * np.ones((size_depth, size_time))),
                     'corr': (['z', 'time'], np.nan * np.ones((size_depth, size_time))),
                     'pg': (['z', 'time'], np.nan * np.ones((size_depth, size_time))),
                     'corr_bt': (['time'], np.nan * np.ones(size_time)),
                     'pg_bt': (['time'], np.nan * np.ones(size_time)),
                     'range_bt': (['time'], np.nan * np.ones(size_time))},
        coords = {'z' : z,
                  'time': t},
        attrs = {'Conventions': 'CF-1.8',
                 'title': '',
                 'institution': '',
                 'source': 'ADCP data, processed with https://github.com/jeanlucshaw/adcp2nc',
                 'description': '',
                 'history': '',
                 'platform': '',
                 'sonar': '',
                 'ping_frequency': '',
                 'beam_angle': '',
                 'bin_size': '',
                 'looking': '',
                 'instrument_serial': '',
                 'contact': 'Jean-Luc.Shaw@dfo-mpo.gc.ca'})

    # Set velocity attributes
    ds.u.attrs['long_name'] = 'Velocity east'
    ds.u.attrs['standard_name'] = 'eastward_sea_water_velocity'
    ds.u.attrs['units'] = 'm s-1'
    ds.v.attrs['long_name'] = 'Velocity north'
    ds.v.attrs['standard_name'] = 'northward_sea_water_velocity'
    ds.v.attrs['units'] = 'm s-1'
    ds.w.attrs['long_name'] = 'Velocity up'
    ds.w.attrs['standard_name'] = 'upward_sea_water_velocity'
    ds.w.attrs['units'] = 'm s-1'
    ds.e.attrs['long_name'] = 'Error velocity'
    ds.e.attrs['description'] = 'Difference between vertical velocity estimates'
    ds.e.attrs['units'] = 'm s-1'

    # Set motion correction attributes
    ds.u_bt.attrs['long_name'] = 'Bottom track velocity east'
    ds.u_bt.attrs['units'] = 'm s-1'
    ds.v_bt.attrs['long_name'] = 'Bottom track elocity north'
    ds.v_bt.attrs['units'] = 'm s-1'
    ds.w_bt.attrs['long_name'] = 'Bottom track velocity up'
    ds.w_bt.attrs['units'] = 'm s-1'
    ds.range_bt.attrs['long_name'] = 'Bottom track depth'
    ds.range_bt.attrs['units'] = 'm'
    ds.corr_bt.attrs['long_name'] = 'Bottom track signal correlation (all beams averaged)'
    ds.pg_bt.attrs['long_name'] = 'Bottom track percentage of 4-beam transformations'
    ds.uship.attrs['long_name'] = 'Platform velocity east'
    ds.uship.attrs['units'] = 'm s-1'
    ds.vship.attrs['long_name'] = 'Platform velocity north'
    ds.vship.attrs['units'] = 'm s-1'

    # Set signal attributes
    ds.amp.attrs['long_name'] = 'Received signal strength (all beams averaged)'
    ds.corr.attrs['long_name'] = 'Received signal correlation (all beams averaged)'
    ds.pg.attrs['long_name'] = 'Percentage of good 4-beam transformations'

    # Set gyro attributes
    ds.heading.attrs['long_name'] = 'ADCP compass heading'
    ds.heading.attrs['units'] = 'degrees'
    ds.roll_.attrs['long_name'] = 'ADCP tilt angle 1'
    ds.roll_.attrs['units'] = 'degrees'
    ds.pitch.attrs['long_name'] = 'ADCP tilt angle 2'
    ds.pitch.attrs['units'] = 'degrees'
    
    # Set time attributes
    """
    The `units` attribute is not set though required by CF-1.8. This
    is because datetime64 are already encoded with this information
    and a units attribute clashes with this encoding when saving to
    netCDF. You can check that this information is however stored in
    the variable's attributes by inspecting it with other tools than
    xarray (eg. ncdump filname.nc | head -n 50).
    """
    ds.time.attrs['long_name'] = 'Time'
    ds.time.attrs['standard_name'] = 'time'
    ds.time.attrs['axis'] = 'T'

    # Set depth attributes
    ds.z.attrs['long_name'] = 'Depth'
    ds.z.attrs['units'] = 'm'
    ds.z.attrs['positive'] = 'down'
    ds.z.attrs['axis'] = 'Z'

    # Set geographical attributes
    ds.lon.attrs['long_name'] = 'Longitude'
    ds.lon.attrs['standard_name'] = 'longitude'
    ds.lon.attrs['units'] = 'degrees_east'
    ds.lat.attrs['long_name'] = 'Latitude'
    ds.lat.attrs['standard_name'] = 'latitude'
    ds.lat.attrs['units'] = 'degrees_north'

    # Set transducer attributes
    ds.temp.attrs['long_name'] = 'ADCP transducer temperature'
    ds.temp.attrs['units'] = 'Celsius'
    ds.depth.attrs['long_name'] = 'ADCP transducer depth'
    ds.depth.attrs['units'] = 'm'

    # Set quality flag attributes
    ds.flags.attrs['long_name'] = 'Quality control flags'
    ds.flags.attrs['flag_values'] = '0 1 3 4 9'
    ds.flags.attrs['flag_meanings'] = 'raw good questionable bad missing'

    return ds


def adcp_qc(dataset,
            amp_th=30,
            pg_th=90,
            corr_th=64,
            roll_th=20,
            pitch_th=20,
            vel_th=5,
            mode_platform_velocity=None,
            mode_sidelobes=None,
            gps_file=None,
            depth=None,
            theta_1=None,
            theta_2=None):
    """
    Perform ADCP quality control.

    Parameters
    ----------
    dataset : xarray.Dataset
        ADCP dataset formatted as done by adcp_init.
    amp_th : float
        Require more than this amplitude values.
    pg_th : float
        Require more than this percentage of good 4-beam transformations.
    corr_th : float
        Require more than this beam correlation value.
    roll_th : float
        Require roll values be smaller than this value (degrees).
    pitch_th : float
        Require pitch values be smaller than this value (degrees).
    mode_platform_velocity : None
        Unknown.
    gps_file : str
        GPS dataset formatted as by gps_init.
    mode_sidelobes : str
        Use fixed depth or bottom track range to remove side lobe
        contamination. Set to either `dep` or `bt`.
    depth : float
        Fixed depth used for removing side lobe contamination.
    theta_1 : float
        Horizontally rotate velocity before motion correction by this value (degrees).
    theta_2 : float
        Horizontally rotate velocity after motion correction by this value (degrees).

    Note
    ----

       Quality control flags follow those used by DAISS for ADCP
       data at Maurice-Lamontagne Institute. Meaning of the flagging
       values is the following.

       * 0: no quality control
       * 1: datum seems good
       * 3: datum seems questionable
       * 4: datum seems bad
       * 9: datum is missing

       Data are marked as questionable if they fail only the 4beam
       transformation test. If they fail the 4beam test and any other
       non-critical tests they are marked as bad. Data likely to be
       biased from sidelobe interference are also marked as bad. If
       pitch or roll is greater than 90 degrees data are also marked
       as bad since the ADCP is not looking in the right direction.

    """
    # Work on copy
    ds = dataset.copy(deep=True)

    # Remove bins with improbable values
    for v in ['u', 'v', 'w']:
        plausible = (ds[v] > -10) & (ds[v] < 10)
        ds['u'] = ds['u'].where(plausible)
        ds['v'] = ds['v'].where(plausible)
        ds['w'] = ds['w'].where(plausible)

    # Check for gps file if required
    if mode_platform_velocity == 'gps':
        try:
            gps = xr.open_dataset(gps_file).interp(time=ds.time)
            ds['lon'].values = gps.lon
            ds['lat'].values = gps.lat
        except:
            raise NameError("GPS file not found...!")

    # Acoustics conditions (True fails)
    corr_condition = np.abs( ds.corr ) < corr_th
    pg_condition = np.abs( ds.pg ) < pg_th
    amp_condition = np.abs( ds.amp ) < amp_th

    # Roll conditions (True fails)
    roll_mean = circmean(ds.roll_.values, low=-180, high=180)
    roll_from_mean = circular_distance(ds.roll_.values, roll_mean, units='deg')
    roll_condition = np.abs(roll_from_mean) > roll_th
    roll_looking_condition = np.abs(roll_from_mean) > 90

    # Pitch conditions (True fails)
    pitch_mean = circmean(ds.pitch.values, low=-180, high=180)
    pitch_from_mean = circular_distance(ds.pitch.values, pitch_mean, units='deg')
    pitch_condition = np.abs(pitch_from_mean) > pitch_th
    pitch_looking_condition = np.abs(pitch_from_mean) > 90

    # Motion conditions (True fails)
    motion_condition = roll_condition | pitch_condition
    looking_condition = roll_looking_condition | pitch_looking_condition
    
    # Outlier conditions (True fails)
    horizontal_velocity = np.sqrt(ds.u ** 2 + ds.v ** 2)
    velocity_condition = np.greater(horizontal_velocity.values,
                                    vel_th,
                                    where=np.isfinite(horizontal_velocity))
    if 'u_bt' in ds.data_vars and 'v_bt' in ds.data_vars:
        bottom_track_condition = (np.greater(abs(ds.u_bt.values),
                                             vel_th,
                                             where=np.isfinite(ds.u_bt.values)) |
                                  np.greater(abs(ds.v_bt.values),
                                             vel_th,
                                             where=np.isfinite(ds.v_bt.values)))

    # Missing condition (True fails)
    missing_condition = (~np.isfinite(ds.u) |
                         ~np.isfinite(ds.v) |
                         ~np.isfinite(ds.w)).values

    # Boolean summary of non-critical tests (True fails)
    ncrit_condition = (corr_condition |
                       amp_condition |
                       motion_condition |
                       velocity_condition)

    # Remove side lob influence (True fails) according to a fixed depths (e.g. Moorings)
    if mode_sidelobes == 'dep':

        # Calculate these here to clarify the code below
        cos_ba = np.cos(np.pi * ds.attrs['beam_angle'] / 180)
        adcp_depth = ds.depth.mean()

        # Dowward looking
        if ds.attrs['looking'] == 'down':
            if depth != None:
                sidelobe_condition = ds.z > (adcp_depth + (depth - adcp_depth) * cos_ba)
            else:
                warnings.warn("Can not correct for side lobes, depth not provided.")

        # Upward looking
        elif ds.attrs['looking'] == 'up':
            sidelobe_condition = ds.z < adcp_depth * (1 - cos_ba)

        # Orientation unknown 
        else:
            warnings.warn("Can not correct for side lobes, looking attribute not set.")
            sidelobe_condition = np.ones(ds.z.values.size, dtype='bool')

    # Remove side lobes using bottom track
    elif mode_sidelobes == 'bt':
        msg = "Removing sidelob interference using BT range not implemented yet."
        raise KeyError(msg)

    # Do not perform side lobe removal
    else:
        sidelobe_condition = np.zeros_like(ds.u.values, dtype='bool')

    # Apply condition to bottom track velocities
    if 'u_bt' in ds.data_vars and 'v_bt' in ds.data_vars:
        for field in ['u_bt', 'v_bt', 'w_bt']:
            ds[field] = ds[field].where( bottom_track_condition )

    # Determine quality flags
    ds['flags'].values = np.ones(ds.flags.shape)
    ds['flags'].values = xr.where(pg_condition, 3, ds.flags)
    ds['flags'].values = xr.where(pg_condition & ncrit_condition, 4, ds.flags)
    ds['flags'].values = xr.where(sidelobe_condition, 4, ds.flags)
    ds['flags'].values = xr.where(looking_condition, 4, ds.flags)
    ds['flags'].values = xr.where(missing_condition, 9, ds.flags)

    # First optional rotation to correct compass misalignment
    if theta_1 not in [None, 0]:
        u, v = rotate_frame(ds.u.values, ds.v.values, theta_1, units='deg')
        ds['u'].values = u
        ds['v'].values = v

        if 'u_bt' in ds.data_vars and 'v_bt' in ds.data_vars:
            u_bt, v_bt = rotate_frame(ds.u_bt.values, ds.v_bt.values, theta_1, units='deg')
            ds['u_bt'].values = u_bt
            ds['v_bt'].values = v_bt

    # Correct for platform motion
    for field in ['u', 'v', 'w']:

        # Bottom track correction in 3D
        if mode_platform_velocity == 'bt':
            ds[field] -= ds['%s_bt' % field].values

        # GPS velocity correction in 2D
        elif mode_platform_velocity == 'gps' and (field in ['u', 'v']):
            ds[field] += np.tile(gps[field].where(np.isfinite(gps.lon.values), 0), (ds.z.size, 1))
            ds['%sship' % field].values = gps[field].values

            # Remove bottom track data if not used
            for v in ['u_bt', 'v_bt', 'w_bt', 'range_bt', 'pg_bt', 'corr_bt']:
                if v in ds.data_vars:
                    ds.drop_vars(names=v)

    # No platform velocity correction
    if mode_platform_velocity != 'gps':
        ds = ds.drop_vars(['uship', 'vship'])

    # Second optional rotation to place in chosen reference frame
    if theta_2 not in [None, 0]:
        u, v = rotate_frame(ds.u.values, ds.v.values, theta_2, units='deg')
        ds['u'].values = u
        ds['v'].values = v

    return ds


def load_rdi_binary(files,
                    adcptype,
                    force_dw=False,
                    force_up=False,
                    min_depth=0,
                    t_offset=0):
    """
    Read Teledyne RDI binary ADCP data to xarray.

    Parameters
    ----------
    files : str or list of str
        File(s) to read in.
    adcptype : str
        Sensor type passed to pycurrents.Multiread. ('wh', 'os')
    force_dw : bool
        Process as downward looking ADCP.
    force_up : bool
        Process as upward looking ADCP.
    min_depth : float
        Require instrument depth be greater that this value in meters.

    Returns
    -------
    xarray.Dataset
        ADCP data.
    
    """
    # Read
    data = Multiread(files, adcptype).read()

    # Check coordinate system
    if not data.trans['coordsystem'] == 'earth':
        raise Warning("Beams 1-4 are in %s coordinate system"
                      % data.trans['coordsystem'])

    # Get data set size and configuration
    t = dayofyear2dt(data.dday + t_offset, data.yearbase)

    # Configure depth of bin centers
    if force_dw or force_up:

        # downwards processing
        if force_dw:
            z = data.dep + np.nanmean(data.XducerDepth)  # depth of the bins
            looking = 'down'

        # upwards processing
        else:
            z = np.nanmean(data.XducerDepth) - data.dep  # depth of the bins
            looking = 'up'

    # downwards processing
    elif not data.sysconfig['up']:
        z = data.dep + np.nanmean(data.XducerDepth)  # depth of the bins
        looking = 'down'

    # upwards processing
    elif data.sysconfig['up']:
        z = np.nanmean(data.XducerDepth) - data.dep  # depth of the bins
        looking = 'up'
    else:
        raise ValueError('Could not determine ADCP orientation.')

    # Init xarray
    ds = adcp_init(z, t)

    # Set up xarray
    ds['u'].values = np.asarray(data.vel1.T)
    ds['v'].values = np.asarray(data.vel2.T)
    ds['w'].values = np.asarray(data.vel3.T)
    ds['e'].values = np.asarray(data.vel4.T)
    ds['corr'].values = np.asarray(np.mean(data.cor, axis=2).T)
    ds['amp'].values = np.asarray(np.mean(data.amp, axis=2).T)
    ds['pg'].values = np.float64(np.asarray(data.pg4.T))
    ds['depth'].values = np.asarray(data.XducerDepth)
    ds['heading'].values = np.asarray(data.heading)
    ds['roll_'].values = np.asarray(data.roll)
    ds['pitch'].values = np.asarray(data.pitch)
    ds['temp'].values = np.asarray(data.temperature)

    # Bottom track data if it exists
    if not (data.bt_vel.data == 0).all():
        ds['u_bt'].values = data.bt_vel.data[:, 0]
        ds['v_bt'].values = data.bt_vel.data[:, 1]
        ds['w_bt'].values = data.bt_vel.data[:, 2]
        ds['range_bt'].values = np.nanmean(data.bt_depth.data, axis=-1)

    # Remove data shallower than min depth
    if min_depth > 0:
        selected = data.XducerDepth >= min_depth
        ds = ds.sel(time=selected)

    # If no bottom track data, drop variables
    else:
        bt_vars = ['u_bt', 'v_bt', 'w_bt', 'range_bt', 'pg_bt', 'corr_bt']
        ds = ds.drop_vars(names=bt_vars)

    # Attributes
    ds.attrs['beam_angle'] = data.sysconfig['angle']
    ds.attrs['ping_frequency'] = data.sysconfig['kHz'] * 1000
    ds.attrs['bin_size'] = data.CellSize
    ds.attrs['looking'] = looking
    ds.attrs['instrument_serial'] = str(data.FL['Inst_SN'])

    # Flip z if upward looking
    if ds.attrs['looking'] == 'up':
        ds = ds.sortby('z')

    # Sort according to time
    ds = ds.sortby('time')

    return ds
