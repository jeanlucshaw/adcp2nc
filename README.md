# adcp2nc
Standalone version of the command line ADCP processing tool in mxtoolbox.

## Installation
Clone the repository on to your system. Intall the package dependencies listed in this repository. The [pycurrents](https://currents.soest.hawaii.edu/docs/adcp_doc/codas_setup/index.html) and and [rti_python](https://github.com/rowetechinc/rti_python) packages are not available through conda. Follow the instructions on their respective links for installation.

## Usage
This utility is meant to be used via command line on linux systems. You can use it by invoking python:

```
$ python /path/to/adcp2nc.py -h
```

or save an executable shell script at /usr/local/bin/adcp2nc which contains

```
#!/path/to/bash
/path/to/python /path/to/adcp2nc.py "$@"
```

such that it can then be called directly from command line, i.e.

```
$ adcp2nc -h
```

## Examples
Sample ADCP data files are included from a Teledyne RDI Worhorse and a Rowetech Inc. Seawatch ADCP. The most basic processing requires 3 arguments: the binary file (or wildcard pattern), the sonar type -- wh (workhorse), os (ocean surveyor), bb (broadband), nb (narrow band) or sw (seawatch) -- and the deployment name to prefix the output with. To apply this basic processing to the sample file run

```
$ python /path/to/adcp2nc.py /path/to/teledyne_workhorse.000 wh sample_data
```

or

```
$ adcp2nc /path/to/teledyne_workhorse.000 wh sample_data
```

The output will be a netCDF file with the name structure `nickname`_`start
date`-`end date`_`ADCP brand (rdi or rti)`.nc . Inside, the data are
organised following [CF
conventions](https://cfconventions.org/). Additional global attributes can
be set programmatically in the netCDF by creating a file name
`info.adcp2nc` next to the raw binaries before processing. This file should
be a `csv` structured as follows

```
attribute,value
name_1,value_1
name_2,value_2
...,...
```

## Features

### Platform velocity correction
ADCPs are often mounted on mobile platforms (e.g., oceanographic buoys or
boats) and thus measure velocity biased by the platform's movement. Three
options are implemented in `adcp2nc` to correct for this bias: no platform
velocity correction, correction using bottom track data and correction
using a position log (GPS).

Platform velocity correction is not needed in certain cases, for example if
the ADCP was mounted on a bottom lander so the user is free to omit this
step. This is the default software behavior. To specify motion correction
the `-m` flag must be supplied either `bt` or `gps`.

If platform velocity correction is using a GPS log is requested, a gps file
containing the logged positions must be supplied using the `-g` option.

```
$ adcp2nc /path/to/teledyne_workhorse.000 wh sample_data -m gps -g lon_lat_time.nc
```

The expected GPS file is in netCDF format and contains:

* Coordinate: `time` in format `numpy.datetime64[ns]`
* Variable: `lon` (longitude east) in decimal degrees
* Variable: `lat` (latitude north) in decimal degrees

Checking the ADCP binary for latitude and longitude data and optionally
using it for platform velocity correction is not yet implemented but part
of the development plan.

### Vertical binning and interpolation
Averaging data into vertical bins can be usefull to smooth noisy ADCP data
or to ensure that different files of an ADCP time series have the same
vertical coordinate. In `adcp2nc` this is handled by the `-z` option which
must be given the path to a grid file. The expected grid file is plain text
as generated by

```
$ seq min_depth bin_size max_depth > z.grid
```

and once the file is generated 

```
$ adcp2nc /path/to/teledyne_workhorse.000 wh sample_data -z z.grid
```

processes with vertical binning. Another way to smooth data is to fill gaps
with vertical linear interpolation. This is achieved by calling the `-f`
option and results in data flagged as modified (5).
