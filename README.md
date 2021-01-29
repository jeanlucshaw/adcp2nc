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
