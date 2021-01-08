# pilyso-io

The Python Image anaLYsis SOftware - IO library. This is a standalone version of the IO library originally bundled with
the mycelium analysis software [*mycelyso*](https://github.com/modsim/mycelyso).

**Warning**: The library is currently intended for internal use only, and subject to change without further notice!
Use at your own risk!

It is plugin-based, and can currently read the following formats using third-party libraries, in a unified manner:

- TIFF
- OME-TIFF
- Zeiss' CZI
- Nikon ND2
- NDIP, an experimental pilyso-io specific meta format for non-destructive image processing

Major focus was spent on allowing access to multidimensional data – multi position data as well – and accompanying metadata, in a unified manner.


# Example
```python
from pilyso_io.imagestack import ImageStack, Dimensions
from pilyso_io.imagestack.readers import *  # loads all plugins

ims = ImageStack('file.nd2')

ims = ims.view(Dimensions.PositionXY)

print(ims.size)

for position in range(ims.size[Dimensions.PositionXY]):
    print(ims[position])  # pixel data
    print(ims.meta[position].calibration)  # calibration
```

# NDIP

Bioimage data is often very large, and copying around data is space and time-consuming. However, quite often a
lot of processing steps have to be performed before the data can be meaningfully analyzed.

To this extend, NDIP is a JSON-based format, which contains individual processing steps, such as rotations or shifts, 
allowing one of the base formats to transparently be mapped into an altered form for the consuming software.

The current format is experimental and subject to change without notice!

# OMERO

See [github.com/modsim/pilyso-io-omero](https://github.com/modsim/pilyso-io-omero) for GPL licensed code for accessing OMERO servers.

# License

BSD

# Citation
 
 If you use `pilyso-io`, please cite [*mycelyso*](https://github.com/modsim/mycelyso):

```
Sachs CC, Koepff J, Wiechert W, Grünberger A, Nöh K (2019)
mycelyso – high-throughput analysis of Streptomyces mycelium live cell imaging data
BMC Bioinformatics, volume 20, 452, doi: 10.1186/s12859-019-3004-1
```

Available on the *BMC Bioinformatics* homepage at [DOI: 10.1186/s12859-019-3004-1](https://dx.doi.org/10.1186/s12859-019-3004-1).