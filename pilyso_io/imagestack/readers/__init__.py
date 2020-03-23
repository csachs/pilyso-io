from .tiff import TiffImageStack
from .ometiff import OMETiffImageStack

from .czi import CziImageStack
from .nd2 import ND2ImageStack
from .ndip import NDIPImageStack

import os
import importlib
from itertools import product

module_lookup_environment_variable = 'PILYSO_IO_MODULES'

module_lookup_prefixes = ['', 'pilyso_io_']

if module_lookup_environment_variable in os.environ:
    for prefix, module_to_load in product(module_lookup_prefixes, os.environ[module_lookup_environment_variable].split(',')):
        try:
            importlib.import_module(prefix + module_to_load)
        except ImportError:
            pass
