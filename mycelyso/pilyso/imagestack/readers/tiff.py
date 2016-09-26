# -*- coding: utf-8 -*-
"""
documentation
"""

from ..imagestack import ImageStack, Dimensions

from skimage.external.tifffile import TiffFile

class TiffImageStack(ImageStack):
    extensions = ('.tif', '.tiff',)

    priority = 1000

    def open(self, location, **kwargs):
        try:
            self.tiff = TiffFile(location.path, fastij=False)
        except TypeError:
            self.tiff = TiffFile(location.path)

        self.set_dimensions_and_sizes([Dimensions.Time], [len(self.tiff.pages)])

    def notify_fork(self):
        self.tiff._fh.close()
        self.tiff._fh.open()

    def get_data(self, what):
        return self.tiff.pages[what[Dimensions.Time]].asarray()

    def get_meta(self, what):
        try:
            calibration = float(self.parameters['calibration'])
        except KeyError:
            calibration = 0.0

        try:
            interval = float(self.parameters['interval'])
        except KeyError:
            interval = 0.0

        position = self.__class__.Position(x=0.0, y=0.0, z=0.0)
        meta = self.__class__.Metadata(time=interval * what[Dimensions.Time], position=position, calibration=calibration)
        return meta
