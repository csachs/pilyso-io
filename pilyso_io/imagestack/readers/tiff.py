# -*- coding: utf-8 -*-
"""
The TIFF reader module implements an ImageStack able to open TIFF image stacks, by using the tifffile module.
"""

from ..imagestack import ImageStack, Dimensions

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tifffile import TiffFile


class TiffImageStack(ImageStack):
    extensions = ('.tif', '.tiff',)

    priority = 1000

    def open(self, location, **kwargs):
        self.tiff = TiffFile(location.path)

        self._tiff_first_series = self.tiff.series[0]

        self._tiff_axes = self._tiff_first_series.axes
        self._tiff_shape = self._tiff_first_series.shape
        self._tiff_memmap = self.tiff.asarray(out='memmap')

        if 'swap_axes' in self.parameters:
            from_, to_ = self.parameters['swap_axes'].split('..')
            from_, to_ = from_ + to_, to_ + from_

            self._tiff_axes = ''.join(
                to_[from_.index(axis)] if axis in from_ else axis
                for axis in self._tiff_axes
            )

        if 'S' in self._tiff_axes and 'C' not in self._tiff_axes:
            self._tiff_axes = self._tiff_axes.replace('S', 'C')

        assert self._tiff_axes.endswith('YX') or self._tiff_axes.endswith('YXS') or self._tiff_axes.endswith('YXC')

        dim = []
        sizes = []

        if 'Q' in self._tiff_axes:
            # Q is "unknown"
            if 'T' not in self._tiff_axes:
                self._tiff_axes = self._tiff_axes.replace('Q', 'T')
            elif 'R' not in self._tiff_axes:
                self._tiff_axes = self._tiff_axes.replace('Q', 'R')
            elif 'Z' not in self._tiff_axes:
                self._tiff_axes = self._tiff_axes.replace('Q', 'Z')
            else:
                raise RuntimeError('There is an unknown axis (Q) in the TIFF file,'
                                   'but as well T, R, Z axes. Please change the TIFF file.')

        for char, size in zip(self._tiff_axes, self._tiff_shape):
            if char in ('X', 'Y'):
                continue

            dim.append(Dimensions.by_char(char))
            sizes.append(size)

        self.set_dimensions_and_sizes(dim, sizes)

    # noinspection PyProtectedMember
    def notify_fork(self):
        self.tiff._fh.close()
        self.tiff._fh.open()

    def get_data(self, what):
        key = tuple([
            what.get(Dimensions.by_char(char), 0) if char not in ('X', 'Y') else slice(None, None, None)
            for char, size
            in zip(self._tiff_axes, self._tiff_shape)
        ])
        data = self._tiff_memmap.__getitem__(key)

        return data

    def get_meta(self, what):
        try:
            calibration = float(self.parameters['calibration'])
        except KeyError:
            try:
                tags = self.tiff.pages[0].tags
                assert tags['ResolutionUnit'].value == 1  # 1 == µm?
                x_resolution, y_resolution = tags['XResolution'].value, tags['YResolution'].value
                assert x_resolution == y_resolution
                calibration = x_resolution[1] / x_resolution[0]
            except (KeyError, AssertionError):
                calibration = 1.0

        try:
            interval = float(self.parameters['interval'])
        except KeyError:
            interval = 1.0

        try:
            time = what[Dimensions.Time]
        except KeyError:
            time = 1.0

        position = self.__class__.Position(x=0.0, y=0.0, z=0.0)
        meta = self.__class__.Metadata(
            time=interval * time,
            position=position,
            calibration=calibration
        )
        return meta
