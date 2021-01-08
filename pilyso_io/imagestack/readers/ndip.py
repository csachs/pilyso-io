# non-destructive image processing

import os
import numpy as np
import json

from ..imagestack import ImageStack, Dimensions, ImageStackFilter


def shift_image(image, shift):
    import cv2

    return cv2.warpAffine(image,
                          np.array([[1.0, 0.0, shift[1]], [0.0, 1.0, shift[0]]]),
                          (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


class TranslationFilter(ImageStackFilter):
    # noinspection PyUnusedLocal
    def __init__(self, shift=None, **kwargs):
        self.shift = shift

    def filter(self, image):
        return shift_image(image, self.shift)


class CropFilter(ImageStackFilter):
    # noinspection PyUnusedLocal
    def __init__(self, crop=None, **kwargs):
        self.crop = crop

    def filter(self, image):
        h_low, h_high, w_low, w_high = self.crop
        h_low, h_high, w_low, w_high = int(h_low), int(h_high), int(w_low), int(w_high)
        return image[h_low:h_high, w_low:w_high]


def rotate_image(image, angle):
    import cv2

    return cv2.warpAffine(image,
                          cv2.getRotationMatrix2D((image.shape[1] * 0.5, image.shape[0] * 0.5), angle, 1.0),
                          (image.shape[1], image.shape[0]))


class RotationFilter(ImageStackFilter):
    # noinspection PyUnusedLocal
    def __init__(self, angle=0.0, **kwargs):
        self.angle = angle

    def filter(self, image):
        return rotate_image(image, self.angle)


filters = {TranslationFilter, CropFilter, RotationFilter}
filter_dict = {filter_.__name__: filter_ for filter_ in filters}


def canonicalize(position):
    position = {
        (k.char if isinstance(k, type) and issubclass(k, Dimensions.Dimension) else k): v for k, v in position.items()
    }

    return tuple(sorted({dim: position[dim] if dim in position else 0 for dim in Dimensions.all_by_char()}.items(),
                        key=lambda ab: ab[0]))


def instantiate_layers(layers):
    return [
        filter_dict[layer['type']](**layer).filter
        for layer in layers
    ]


class NDIPImageStack(ImageStack):
    extensions = ('.ndip',)

    priority = 500

    def open(self, location, **kwargs):
        with open(location.path) as fp:
            data = json.load(fp)

        try:
            import cv2
        except ImportError:
            print("NDIP does need OpenCV for accelerated image processing.")
            raise

        assert 'version' in data
        assert 'type' in data

        assert data['version'] == 1
        assert data['type'] == 'ndip'

        assert 'data' in data

        first_layer = data['data']['input_layers'][0]

        output_layers = data['data']['output_layers']

        assert first_layer['type'] == 'input'

        cwd = os.getcwd()

        try:

            if os.path.isfile(location.path):
                os.chdir(os.path.dirname(os.path.abspath(location.path)))

            self._ndip_ims = ImageStack(first_layer['uri'])

        finally:
            os.chdir(cwd)

        self.set_dimensions_and_sizes(self._ndip_ims.dimensions, self._ndip_ims.sizes)

        self._ndip_output_filters = instantiate_layers(output_layers)

        self._ndip_specific_filters = {
            canonicalize(layer['position']): instantiate_layers(layer['layers'])
            for layer in data['data']['specific_layers']
        }

    def get_data(self, what):
        image = self._ndip_ims.__getitem__(
            tuple([what[dimension] if dimension in what else 0 for dimension in self._ndip_ims.dimensions])
        )

        key = canonicalize(what)

        if key in self._ndip_specific_filters:
            for filter_func in self._ndip_specific_filters[key]:
                image = filter_func(image)

        for filter_func in self._ndip_output_filters:
            image = filter_func(image)

        return image

    def get_meta(self, what):

        meta = self._ndip_ims.meta.__getitem__(
            tuple([what[dimension] if dimension in what else 0 for dimension in self._ndip_ims.dimensions])
        )

        return meta
