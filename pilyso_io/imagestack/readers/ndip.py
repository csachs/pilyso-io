# non-destructive image processing

import numpy as np
import json

from ..imagestack import ImageStack, Dimensions, ImageStackFilter


def shift_image(image, shift, background='input'):
    """

    :param image:
    :param shift:
    :param background:
    :return: :raise ValueError:
    """

    vertical, horizontal = shift
    vertical, horizontal = round(vertical), round(horizontal)
    height, width = image.shape

    if vertical < 0:
        source_vertical_lower = -vertical
        source_vertical_upper = height
        destination_vertical_lower = 0
        destination_vertical_upper = vertical
    else:
        source_vertical_lower = 0
        source_vertical_upper = height - vertical
        destination_vertical_lower = vertical
        destination_vertical_upper = height

    if horizontal < 0:
        source_horizontal_lower = -horizontal
        source_horizontal_upper = width
        destination_horizontal_lower = 0
        destination_horizontal_upper = horizontal
    else:
        source_horizontal_lower = 0
        source_horizontal_upper = width - horizontal
        destination_horizontal_lower = horizontal
        destination_horizontal_upper = width

    if background == 'input':
        new_image = image.copy()
    elif background == 'blank':
        new_image = np.zeros_like(image)
    else:
        raise ValueError("Unsupported background method passed. Use background or blank.")

    new_image[
        destination_vertical_lower:destination_vertical_upper,
        destination_horizontal_lower:destination_horizontal_upper
    ] = image[
        source_vertical_lower:source_vertical_upper,
        source_horizontal_lower:source_horizontal_upper
    ]
    return new_image


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


try:
    # noinspection PyUnresolvedReferences
    import cv2

    def rotate_image(image, angle):
        """
        Rotates image for angle degrees. Shape remains the same.

        :param image: input image
        :param angle: angle to rotate
        :type image: numpy.ndarray
        :type angle: float
        :rtype: numpy.ndarray
        :return: rotated image

        >>> rotate_image(np.array([[1, 0, 0, 0],
        ...                        [0, 1, 0, 0],
        ...                        [0, 0, 1, 0],
        ...                        [0, 0, 0, 1]], dtype=np.uint8), 45.0)
        array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [1, 1, 1, 1],
               [0, 0, 0, 0]], dtype=uint8)
        """

        return cv2.warpAffine(image,
                              cv2.getRotationMatrix2D((image.shape[1] * 0.5, image.shape[0] * 0.5), angle, 1.0),
                              (image.shape[1], image.shape[0]))

except ImportError:
    # DO NOT USE from scipy.misc import imrotate
    from scipy.ndimage.interpolation import rotate

    def rotate_image(image, angle):
        """

        :param image:
        :param angle:
        :return:
        """
        return rotate(image, angle=angle, reshape=False)


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

        assert 'version' in data
        assert 'type' in data

        assert data['version'] == 1
        assert data['type'] == 'ndip'

        assert 'data' in data

        first_layer = data['data']['input_layers'][0]

        output_layers = data['data']['output_layers']

        assert first_layer['type'] == 'input'

        self._ndip_ims = ImageStack(first_layer['uri'])

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
