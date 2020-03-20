import numpy as np


def parse_range(s, maximum=0):
    """
    Parses a range string.

    Supported are: num-num
    num-num%interval
    num-num,number
    num-num,~not_number,number

    etc.

    :param s: the string
    :param maximum: the maximum number acceptable in the range
    :return: a list of values
    """
    maximum -= 1
    splits = s.replace(' ', '').replace(';', ',').split(',')

    ranges = []
    remove = []

    not_values = False

    for frag in splits:
        if frag[0] == '~':
            not_values = not not_values
            frag = frag[1:]

        if '-' in frag:
            f, t = frag.split('-')

            interval = 1

            if '%' in t:
                t, _interval = t.split('%')
                interval = int(_interval)

            if t == '':
                t = maximum

            f, t = int(f), int(t)

            t = min(t, maximum)

            parsed_fragment = range(f, t + 1, interval)
        else:
            parsed_fragment = [int(frag)]

        if not_values:
            remove += parsed_fragment
        else:
            ranges += parsed_fragment

    return list(sorted(set(ranges) - set(remove)))


def prettify_numpy_array(arr, space_or_prefix):
    """
    Returns a properly indented string representation of a numpy array.

    :param arr:
    :param space_or_prefix:
    :return:
    """
    six_spaces = ' ' * 6
    prepared = repr(np.array(arr)).replace(')', '').replace('array(', six_spaces)
    if isinstance(space_or_prefix, int):
        return prepared.replace(six_spaces, ' ' * space_or_prefix)
    else:
        return space_or_prefix + prepared.replace(six_spaces, ' ' * len(space_or_prefix)).lstrip()

