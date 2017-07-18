import functools as ft
import itertools as it
import more_itertools as mit


def stream_modifier(f=lambda x: x, i=None, o=None):
    input_keys = mit.always_iterable(i)
    output_keys = mit.always_iterable(o)
    iterable_return_val = _iterable_according_to(o)
    assert(all(isinstance(key, str) for key in input_keys + output_keys))

    def modified_stream(stream):
        assert isinstance(stream, dict)
        assert all(key in stream for key in input_keys)

        arguments = map(stream.get, input_keys)

        return_vals = iterable_return_val(f(*arguments))

        stream_updates = dict(zip(output_keys, return_vals))
        return_stream = _dict_merge(stream_updates, stream)
        return return_stream

    return modified_stream


def compose(stream_modifiers):

    def composed(stream):
        return ft.reduce(lambda x, f: f(x), stream_modifiers, stream)

    return composed


# Private
def _iterable_according_to(obj):

    def always_iterable(other):
        if isinstance(obj, str):
            return (other,)
        else:
            return mit.always_iterable(other)

    return always_iterable


def _dict_merge(update_dict, default_dict):
    carryover_items = ((key, val) for key, val in default_dict.items()
                       if key not in update_dict)
    update_items = update_dict.items()
    return dict(it.chain(carryover_items, update_items))
