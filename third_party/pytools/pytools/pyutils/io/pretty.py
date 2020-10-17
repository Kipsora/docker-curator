from pytools.pyutils.io import io_registry_group
from pytools.pyutils.io.file_system import as_file_descriptor

__all__ = ['load_json', 'dump_json', 'load_yaml', 'dump_yaml', 'loads_json', 'dumps_json', 'loads_yaml', 'dumps_yaml',
           'dumps_table']


def load_json(file, **kwargs):
    import json
    with as_file_descriptor(file, mode='r') as reader:
        return json.load(reader, **kwargs)


def dump_json(file, data, **kwargs):
    import json
    kwargs.setdefault('indent', 4)
    with as_file_descriptor(file, mode='w') as writer:
        json.dump(data, writer, **kwargs)


def loads_json(data, **kwargs):
    import json
    return json.loads(data, **kwargs)


def dumps_json(data, **kwargs):
    import json

    class JSONExtendedEncoder(json.JSONEncoder):
        def default(self, o):
            # Adapted from https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
            if hasattr(o, '__jsonify__'):
                return o.__jsonify__()
            if hasattr(o, '__jsonstr__'):
                return o.__jsonstr__

            try:
                import numpy as np
                if isinstance(o, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                                  np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(o)
                if isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
                    return float(o)
                if isinstance(o, (np.ndarray,)):
                    return o.tolist()
            except ModuleNotFoundError:
                pass
            try:
                return super(JSONExtendedEncoder, self).default(o)
            except TypeError:
                return str(o)

    kwargs.setdefault('indent', 4)
    kwargs.setdefault('cls', JSONExtendedEncoder)
    return json.dumps(data, **kwargs)


try:
    import yaml


    def load_yaml(file, **kwargs):
        kwargs.setdefault('Loader', yaml.FullLoader)
        with as_file_descriptor(file, mode='r') as reader:
            return yaml.load(reader, **kwargs)


    def dump_yaml(file, data, **kwargs):
        with as_file_descriptor(file, mode='w') as writer:
            return yaml.dump(data, writer, **kwargs)


    def loads_yaml(data, **kwargs):
        kwargs.setdefault('Loader', yaml.FullLoader)
        return yaml.load(data, **kwargs)


    def dumps_yaml(data, **kwargs):
        kwargs.setdefault('indent', 4)
        return yaml.dump(data, **kwargs)


    io_registry_group.register('load', '.yaml', load_yaml)
    io_registry_group.register('dump', '.yaml', dump_yaml)
    io_registry_group.register('load', '.yml', load_yaml)
    io_registry_group.register('dump', '.yml', dump_yaml)
except ModuleNotFoundError:
    pass


def dumps_table(data, **kwargs):
    assert isinstance(data, dict)
    indent = kwargs.get('indent', 0)
    max_key_length = max(map(len, map(str, data.keys())))
    lines = [' ' * indent + str(k) + ': ' + ' ' * (max_key_length - len(str(k))) + str(v) for k, v in data.items()]
    return '\n'.join(lines)


io_registry_group.register('load', '.json', load_json)
io_registry_group.register('dump', '.json', dump_json)
