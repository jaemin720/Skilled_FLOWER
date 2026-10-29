"""Bundle-bound, unbatched numpy RPC contract (no robot dependencies)."""
import hashlib
import json
import math
import zipfile
from io import BytesIO
import numpy as np

PROTOCOL = 'clean-share-policy-v2'
TRANSPORT = 'http-npz'
MEDIA_TYPE = 'application/x-npz'
MAX_PAYLOAD_BYTES = 64 * 1024**2


def bundle_id(preprocessor):
    data = dict(config=preprocessor.config, normalization=preprocessor.stats)
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def metadata(preprocessor):
    """Describe the source data bundle; action_shape is its stored chunk shape."""
    contract = preprocessor.config['actions']
    return dict(protocol=PROTOCOL, transport=TRANSPORT, bundle_id=bundle_id(preprocessor),
                observations='preprocessed', actions='normalized',
                action_shape=preprocessor.stats['schema']['action_shape'],
                action_contract=contract, simulation_only=False)


def validate_server(preprocessor, server):
    expected = metadata(preprocessor)
    for key in ('protocol', 'transport', 'bundle_id', 'observations', 'actions', 'action_shape', 'action_contract'):
        if server.get(key) != expected[key]:
            raise ValueError(f'Policy server {key} differs from the preprocessing bundle')
    if server.get('simulation_only'):
        raise ValueError('This is an offline dummy/test server; robot execution is disabled. Use --dry-run-hdf5.')


def validate_observation(preprocessor, observation):
    shapes = preprocessor.stats['schema']['shapes']
    if set(observation) != set(shapes):
        raise ValueError(f'Observation keys must be {sorted(shapes)}, got {sorted(observation)}')
    result = {}
    images = {e['target'] for e in preprocessor.config['observations'] if e['type'] == 'image'}
    for key, shape in shapes.items():
        value = np.asarray(observation[key])
        if list(value.shape) != shape or value.dtype.kind not in 'fiu' or not np.all(np.isfinite(value)):
            raise ValueError(f'Invalid observation {key}: expected finite {shape}, got {value.shape}')
        if key in images:
            if np.any(value < 0) or np.any(value > 255) or np.any(value != np.floor(value)):
                raise ValueError(f'{key} must contain RGB uint8 pixel values')
            value = value.astype(np.uint8)
        else:
            value = value.astype(np.float32)
        result[key] = value
    return result


def validate_actions(preprocessor, actions):
    """Validate a live response [H, A], independent of the dataset horizon."""
    value = np.asarray(actions, dtype=np.float32)
    action_dim = preprocessor.stats['schema']['action_shape'][1]
    if (value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != action_dim or
            not np.all(np.isfinite(value))):
        raise ValueError(f'Expected finite normalized actions [H, {action_dim}] with H >= 1, got {value.shape}')
    if value.nbytes > MAX_PAYLOAD_BYTES:
        raise ValueError('RPC payload exceeds 64 MiB')
    return value


def validate_transport(server):
    if server.get('protocol') != PROTOCOL or server.get('transport') != TRANSPORT:
        raise ValueError(f'Expected {PROTOCOL} / {TRANSPORT}; update client and server together')


def pack_arrays(arrays):
    """One uncompressed NPZ format for observations AND action chunks."""
    if not isinstance(arrays, dict) or not arrays or len(arrays) > 128:
        raise ValueError('RPC payload must be a nonempty dict of up to 128 numeric arrays')
    buffer = BytesIO()
    # Writing members explicitly avoids np.savez reserved keyword collisions
    # and permits allow_pickle=False on NumPy 1.x as well as NumPy 2.x.
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as archive:
        for key, values in arrays.items():
            if not isinstance(key, str) or not key or '\x00' in key:
                raise ValueError('Array keys must be nonempty strings without NUL')
            value = np.asarray(values)
            if value.dtype.kind not in 'biuf':
                raise ValueError(f'RPC array {key} must have a numeric dtype, got {value.dtype}')
            if value.nbytes + buffer.tell() > MAX_PAYLOAD_BYTES:
                raise ValueError('RPC payload exceeds 64 MiB')
            with archive.open(key + '.npy', 'w') as member:
                np.lib.format.write_array(member, value, allow_pickle=False)
    data = buffer.getvalue()
    if len(data) > MAX_PAYLOAD_BYTES:
        raise ValueError('RPC payload exceeds 64 MiB')
    return data


def unpack_arrays(data):
    if len(data) > MAX_PAYLOAD_BYTES:
        raise ValueError('RPC payload exceeds 64 MiB')
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            members = archive.infolist()
            if not 1 <= len(members) <= 128 or sum(m.file_size for m in members) > MAX_PAYLOAD_BYTES:
                raise ValueError('Invalid NPZ member count or expanded payload size')
            result = {}
            for member in members:
                key = member.filename[:-4]
                if (not member.filename.endswith('.npy') or not key or key in result or
                        member.compress_type != zipfile.ZIP_STORED):
                    raise ValueError('Expected unique, uncompressed NPY members')
                contents = archive.read(member)
                stream = BytesIO(contents)
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported NPY version')
                # Validate the declared allocation before numpy loads the array.
                if (dtype.kind not in 'biuf' or
                        math.prod(shape) * dtype.itemsize != len(contents) - stream.tell()):
                    raise ValueError('Invalid numeric NPY shape, dtype or data length')
                stream.seek(0)
                result[key] = np.load(stream, allow_pickle=False)
            return result
    except (zipfile.BadZipFile, EOFError, OSError) as exc:
        raise ValueError('Invalid NPZ payload') from exc
