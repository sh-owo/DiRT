import numpy as np
import jax
from jax.experimental import mesh_utils

jtu = jax.tree_util
NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec


def create_mesh(device_mesh_shape, axis_names):
    devices = mesh_utils.create_device_mesh(device_mesh_shape)
    return jax.sharding.Mesh(devices, axis_names)


def tree_broadcast(prefix, target):
    def _broadcast(leaf, subtree):
        return jtu.tree_map(lambda _: leaf, subtree)
    return jtu.tree_map(_broadcast, prefix, target)


def reshard(tree, shardings):
    def _make_global_arr(x, shard, shape):
        if hasattr(x, "sharding") and x.sharding.is_equivalent_to(shard, len(shape)):
            return x
        if not getattr(x, "is_fully_addressable", True):
            raise RuntimeError("Trying to reshard a non-fully-addressable array.")
        x = jax.device_get(x)
        xs = [jax.device_put(x[s], device=d)
              for d, s in shard.addressable_devices_indices_map(shape).items()]
        return jax.make_array_from_single_device_arrays(shape, shard, xs)

    shapes = jtu.tree_map(np.shape, tree)
    shardings = tree_broadcast(shardings, tree)
    return jtu.tree_map(_make_global_arr, tree, shardings, shapes)


def get_data_shard_fn(mesh, sharding_spec):
    n_procs = jax.process_count()
    def shard(x):
        local_ds = mesh.local_devices
        xs = jax.device_put(np.split(x, len(local_ds), axis=0), local_ds)
        global_shape = (x.shape[0] * n_procs, *x.shape[1:])
        return jax.make_array_from_single_device_arrays(global_shape, sharding_spec, xs)
    return shard


def _is_array_leaf(x):
    return isinstance(x, jax.Array)


def get_param_shardings(params, mesh, shard_fsdp=True, threshold=2**18):
    if mesh is None:
        return jtu.tree_map(lambda x: None, params, is_leaf=_is_array_leaf)

    def _get_sharding(x):
        if not isinstance(x, jax.Array):
            return None
        if x.ndim >= 2 and x.size > threshold and shard_fsdp:
            return NamedSharding(mesh, P(None, 'data'))
        return NamedSharding(mesh, P())

    return jtu.tree_map(_get_sharding, params, is_leaf=_is_array_leaf)


def shard_params(params, mesh, shard_fsdp=True, threshold=2**18):
    shardings = get_param_shardings(params, mesh, shard_fsdp, threshold)
    if not shard_fsdp or mesh is None:
        return params
    return jax.device_put(params, shardings)


def get_data_sharding(mesh, batch_ndim=2):
    data_axes = (('replica', 'data'),) + (None,) * (batch_ndim - 1)
    return NamedSharding(mesh, P(*data_axes))


def get_replicated_sharding(mesh):
    return NamedSharding(mesh, P())
