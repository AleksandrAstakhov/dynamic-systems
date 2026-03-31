from flax import nnx

state_axes = nnx.StateAxes({(nnx.Param, "default"): 0, ...: None})

@nnx.vmap(in_axes=(0, None, None), out_axes=0)
def create_v_model(rngs: nnx.Rngs, model_cls, model_args):
    return model_cls(rngs=rngs, **model_args)