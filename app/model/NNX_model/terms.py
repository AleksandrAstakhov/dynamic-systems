import diffrax as dfx
import jax.numpy as jnp
import equinox as eqx

class DriftDiffusionTerm(dfx.AbstractTerm):
    _model: eqx.Module
    _brownian: dfx.VirtualBrownianTree


    def contr(self, t0, t1, **kwargs):

        dt = t1 - t0

        dW = self._brownian.evaluate(t0, t1)

        return dt, dW

    def vf(self, t, y, args):
        return self._model(t, y, args)

    def prod(self, vf, control):

        dt, dW = control
        drift, diff = vf

        return drift * dt + diff * dW 

    def vf_prod(self, t, y, args, control):

        dt, dW = control
        drift, diff = self._model(t, y, args)

        return drift * dt + diff * dW
