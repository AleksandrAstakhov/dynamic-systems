from flax import nnx
import jax
import jax.numpy as jnp
import diffrax


class GrandDiffuser(nnx.Module):

    def __init__(
        self,
        in_dim,
        model_dim,
        out_dim,
        num_heads,
        head_dim,
        num_chanels,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_heads, self.head_dim = num_heads, head_dim

        self.init_enc = nnx.Linear(in_dim, model_dim, rngs=rngs)

        self.k_proj = nnx.Linear(model_dim, num_heads * head_dim, rngs=rngs)
        self.q_proj = nnx.Linear(model_dim, num_heads * head_dim, rngs=rngs)
        self.v_proj = nnx.Linear(model_dim, num_heads * head_dim, rngs=rngs)

        self.ones = jnp.eye(num_chanels)[None, :, None, :]

        self.out_proj = nnx.Linear(num_heads * head_dim, out_dim, rngs=rngs)

    def __call__(self, t, x):
        B, S, C, D = x.shape

        x = x.reshape(B * S, C, D)

        h = self.init_enc(x)

        k = self.k_proj(h)
        q = self.q_proj(h)
        v = self.v_proj(h)

        def split_heads(x):
            return x.reshape(B * S, C, self.num_heads, self.head_dim)

        k = split_heads(k)
        q = split_heads(q)
        v = split_heads(v)

        attn_logits = jnp.matmul(
            k.transpose(0, 2, 1, 3),
            q.transpose(0, 2, 3, 1),
        )
        attn_logits = attn_logits / jnp.sqrt(self.head_dim).astype(attn_logits.dtype)

        attn = jax.nn.softmax(attn_logits, axis=-1)

        out = jnp.matmul(attn, v.transpose(0, 2, 1, 3))
        out = out.transpose(0, 2, 1, 3)

        out = out.reshape(B * S, C, self.num_heads * self.head_dim)
        out = self.out_proj(out)

        out = out.reshape(B, S, C, -1)

        return out


class Grand(nnx.Module):

    def __init__(
        self,
        in_dim,
        model_dim,
        out_dim,
        num_heads,
        head_dim,
        num_chanels,
        *,
        rngs: nnx.Rngs,
        solver_cls=diffrax.Tsit5,
    ):

        self.diffuser = GrandDiffuser(
            in_dim, model_dim, out_dim, num_heads, head_dim, num_chanels, rngs=rngs
        )

        self.solver = solver_cls()

    def __call__(self, x, t):

        def f(t, y, args):
            return self.diffuser(t, y)

        term = diffrax.ODETerm(f)

        sol = diffrax.diffeqsolve(
            term,
            self.solver,
            t0=t[0],
            t1=t[-1],
            dt0=0.1,

            max_steps=32,

            y0=x,

            saveat=diffrax.SaveAt(t1=True),
        )

        return sol.ys[-1]
