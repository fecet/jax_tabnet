# %%

from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import lax

# %%

def multiply_no_nan(x, y):
    dtype = jnp.result_type(x, y)
    return jnp.where(jnp.equal(x, 0.0), jnp.zeros((), dtype=dtype), jnp.multiply(x, y))

def reshape_to_broadcast(array: jnp.array, shape: tuple, axis: int):
    """ reshapes the `array` to be broadcastable to `shape`"""
    new_shape = [1 for _ in shape]
    new_shape[axis] = shape[axis]
    return jnp.reshape(array, new_shape)

def spmax(z):
    sort_z = jnp.flip(jnp.sort(z))
    k = jnp.arange(z.shape[-1]) + 1
    z_cumsum = jnp.cumsum(sort_z)
    k_array = 1 + k*sort_z
    k_z = jnp.where(z_cumsum<k_array)[0]
    # print(f"k_array:{k_array}")
    # print(f"z_cumsum:{z_cumsum}")
    # print(f"kz array:{k_z}")
    # print(f"sort_z:{sort_z}")
    k_z = jnp.max(k_z)
    tau_z = (z_cumsum[k_z]-1)/(k_z+1)
    # print(f"tau_z:{tau_z}")
    res = z - tau_z
    t = jnp.where(res>0,res,0.)
    return t

# @partial(jax.custom_jvp, nondiff_argnums=(1,))
@partial(jax.jit, static_argnums=(1,))
def _sparsemax(x, axis):
    # get indices of elements in the right axis
    # and reshape to allow broadcasting to other dimensions
    idxs = jnp.arange(x.shape[axis]) + 1
    idxs = reshape_to_broadcast(idxs, x.shape, axis)

    # calculate number of elements that belong to the support
    sorted_x = jnp.flip(lax.sort(x, dimension=axis), axis=axis)
    cum = jnp.cumsum(sorted_x, axis=axis)
    k = jnp.sum(jnp.where(1 + sorted_x * idxs > cum, 1, 0), axis=axis, keepdims=True)

    # calculate threshold and project to simplex
    threshold = (jnp.take_along_axis(cum, k - 1, axis=axis) - 1) / k
    return jnp.maximum(x - threshold, 0)

# %%

@jax.custom_jvp
def f(x):
    # return jnp.sum(x**2)
    return jnp.sum(x**2)

x = jnp.array([0.1,0.2,0.6])
# x = jnp.array([0.6])
# x = 0.5

@f.defjvp
def f_jvp(p,t):
    x, = p
    dx, = t
    return f(x), dx

jax.grad(f)(x)

# %%

"""f(x,y) = y*x**2+y+2"""

@jax.custom_jvp
def n1(x):
    return x

@n1.defjvp
def n1_jvp(p,t):
    x = p[0]
    dx = t[0]
    return x+2., dx 

def n2(y):
    return y

def n4(x):
    return x**2

def n5(x,y):
    return x*y

def n6(y):
    return y+2

def n7(x,y):
    return x+y

def f(x,y):
    n1_ = n1(x)
    n2_ = n2(y)
    n4_ = n4(n1_)
    n5_ = n5(n4_,n2_)
    n6_ = n6(y)
    n7_ = n7(n5_,n6_)
    return n7_

jax.grad(f,argnums=0)(3.,4.)


# %%

key = jax.random.PRNGKey(42)
w = jax.random.normal(key,shape=(4,5))
b = jnp.ones(shape=(4,))
x = jax.random.normal(key,shape=(5,))

f = lambda x:w@x+b
l = lambda p,x,y:jnp.mean(0.5*(y-x)**2)

# %%

y = jnp.array([0.1,0.2,0.5,0.3])
jax.grad(l,argnums=1)(f(x),y)

# %%

def model(theta,x):
    w,b = thea

# %%

x = jnp.array([2.,3.,1.,3.2,0.8])
x = jax.random.normal(key,shape=(5,))
# x = jnp.array([0.5,0.])
res = spmax(x)
res1 = _sparsemax(x, -1)
res1

# %%

x = jnp.array([2.,3.,1.,3.2,0.8])
def forward_fn(x):
    lin = hk.Linear(5)
    return jax.nn.selu(lin(x))

h = hk.without_apply_rng(hk.transform(forward_fn))
rng_key = jax.random.PRNGKey(43)
params = h.init(rng_key,x=x)

# %%

h.apply(x=x,params=params)

# %%

g = 1.5
p0 = jnp.array([1.,1.,1.,1.,1.])
m1 = spmax(p0*h.apply(x=x,params=params))
p1 = g-m1
m2 = spmax(p1*h.apply(x=x*m1,params=params))
p2 = (g-m1)*(g-m2)
m3 = spmax(p2*h.apply(x=x*m2,params=params))

# %%

def test_sort(x):
    # x = jnp.sort(x)
    # x = jnp.flip(x)
    y = jnp.where(x>1.)[0]
    y = jnp.max(y).astype(jnp.float32)
    return jnp.sum(x) / y

x = jnp.array([2.,3.,1.,4.,3.2,0.8])
jax.grad(test_sort)(x)
# test_sort(x)

# %%

from functools import partial

a = jnp.array([1.,2.,3.])
tree = ({"a":a,"b":2*a},{"a":3*a,"b":4*a})
# tree = ((a,2*a),(3*a,4*a))

def cumsum(prev,t):
    return prev+t,prev+t

jax.lax.scan(cumsum,init=a,xs=tree)

# %%

current_params = []

def transform(f):

    def apply_f(params,*args,**kwargs):
        current_params.append(params)
        outs = f(*args,**kwargs)
        # current_params.pop()
        return outs
    
    return apply_f

def get_params(id):
    return current_params[-1][id]

class Mymodule:
    def apply(self,x):
        a = get_params("w")*x
        b = get_params("w")

tr = transform(Mymodule().apply)
tr

# # %%
#
# params = {"w":5}
# tr(params,5)

# %%

jtr = jax.jit(tr)
jax.make_jaxpr(jtr)(params,5.)

# %%

x = jnp.zeros([5,])
def forward_fn(x):
    net = hk.nets.MLP([10,20,10])
    return net(x)

f = hk.transform(forward_fn)
rng_key = jax.random.PRNGKey(42)
params = f.init(rng_key,x=x)

# %%

def outer(x):
    @hk.transform
    # @hk.transparent
    def inner(t):
        net = hk.nets.MLP([10,20])
        return net(t)
    init_rng = hk.next_rng_key()
    # init_rng = jax.random.PRNGKey(42)
    params = hk.lift(inner.init)(init_rng,x)
    return jax.tree_map(lambda t:t.shape, params)

f = hk.transform(outer)
rng_key = jax.random.PRNGKey(42)
x = jnp.zeros([5,])
params = f.init(rng_key,x=x)
params

# %%

f.apply(x=x,params=params,rng=rng_key)

# %%

@partial(jax.custom_jvp, nondiff_argnums=(1,))
@partial(jax.jit, static_argnums=(1,))
def _sparsemax(x, axis):
    # get indices of elements in the right axis
    # and reshape to allow broadcasting to other dimensions
    idxs = jnp.arange(x.shape[axis]) + 1
    idxs = reshape_to_broadcast(idxs, x.shape, axis)

    # calculate number of elements that belong to the support
    sorted_x = jnp.flip(lax.sort(x, dimension=axis), axis=axis)
    cum = jnp.cumsum(sorted_x, axis=axis)
    k = jnp.sum(jnp.where(1 + sorted_x * idxs > cum, 1, 0), axis=axis, keepdims=True)

    # calculate threshold and project to simplex
    threshold = (jnp.take_along_axis(cum, k - 1, axis=axis) - 1) / k
    return jnp.maximum(x - threshold, 0)


@_sparsemax.defjvp
@partial(jax.jit, static_argnums=(0,))
def _sparsemax_jvp(axis, primals, tangents):
    # unpack arguments
    x = primals[0]
    dx = tangents[0]

    # calculate entmax p and auxiliary s
    p = _sparsemax(x, axis)
    s = jnp.where(p > 0, 1, 0)

    # jvp as simplified product with jacobian
    dy = dx * s
    g = jnp.sum(dy, axis=axis) / jnp.sum(s, axis=axis)
    dy = dy - jnp.expand_dims(g, axis) * s
    return p, dy


# %%

key = jax.random.PRNGKey(42)

x = jnp.array([2.,3.,1.,3.2,0.8])
x = jax.random.normal(key,shape=(5,))
x

# %%

def f(x):
    return jnp.product(_sparsemax(x,axis=-1))

jax.jacfwd(partial(_sparsemax, axis=-1))(x)

# %%

def sparse_max(z):
    sort_z = jnp.flip(jnp.sort(z))
    k = jnp.arange(z.shape[-1]) + 1
    z_cumsum = jnp.cumsum(sort_z)
    k_array = 1 + k*sort_z
    k_z = jnp.sum(jnp.where(z_cumsum<k_array,1,0))-1
    tau_z = (z_cumsum[k_z]-1)/(k_z+1)
    res = z - tau_z
    t = jnp.where(res>0,res,0.)
    return t

def sparse_max_nd(z,axis):
    if z.ndim <= 1:
        return spmax(z)
    else:
        z = jnp.swapaxes(z, -1, axis)
        pre_shape = z.shape
        out = jax.vmap(spmax)(jnp.vstack(z))
        return jnp.swapaxes(out.reshape(pre_shape),axis,-1)

# %%

key = jax.random.PRNGKey(42)
a = jax.random.normal(key, shape=(2,3,4))

def f(x):
    axis = 2
    return spmax_nd(x, axis=axis)


# %%

spmax(a[0,:,1])

# %%

a = np.arange(24).reshape(2,3,4)
a

# %%

jnp.mean(a,[0,1],keepdims=True)

# %%

jnp.split(a,2,axis=-1)
jax.nn.glu(a,axis=-1)


# %%

a = jnp.arange(12.).reshape(2,6)
jnp.split(a,[3,],axis=-1)

# %%

a

