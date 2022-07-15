import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial
from typing import *

states = {}

class GLULayer(hk.Module):
    def __init__(
        self,
        kernel_size = 3,
        input_repr = "NWC",
        kernel_repr = "WIO",
        output_repr = "NWC",
        output_channel = None,
        name=None,
    ):
        super().__init__(name=name)
        self.kernel_size = kernel_size
        self.input_repr = input_repr
        self.kernel_repr = kernel_repr
        self.output_repr = output_repr

        self.output_channel = output_channel

    def __call__(self,x):
        """
        x is a B*N*m where 
        B is the batch size 
        N is the sequence length and 
        m is the embedding size

        the output shape are the same
        """
        input_shape = x.shape
        output_channel = self.output_channel if self.output_channel is not None else x.shape[-1]
        kernel_shape = (self.kernel_size,x.shape[-1],output_channel,)
        dn = jax.lax.conv_dimension_numbers(
                input_shape, 
                kernel_shape, 
                (self.input_repr,self.kernel_repr,self.output_repr),
        )
        """Init kernel"""
        stddev = 1. / jnp.sqrt(self.kernel_size)
        kernel_init = hk.initializers.TruncatedNormal(stddev=stddev)
        kernel_w = hk.get_parameter("kernel_w", shape=kernel_shape, dtype=x.dtype, init=kernel_init)
        b_w = hk.get_parameter("bias_w", shape=(output_channel,), dtype=x.dtype, init=jnp.zeros)
        kernel_e = hk.get_parameter("kernel_e", shape=kernel_shape, dtype=x.dtype, init=kernel_init)
        b_e = hk.get_parameter("bias_e", shape=(output_channel,), dtype=x.dtype, init=jnp.zeros)

        conv_w = jax.lax.conv_general_dilated(x, kernel_w, (1,), "SAME", dimension_numbers=dn) + b_w
        conv_e = jax.lax.conv_general_dilated(x, kernel_e, (1,), "SAME", dimension_numbers=dn) + b_e
        gated = jax.nn.sigmoid(conv_e)
        out = conv_w * gated
        return out

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
    """calculate sparse_max at a certain axis"""
    if z.ndim <= 1:
        return sparse_max(z)
    else:
        z = jnp.swapaxes(z, -1, axis)
        pre_shape = z.shape
        out = jax.vmap(sparse_max)(jnp.vstack(z))
        return jnp.swapaxes(out.reshape(pre_shape),axis,-1)

class GLUActivate(hk.Module):
    def __init__(self, out_size=None, name=None):
        super().__init__(name=name)
        self.out_size = out_size

    def __call__(self,x):
        input_size = x.shape[-1]
        output_size = self.out_size if self.out_size is not None else input_size
        dtype = x.dtype

        w_shape = (input_size,output_size)
        stddev = 1./np.sqrt(input_size)
        w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        
        b_shape = (output_size,)
        b_init = jnp.zeros

        w1 = hk.get_parameter("w1", shape=w_shape, dtype=dtype, init=w_init)
        w2 = hk.get_parameter("w2", shape=w_shape, dtype=dtype, init=w_init)
        b1 = hk.get_parameter("b1", shape=b_shape, dtype=dtype, init=b_init)
        b2 = hk.get_parameter("b2", shape=b_shape, dtype=dtype, init=b_init)

        out = jnp.einsum("...j,jk->...k",x,w1)+b1
        gated = jax.nn.sigmoid(jnp.einsum("...j,jk->...k",x,w2)+b2)
        return out * gated

class FeatureBlock(hk.Module):
    def __init__(
        self,
        fc_outsize:int = 256,
        bn_config:Dict = None,
        name:str = None,
    ):
        super().__init__(name=name)

        bn_config = dict(bn_config or {})
        bn_config.setdefault("decay_rate", 0.9)
        bn_config.setdefault("eps", 1e-5)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)

        self.fc1 = hk.Linear(fc_outsize, name="fc1")
        self.bn1 = hk.BatchNorm(name="bn1",**bn_config)
        self.fc2 = hk.Linear(fc_outsize, name="fc2")
        self.bn2 = hk.BatchNorm(name="bn2",**bn_config)
    
    def __call__(
        self,
        x, # input jnp array
    ):
        global states
        # num_call = hk.get_state("num_call",[],dtype=jnp.int32,init=jnp.zeros)
        states.setdefault("num_call",0)
        num_call = states["num_call"]

        """
        x: B*N where N is the feature dim and B is the batch size
        out: B*N which is same as x for default
        """
        # is_training = hk.get_state("is_training", [], jnp.ones)
        is_training = states["is_training"]

        glu1 = GLUActivate(out_size=x.shape[-1],name="glu1")
        glu2 = GLUActivate(out_size=x.shape[-1],name="glu2")
        
        out = self.fc1(x)
        out = self.bn1(out, is_training=is_training)
        out = glu1(out)
        if num_call == 0:
            # states["num_features"] = x.shape[-1]
            # states["p"] = jnp.ones(x.shape)
            states.setdefault("num_features",x.shape[-1])
            states.setdefault("p",jnp.ones(x.shape))
            # hk.set_state("num_features", x.shape[-1])
            # hk.set_state("p", jnp.ones(x.shape))
            out1 = out
        else:
            out1 = (out+x)*np.sqrt(0.5)

        out = self.fc2(x)
        out = self.bn2(out, is_training=is_training)
        out = glu2(out)
        out = (out+out1)*np.sqrt(0.5)

        # hk.set_state("num_call", num_call+1)
        states["num_call"] += 1
        return out
        
class AttentiveTransformer(hk.Module):
    def __init__(
        self,
        gamma:float = 1.5,
        sparse_max_axis:int = -1,
        bn_config:Dict = None,
        name = None,
    ):
        super().__init__(name=name)
        bn_config = dict(bn_config or {})
        bn_config.setdefault("decay_rate", 0.9)
        bn_config.setdefault("eps", 1e-5)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)

        self.bn = hk.BatchNorm(name="bn", **bn_config)
        self.gamma = gamma

        """
        M[i] = spmax(P[i-1]*h_i(a[i-1]))
        P[i] = cumprod_i (gamma-M[i])
        so we just need to store a state that represent 
        P[i-1] for current state and update after calculating
        M[i], where P[0] is initialized by jnp.ones
        """

    def __call__(
        self,
        a:jnp.array, # represent a[i-1] for ith step
    ) -> jnp.array: # return the mask
        """
        a: B*Na array
        """
        global states
        # num_features = hk.get_state("num_features")
        num_features = states["num_features"]
        fc = hk.Linear(num_features, name="fc")

        # is_training = hk.get_state("is_training")
        is_training = states["is_training"]
        out = self.bn(fc(a),is_training=is_training)
        # p_prev = hk.get_state("p") # p is a B*N matrix
        p_prev = states["p"]
        mask = sparse_max_nd(p_prev*out, axis=-1) # fix axis=-1

        p = p_prev*(self.gamma-mask)
        # hk.set_state("p", p)
        states["p"] = p
        return mask

class PublicFeatureTransformer(hk.Module):
    def __init__(
        self,
        num_blocks:int = 1,
        name = None,
    ): 
        super().__init__(name=name)
        self.blocks = [FeatureBlock() for _ in range(num_blocks)]

    def __call__(
        self,
        x, # B*N
    ):
        out = x
        for i in range(len(self.blocks)):
            out = self.blocks[i](out)
        return out

class FeatureTransformer(hk.Module):
    def __init__(
        self,
        public_transformer:PublicFeatureTransformer,
        num_private_blocks:int = 1,
        nd:int = None,
        name=None,
    ):
        super().__init__(name=name)
        self.blocks = [FeatureBlock() for _ in range(num_private_blocks)]
        self.public = public_transformer
        self.nd = nd

    def __call__(self,x):
        out = self.public(x)
        for i in range(len(self.blocks)):
            out = self.blocks[i](out)
        # nd = self.nd if self.nd is not None else int(0.5*hk.get_state("num_features"))
        nd = self.nd if self.nd is not None else int(0.5*states["num_features"])
        a,d = jnp.split(out,[nd,],axis=-1)
        return a,d

class TabnetEncoder(hk.Module):
    def __init__(
        self,
        n_step:int = 3,
        gamma:float = 1.5,
        output_size:int = None,
        name=None,
    ):
        super().__init__(name=name)
        self.n_step = n_step
        self.public_transformer = PublicFeatureTransformer(name="encoder_public")

        self.feature_transformers = [
            FeatureTransformer(self.public_transformer)
            for _ in range(n_step)
        ]
        self.attentive_transformers = [
            AttentiveTransformer(gamma=gamma)
            for _ in range(n_step)
        ]

        self.first_feature_transformer = FeatureTransformer(self.public_transformer,name="first_feature_transformer")

        bn_config = {}
        bn_config.setdefault("decay_rate", 0.9)
        bn_config.setdefault("eps", 1e-5)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)
        self.feature_bn = hk.BatchNorm(name="feature_bn",**bn_config)
        self.output_size = output_size
    
    def __call__(self,x,is_training):
        # hk.set_state("is_training", is_training)
        # is_training = hk.get_state("is_training")
        global states
        states.setdefault("is_training", is_training)
        is_training = states["is_training"]

        feature = self.feature_bn(x, is_training=is_training)

        a0 = self.first_feature_transformer(feature)[0]
        a = a0
        d_sum = 0.

        masks = []

        for i in range(self.n_step):
            mask = self.attentive_transformers[i](a)
            masks.append(mask)
            out = feature*mask
            a,d = self.feature_transformers[i](out)
            d_sum += jax.nn.relu(d)

        output_size = self.output_size if self.output_size is not None else x.shape[-1]
        final_dsum_fc = hk.Linear(output_size,name="d_fc")
        d_out = final_dsum_fc(d_sum)

        """Clear Global States"""
        states = {}
        
        return d_out, masks




