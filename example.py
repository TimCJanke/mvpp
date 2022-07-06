import numpy as np
from cgm_models import cgm

N = 128
D = 3
S = 50
M = 7
#E = 35
L = 17

mus = np.random.normal(size=(N,D,1+1))
sd = np.random.normal(size=(N,D,1+1))
x = np.random.normal(size=(N,D,M))
y = np.random.normal(size=(N,D,1))

mdl=cgm(dim_out = D, 
        dim_in_mean = mus.shape[-1], 
        dim_in_std = sd.shape[-1],
        dim_in_features = x.shape[-1], 
        dim_latent = L, 
        n_samples_train = S,
        latent_dist="normal",
        model_type="ws")

mdl.model.summary()

        
# fit model
mdl.fit(x = [mus, sd, x], y = y)