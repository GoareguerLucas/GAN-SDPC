import os
import AEGEAN as AG

# VANILLA

opt = AG.init()
print(opt)
#if os.path.isdir('runs/vanilla')
AG.learn(opt)

# what's the effect of a smaller latent_dim ?
opt = AG.init()
opt.latent_dim, opt.runs_path = 4, 'small_latent'
AG.learn(opt)

# what's the effect of a smaller eps in batch norm ?
opt = AG.init()
opt.runs_path = 'small_eps'
opt.bn_eps = 1e-10
AG.learn(opt)

opt = AG.init()
opt.runs_path = 'big_eps'
opt.bn_eps = 1e-2
AG.learn(opt)

opt = AG.init()
opt.runs_path = 'small_momentum'
opt.bn_momentum = .01
AG.learn(opt)

opt = AG.init()
opt.runs_path = 'big_momentum'
opt.bn_momentum = .9
AG.learn(opt)
