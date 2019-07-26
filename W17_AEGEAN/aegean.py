import AEGEAN as AG

# VANILLA
opt = AG.init()
print(opt)
AG.learn(opt)

# what's the effect of a smaller latent_dim ? 
opt = AG.init()
opt.latent_dim, opt.runs_path = 4, 'small_latent'
AG.learn(opt)
