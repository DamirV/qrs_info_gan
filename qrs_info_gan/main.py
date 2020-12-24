import os
import qrs_info_gan
import params

os.chdir("graphs")
pr = params.Params("1", batch_size=12, code_dim=2, latent_dim=10, num_channels=1)
qrs_info_gan.experiment(pr)
