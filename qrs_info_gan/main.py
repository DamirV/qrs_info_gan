import os
import qrs_info_gan
import params

os.chdir("graphs")
pr = params.Params("1", batch_size=16, code_dim=2, latent_dim=10, num_channels=1, num_epochs=10, patch_len=512)
qrs_info_gan.experiment(pr)
