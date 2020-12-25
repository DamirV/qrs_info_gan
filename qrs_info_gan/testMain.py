import torch
import qrs_info_gan
import params

path = "D:/Projects/qrs_info_gan/graphs/2020-12-25--13-19-22"
pr = params.Params("1", batch_size=16, code_dim=2, latent_dim=10, num_epochs=10, patch_len=256)

generator = qrs_info_gan.Generator(latent_dim=pr.latent_dim,
                                    n_classes=pr.n_classes, code_dim=pr.code_dim,
                                    patch_len=pr.patch_len)

discriminator = qrs_info_gan.Discriminator(n_classes=pr.n_classes,
                                            code_dim=pr.code_dim,
                                            patch_len=pr.patch_len)


discriminator = qrs_info_gan.loadModel(discriminator, "discriminator", path)
generator = qrs_info_gan.loadModel(generator, "generator", path)

print(discriminator)
print("---------------")
print(generator)

