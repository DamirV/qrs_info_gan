import os
import qrs_info_gan
import params

os.chdir("graphs")
pr = params.Params(radius=8, entry=8, num_epochs=20, lr=0.001, batch_size=1)
qrs_info_gan.experiment(pr)

pr = params.Params(radius=16, entry=8, num_epochs=20, lr=0.001, batch_size=1)
qrs_info_gan.experiment(pr)

pr = params.Params(radius=32, entry=8, num_epochs=20, lr=0.001, batch_size=1)
qrs_info_gan.experiment(pr)

pr = params.Params(radius=64, entry=8, num_epochs=10, lr=0.001, batch_size=1)
qrs_info_gan.experiment(pr)

pr = params.Params(radius=128, entry=8, num_epochs=20, lr=0.001, batch_size=1)
qrs_info_gan.experiment(pr)
