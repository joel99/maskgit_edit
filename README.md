# Project setup notes
Create a new conda env, install ipykernel for notebook operation:
`conda install ipykernel --update-deps --force-reinstall`
`conda install -c conda-forge opencv`
`pip install -r requirements.txt`
`pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
Also at some point protobuf gets too new for the ckpts in this lib, so downgrade:
`pip install protobuf==3.20.*`
[Also matplotlib got wrecked at some point](https://github.com/espnet/espnet/issues/4573#issuecomment-1218707672):
`pip uninstall matplotlib`
`pip install --no-cache-dir "matplotlib"`
Also we use torchvision to load imagenet, wee...
`pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
(Because tf-ds requires full tar, but my cluster happened to have untar-ed version sitting around)



---

Other notes:
JAX Cuda support (assumes your device has some cuda available, replace appropriately)
`pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
Or
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

