# Project setup notes
Beyond
`pip install -r requirements.txt`
We also need:
OpenCV
`conda install -c conda-forge opencv`
JAX Cuda support (assumes your device has some cuda available, replace appropriately)
`pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
Or
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Consider also that you might need to downgrade matplotlib to explicitly use the numpy that came [from pip](https://github.com/espnet/espnet/issues/4573#issuecomment-1218707672)..
Downgrade protobuf
`pip install protobuf==3.20.*`
