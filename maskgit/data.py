# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
import numpy as np
import jax.numpy as jnp
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                    shuffle=False, sampler=None,
                    batch_sampler=None, num_workers=0,
                    pin_memory=False, drop_last=False,
                    timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)

class JustCast(object):
    def __call__(self, pic):
        return np.array(pic, dtype=jnp.float32) / 255.
        # return np.ravel(np.array(pic, dtype=jnp.float32) / 255.)

# Custom data loading function
# We switch to pytorch since I can't figure how to get tfds working without having the tar rather than unzipped
def get_datasets(data_dir='./data/imagenet_full'):
    breakpoint()
    transform = transforms.Compose([
        transforms.RandomResizedCrop((256, 256)),
        JustCast()
    ])
    train_dataset = torchvision.datasets.ImageNet(data_dir, split='train', transform=transform)
    test_dataset = torchvision.datasets.ImageNet(data_dir, split='val', transform=transform)
    return train_dataset, test_dataset
