from torch_utils import TorchLossWrapper


def get_loss(name='L1Loss', *args, **kwargs):
    return TorchLossWrapper(name, *args, **kwargs)
