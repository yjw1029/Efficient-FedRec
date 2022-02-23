from data.test import NewsDataset, UserDataset

from data.base import TrainBaseDataset, train_base_collate_fn
from data.user import TrainUserDataset, train_user_collate_fn
from data.mpc import TrainMPCDataset

def get_dataset(name):
    return eval(name)

def get_collate_fn(name):
    if name is None:
        return None
    return eval(name)