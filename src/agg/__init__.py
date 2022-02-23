from agg.base import BaseAggregator
from agg.user import UserAggregator
from agg.mpc import MPCAggregator

def get_agg(name):
    return eval(name)