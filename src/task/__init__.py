from task.train import *
from task.base import BaseTask
from task.test import TestTask

def get_task(name):
    return eval(name)