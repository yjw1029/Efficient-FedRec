from model.plmnr import TextEncoder, UserEncoder, PLMNR

def get_model(name):
    return eval(name)