import torch
import torch.nn as nn

from enet.util import log


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        self.hyperparams = None
        self.device = torch.device("cpu")

    def __getnewargs__(self):
        # for pickle
        return self.hyperparams

    def __new__(cls, *args, **kwargs):
        log('created %s with params %s' % (str(cls), str(args)))

        instance = super(Model, cls).__new__(cls)
        instance.__init__(*args, **kwargs)
        return instance

    def test_mode_on(self):
        self.test_mode = True
        self.eval()

    def test_mode_off(self):
        self.test_mode = False
        self.train()

    def parameters_requires_grads(self):
        return list(filter(lambda p: p.requires_grad, self.parameters()))

    def parameters_requires_grad_clipping(self):
        return self.parameters_requires_grads()

    def save_model(self, path):
        state_dict = self.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = value.cpu()
        torch.save(state_dict, path)

    def load_model(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
