from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.adamax import Adamax
from torch.optim.adamw import AdamW
from torch.optim.asgd import ASGD
from torch.optim.lbfgs import LBFGS
from torch.optim.rmsprop import RMSprop
from torch.optim.rprop import Rprop
from torch.optim.sgd import SGD

from pytools.legacy.pytorch import curator

curator.register_module('optimizer', Adam)
curator.register_module('optimizer', Adagrad)
curator.register_module('optimizer', Adamax)
curator.register_module('optimizer', Adadelta)
curator.register_module('optimizer', AdamW)
curator.register_module('optimizer', ASGD)
curator.register_module('optimizer', LBFGS)
curator.register_module('optimizer', Rprop)
curator.register_module('optimizer', RMSprop)
curator.register_module('optimizer', SGD)
