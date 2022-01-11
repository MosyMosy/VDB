# from .resnet import *
from .dataparallel_wrapper import *
from .resnet10 import ResNet10

from .resnet10_BT import ResNet10_BT
# from .resnet10_BIT import ResNet10_BIT
from .resnet10_BTrans import ResNet10_BTrans
# from .resnet10_BITrans import ResNet10_BITrans

from .resnet_BT import resnet18 as resnet18_BT
# from .resnet_BIT import resnet18 as resnet18_BIT
from .resnet_BTrans import resnet18 as resnet18_BTrans
# from .resnet_BITrans import resnet18 as resnet18_BITrans

from .resnet12 import Resnet12

from .resnet import resnet18 as resnet18_s