import torch

device = None
if torch.cuda.is_available() and args.gpu >= 0:
    device = torch.device(f'cuda:{args.gpu}')
    print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
else:
    device = 'cpu'

data_path = 'D:/downloaded_DS' #'./data'
miniImageNet_path = data_path + '/miniImagenet'
DTD_path = data_path + '/dtd/images/'

ISIC_path = data_path + "/ISIC2018"
ChestX_path = data_path + "/ChestX-Ray8"
CropDisease_path = data_path + "/plant-disease/dataset/train/"
EuroSAT_path = data_path + "/EuroSAT/2750"
art_path = data_path + "/art"
Imagenetv2_path = data_path + "/imagenetv2"
underwater_path = data_path + "/underwater"
content_path = './content'

num_workers = 8
