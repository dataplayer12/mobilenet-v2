import darklight as dl
import torch
from mobilenet_v2 import MobileNetV2
from torchvision import models

teacher=models.resnet152(pretrained=True)
student= MobileNetV2(imsize=[224, 224], inchannels=3, nclasses=1000)
dl.exportonnx(teacher, 'rn152.onnx', bsize=1, hw=[224,224])

del teacher #free up CPU or GPU memory used by teacher

dm=dl.ImageNetManager('/sfnvme/imagenet/', size=[224,224], bsize=128)

opt_params={
	'optimizer': torch.optim.AdamW,
	'okwargs': {'lr': 1e-4, 'weight_decay':0.05},
	'scheduler':torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
	'skwargs': {'T_0':10,'T_mult':2},
	'amplevel': 'O2'
	}

stream=torch.cuda.Stream()

with torch.cuda.stream(stream):
	trainer=dl.StudentTrainer(student, dm, 'rn152.onnx', opt_params=opt_params)
	trainer.train(epochs=50, save='dltest_{}.pth')