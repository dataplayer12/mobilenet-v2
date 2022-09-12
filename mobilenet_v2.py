import torch
import numpy as np
import torch.nn as nn
from math import ceil
import os
import time
import sys

class MobileNetV2Block(nn.Module):
	def __init__(self, inchannels, outchannels, stride, expansion):
		super().__init__()
		self.inchannels=inchannels
		self.outchannels=outchannels
		self.expchannels=expansion*inchannels #expanded channels
		
		self.pointwise = nn.Conv2d(
			self.inchannels, 
			self.expchannels, 
			kernel_size=1, 
			stride=1
			)
		
		self.activation=nn.ReLU6()
		self.depthwise=nn.Conv2d(
			self.expchannels,
			self.expchannels, 
			kernel_size=3,
			stride=stride,
			padding=1,
			groups=self.expchannels,
			)
			
		self.linearconv=nn.Conv2d(
			self.expchannels,
			self.outchannels,
			kernel_size=1,
			stride=1
			)
		
		self.add_after_conv= (inchannels==outchannels) and (stride==1)
		#add redidual connection only when possible

	def forward(self, x):
		xcopy=x
		
		x=self.pointwise(x)
		x=self.activation(x)
		x=self.depthwise(x)
		x=self.activation(x)
		x=self.linearconv(x)

		if self.add_after_conv:
			x+=xcopy
		
		return x
		
class IRLayerStack(nn.Module):
	def __init__(self, nlayers, inch, outch, expansion, stride):
		super().__init__()

		block=MobileNetV2Block(
				inchannels=inch, 
				outchannels=outch,
				stride=stride,
				expansion=expansion
				)
		bnorm=nn.BatchNorm2d(outch)

		self.mlist=nn.ModuleList([block, bnorm])
		
		for ix in range(nlayers-1):
			block=MobileNetV2Block(
				inchannels=outch, 
				outchannels=outch,
				stride=1,
				expansion=expansion
				)
			bnorm=nn.BatchNorm2d(outch)
			self.mlist.extend([block, bnorm])
		
		self.net=nn.Sequential(*self.mlist)
	
	def forward(self, x):
		return self.net(x)
		

class MobileNetV2(nn.Module):
	def __init__(self, imsize=[224, 224], inchannels=3, nclasses=1000):
		super().__init__()
		#there are 19 layers in this model
		self.nclasses=nclasses
		avgpooldims=[ceil(imsize[0]/32),ceil(imsize[1]/32)]
		
		self.initialconv=nn.Sequential(nn.Conv2d(
			inchannels,
			32,
			kernel_size=3,
			stride=2,
			padding=1
			), #output: 112x112x32
			nn.BatchNorm2d(32)
			)
		
		self.bottleneck1=nn.Sequential(
			MobileNetV2Block(
			inchannels=32, 
			outchannels=16,
			stride=1,
			expansion=1
			),
			nn.BatchNorm2d(16)
		)
		
		self.bottleneck2=IRLayerStack(nlayers=2, inch=16, outch=24, expansion=6, stride=2)
		
		self.bottleneck3=IRLayerStack(nlayers=3, inch=24, outch=32, expansion=6, stride=2)
				
		self.bottleneck4=IRLayerStack(nlayers=4, inch=32, outch=64, expansion=6, stride=2)
		
		self.bottleneck5=IRLayerStack(nlayers=3, inch=64, outch=96, expansion=6, stride=1)
		
		self.bottleneck6=IRLayerStack(nlayers=3, inch=96, outch=160, expansion=6, stride=2)
		
		self.bottleneck7=IRLayerStack(nlayers=1, inch=160, outch=320, expansion=6, stride=1)
		
		self.finalconv=nn.Sequential(
		nn.Conv2d(320, 1280, kernel_size=1),
		nn.BatchNorm2d(1280)
		)
		
		self.avgpool=nn.AvgPool2d(kernel_size=avgpooldims)
		self.flatten=nn.Flatten()
		self.classify=nn.Linear(1280, self.nclasses)
		
	def forward(self, x):
		x=self.initialconv(x)
		x=self.bottleneck1(x)
		x=self.bottleneck2(x)
		x=self.bottleneck3(x)
		x=self.bottleneck4(x)
		x=self.bottleneck5(x)
		x=self.bottleneck6(x)
		x=self.bottleneck7(x)
		x=self.finalconv(x)
		x=self.avgpool(x)
		x=self.flatten(x)
		out=self.classify(x)
		return out
		
if __name__=="__main__":
	pass
	net=MobileNetV2(imsize=[224,224], inchannels=3, nclasses=1000)
	x=torch.randn(1,3,224,224)
	y=net(x)
	print('Model works..')
	print(y.shape)
	