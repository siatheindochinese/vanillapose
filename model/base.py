import torch
import torch.nn as nn
from torchvision.models import resnet50

class Encoder_ResNet50(nn.Module):
	def __init__(self):
		super(Encoder_ResNet50, self).__init__()
		
		# init pretrained resnet50
		backbone_tmp = resnet50(pretrained=True)
		num_filters = backbone_tmp.fc.in_features
		layers = list(backbone_tmp.children())[:-1]
		self.backbone = nn.Sequential(*layers)
	
	def forward(self, inp):
		oup = self.backbone(inp)
		return oup
		
class MLP(nn.Module):
	def __init__(self, in_channels, n_layers, out_channels):
		super(MLP, self).__init__()
		
		layers = []
		channel_range = in_channels - out_channels
		d_channel = int(channel_range/n_layers)
		prev = in_channels
		for i in range(n_layers):
			if i == n_layers - 1:
				layers.append(nn.Linear(prev, out_channels))
			else:
				layers.append(nn.Linear(prev, prev - d_channel))
				layers.append(nn.ReLU(inplace=True))
				prev = prev - d_channel
		self.backbone = nn.Sequential(*layers)
		
	def forward(self, inp):
		oup = self.backbone(inp)
		return oup

class Base(nn.Module):
	def __init__(self, encoder, mlp):
		super(Base, self).__init__()
		self.encoder = encoder
		self.mlp = mlp
		
	def forward(self, x):
		x = self.encoder(x)
		x = torch.flatten(x, 1)
		x = self.mlp(x)
		
		return x
		
def BaseModel(encoder_name, mlp_layers):
	if encoder_name == 'resnet50':
		encoder = Encoder_ResNet50()
		in_channels = 2048
	
	out_channels = 6 # 6DoF pose
	
	mlp = MLP(in_channels, mlp_layers, out_channels)
	return Base(encoder, mlp)
	
# DEBUG
@torch.no_grad()
def main():
	resnet_mlp3 = BaseModel('resnet50',3)
	layers = list(resnet_mlp3.children())
	for layer in layers:
		print(layer)
		print('')
		print('')
		print('')
	
	# pass a 480x640 image
	inp = torch.rand(1,3,480,640)
	oup = resnet_mlp3(inp)
	print(oup)
	
if __name__ == '__main__':
	main()
