import os
import cv2
import numpy as np
import hydra
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.base import BaseModel
from data.pkl_dataset import PKLDataset
from utils.vis_utils import draw_axis

@torch.no_grad()
@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if cfg.model.type != 'base':
		print('model specified in config is not a base.py model!')
		return
		
	# load model and weights
	model = BaseModel(cfg.model.encoder, cfg.model.mlp_layers).to(device)
	model_weights_pth = cfg.model.weights_pth
	checkpoint = torch.load(model_weights_pth)
	model.load_state_dict(checkpoint['model_state_dict'])
	
	# load dataset
	dataset = PKLDataset(cfg.dataset.dir)
	
	# load camera intrinsis:
	K = np.loadtxt(cfg.intrin)
	
	for idx in range(len(dataset)):
		data = dataset[idx]
		rgb_inp = torch.from_numpy(data[0][None]/255.).float().to(device)
		rnt_gt = data[1]
		print('rnt_gt =', rnt_gt)
		
		rnt_pred = model(rgb_inp)[0].detach().cpu().numpy()
		print('rnt_pred =', rnt_pred)
		rvec, tvec = rnt_pred[:3], rnt_pred[3:]
		print('d =', np.linalg.norm(rnt_gt[3:] - tvec))
		print('')
		
		bgr = cv2.cvtColor(data[0].transpose(1,2,0), cv2.COLOR_RGB2BGR)
		#bgr = draw_axis(bgr, rnt_gt[:3], rnt_gt[3:], K) gt
		bgr = draw_axis(bgr, rvec, tvec, K)
		cv2.imshow('frame', bgr)
		if cv2.waitKey(0) & 0xFF == ord('q'):
			break
	
if __name__ == '__main__':
	main()
