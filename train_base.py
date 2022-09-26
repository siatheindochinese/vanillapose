import os
import hydra
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.base import BaseModel
from data.pkl_dataset import PKLDataset

@hydra.main(config_path="configs/", config_name="resnet50_mlp3.yaml")
def main(cfg):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if cfg.model.type != 'base':
		print('model specified in config is not a base.py model!')
		return
		
	# load Tensorboard SummaryWriter
	writer = SummaryWriter()
	
	# load model
	model = BaseModel(cfg.model.encoder, cfg.model.mlp_layers).to(device)
	
	# load checkpoint (if resuming training)
	if cfg.train_params.resume_from_checkpoint is False:
		start_epoch = 0
	else:
		checkpoints_dir = os.path.join(writer.log_dir,'checkpoints')
		if not os.path.exists(checkpoints_dir):
			os.makedirs(checkpoints_dir)
		start_epoch = cfg.train_params.start_epoch
	
	# load hyperparams
	epochs = cfg.train_params.epochs
	bs = cfg.train_params.bs
	lr = cfg.train_params.lr
	
	# load dataset and dataloader
	dataset = PKLDataset(cfg.dataset.dir)
	len_train = int(len(dataset) * float(cfg.dataset.train_ratio))
	len_val = len(dataset) - len_train
	train, val = torch.utils.data.random_split(dataset, (len_train, len_val))
	
	train_dataloader = DataLoader(train, batch_size=bs, shuffle=True, drop_last=True)
	val_dataloader = DataLoader(val, batch_size=bs, shuffle=True, drop_last=True)
	
	# training loss
	criterion = nn.MSELoss()
	
	# optimizer
	optimizer=torch.optim.SGD(model.parameters(), lr = lr)
	
	print('config loaded, beginning training...')
	print('')
	
	# training loop
	for epoch in range(start_epoch, epochs):
		print('epoch =', epoch)
		running_train_loss = 0
		num_train_batches = 0
		running_val_loss = 0
		num_val_batches = 0
		
		train_iterator = iter(train_dataloader)
		val_iterator = iter(val_dataloader)
		
		# forward + backward pass per batch
		for i in range(len_train // bs):
			print('     batch =', i)
			optimizer.zero_grad()
			
			# load batch
			batch_inp, batch_gt = next(train_iterator)
			batch_inp, batch_gt = batch_inp.float().to(device), batch_gt.float().to(device)
			batch_oup = model(batch_inp)
			
			loss = criterion.to(device)(batch_oup, batch_gt)
			print('     batch loss =', loss)
			
			loss.backward()
			optimizer.step()
			
			# compute stats
			running_train_loss += loss.detach().cpu().item()
			num_train_batches += 1
			
			# optimise GPU RAM
			torch.cuda.empty_cache()
			
		# compute val loss
		# all done on cpu
		with torch.no_grad():
			for i in range(len_val // bs):
				running_val_loss = 0
				num_batches = 0
				batch_inp, batch_gt = next(val_iterator)
				batch_inp = batch_imgs.float()
				batch_oup = model.cpu()(batch_inp)
				loss = criterion(batch_oup, batch_gt)
				
				running_val_loss += loss.detach().cpu().item()
				num_val_batches += 1
		
		# Compile stats and save to tensorboard SummaryWriter
		avg_train_loss = running_train_loss / num_train_batches
		avg_val_loss = running_val_loss / num_val_batches
		print('epoch train loss =', avg_train_loss)
		print('epoch test loss =', avg_train_loss)
		writer.add_scalar('Loss/train', avg_train_loss, epoch)
		writer.add_scalar('Loss/val', avg_val_loss, epoch)
		
		# save model at every epoch
		checkpoint_pth = os.path.join(checkpoints_dir, '0.tar')
		checkpoint = {'epoch': epoch,
					  'model_state_dict': model.state_dict(),
					  }
		torch.save(checkpoint, checkpoint_pth)
	
if __name__ == '__main__':
	main()
