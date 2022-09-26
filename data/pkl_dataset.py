from torch.utils.data import Dataset
import pickle as pkl
import os
import os.path as osp
import cv2
import numpy as np

class PKLDataset(Dataset):
	def __init__(self, data_dir):
		self.data_dir = data_dir
		self.data_list = os.listdir(self.data_dir)
		self.data_list = list(filter(lambda x:x[-3:]=='pkl', self.data_list))
		self.length = len(self.data_list)
		
	def __len__(self):
		return self.length
		
	def __getitem__(self, idx):
		filename = str(idx)+'.pkl'
		filepath = osp.join(self.data_dir, filename)
		data = pkl.load(open(filepath,'rb'))
		
		rgb = data['rgb']
		depth = data['depth']
		mask = data['mask']
		K = data['K']
		RT = data['RT']
		cls_typ = data['cls_typ']
		
		r = cv2.Rodrigues(RT[:,:3])[0].squeeze()
		t = RT[:,3]
		rnt = np.concatenate((r,t),axis=0)
		
		# (w, h, c to c, h, w)
		rgb = rgb.transpose(2, 0, 1)
		
		return rgb, rnt
		
# DEBUG
def main():
	dataset = PKLDataset('/home/intern1/Linemod_preprocessed/renders/ufcoco')
	print(len(dataset))
	cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	idx = 0
	while True:
		data = dataset[idx]
		rgb = data[0].transpose(1,2,0)
		rnt = data[1]
		print(rnt)
		bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
		
		cv2.imshow('frame', bgr)
		if cv2.waitKey(0) & 0xFF == ord('q'):
			break
		
		idx += 1
		if idx == len(dataset):
			break
	cv2.destroyAllWindows()
	
if __name__ == '__main__':
	main()
