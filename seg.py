import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.utils.data as Data
from tqdm import tqdm

from torchvision import models
from torchvision.models.vgg import VGG

import logging
import sys
import urllib
from thop import profile

import time
from PIL import Image
from os import listdir
import shutil
from evaluation import Bin_classification_cal
from torchstat import stat
from collections import OrderedDict
import copy
from segNet import segnet

IMAGE_SIZE = [672, 752]

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_logger(log_path='log_path'):
	if not os.path.exists(log_path):
		os.mkdir(log_path)
	timer = time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime())
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('[%(levelname)s]   %(asctime)s    %(message)s')
	txthandle = logging.FileHandler((log_path+'/'+timer+'log.txt'))
	txthandle.setFormatter(formatter)
	logger.addHandler(txthandle)
	return logger

def show(tensor, strIndex):
	img = tensor[0][0]
	lab = tensor[1][0]
	out = tensor[2][0]
	
	img = img.detach().cpu().squeeze().numpy()
	lab = lab.detach().cpu().squeeze().numpy()
	out = out.detach().cpu().squeeze().numpy()
	
	img_name = "./savepng/" + strIndex + '_img.jpg'
	lab_name = "./savepng/" + strIndex + '_lab.jpg'
	out_name = "./savepng/" + strIndex + '_out.jpg'

	cv2.imwrite(img_name, img)
	cv2.imwrite(lab_name, lab*255)
	cv2.imwrite(out_name, out*255)


	plt.figure()
	ax1 = plt.subplot(1,3,1)
	ax1.set_title('Input')
	plt.imshow(cv2.split(img)[0], cmap="gray")
	ax2 = plt.subplot(1,3,2)
	ax2.set_title('Label')
	plt.imshow(lab, cmap="gray")
	ax3 = plt.subplot(1,3,3)
	ax3.set_title('Output')
	plt.imshow(out, cmap="gray")

	picName = './visualization/' + strIndex + '.jpg'
	plt.savefig(picName)
	plt.cla()
	plt.close("all")

def caculate(output, label, clear=False):
	cal = Bin_classification_cal(output, label, 0.5, clear)
	return cal.caculate_total()


def del_models(file_path, count=5):
	dir_list = os.listdir(file_path)
	if not dir_list:
		print('file_path is empty: ', file_path)
		return
	else:
		dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
		print('dir_list: ', dir_list)
		if len(dir_list) > 5:
			os.remove(file_path + '/' + dir_list[0])

		return dir_list

def ImageBinarization(img, threshold=1):
	img = np.array(img)
	image = np.where(img > threshold, 1, 0)
	return image

def label_preprocess(label):
	label_pixel = ImageBinarization(label)
	return  label_pixel

def cvTotensor(img):
	img = (np.array(img[:, :, np.newaxis]))
	img = np.transpose(img,(2,0,1))
	img = (np.array(img[np.newaxis, :,:, :]))    
	tensor = torch.from_numpy(img)
	tensor = torch.as_tensor(tensor, dtype=torch.float32)
	return tensor

def cvTotensor_img(img):
	img = np.transpose(img,(2,0,1))
	img = (np.array(img[np.newaxis, :,:, :]))    
	tensor = torch.from_numpy(img)
	tensor = torch.as_tensor(tensor, dtype=torch.float32)
	return tensor

iterations = 0
net = segnet(1,1)
print(net)
net.cuda()

def normalization(data):
	_range = np.max(data) - np.min(data) + 0.0001
	return (data - np.min(data)) / _range

def getInput_and_Label_generator(data_path):
	img_Path = data_path + "/img"
	dmp_Path = data_path + "/img_dmp"
	l = os.listdir(img_Path)
	random.shuffle(l)
	for filename in l:
		img_name = img_Path + '/' + filename
		# print(img_name)
		label_name = data_path + '/lab/' + filename.split('.')[0] + "_label.jpg"
		# print(label_name)
		if os.path.exists(dmp_Path):
			dmp_name = dmp_Path + '/' + filename
			dmp = cv2.imread(img_name, 0)
			dmp = cv2.resize(dmp, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
			dmp = normalization(dmp)
		else:
			dmp = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1]))

		img = cv2.imread(img_name, 0)
		img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)

		img = cvTotensor(img)
		lab = cv2.imread(label_name, 0)
		lab = cv2.resize(lab, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)

		lab = cvTotensor(label_preprocess(lab))
		dmp = cvTotensor(dmp)
		yield img, lab, dmp
		
def getInput_and_Label_generator_valid(data_path):
	img_Path = data_path + "/img"
	dmp_Path = data_path + "/img_dmp"
	l = os.listdir(img_Path)

	for filename in l:
		img_name = img_Path + '/' + filename
		label_name = data_path + '/lab/' + filename.split('.')[0] + "_label.jpg"

		if os.path.exists(dmp_Path):
			dmp_name = dmp_Path + '/' + filename
			dmp = cv2.imread(img_name, 0)
			dmp = cv2.resize(dmp, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
			dmp = normalization(dmp)

		img = cv2.imread(img_name, 0)
		img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)

		img = cvTotensor(img)

		lab = cv2.imread(label_name, 0)
		lab = cv2.resize(lab, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)

		lab = cvTotensor(label_preprocess(lab))
		dmp = cvTotensor(dmp)
		yield img, lab, dmp

def train(net, epoch, iterations, loss_stop, positive_path, negative_path):
	net.train()
	epoch_loss = 0.0
	print('train...')
	g_postive = getInput_and_Label_generator(positive_path)
	g_negative = getInput_and_Label_generator(negative_path)

	for iters in tqdm(range(iterations)):
		for index in range(2):
			if index == 0:
				inputs, labels, bmps = next(g_postive)
			else:
				inputs, labels, bmps = next(g_negative)

			inputs = inputs.cuda()
			labels = labels.cuda()
			bmps = bmps.cuda()

			optimizer.zero_grad()
			outputs = net(inputs, bmps)

			lab = labels.detach().cpu().squeeze().numpy()
			out = outputs.detach().cpu().squeeze().numpy()

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			epoch_loss += loss

	epoch_loss_mean = epoch_loss / iterations
	print('Train Epoch: {}\t Total Loss: {:.6f}\t Average Loss: {:.6f}'.format(epoch, epoch_loss.item(), epoch_loss_mean.item()))
	logger.info('Train Epoch:[{}] , loss: {:.6f}'.format(epoch, epoch_loss.item()))
	if epoch_loss < loss_stop:
		return True, epoch_loss
	else:
		return False, epoch_loss


def valid(net, epoch, img_path):
	#net.eval()
	valid_loss = 0.0
	img_Path = img_path + "/img"
	l = os.listdir(img_Path)
	iterations = 50#len(l)
	print('img_Path: ', img_Path, 'len: ', iterations)
	g_data = getInput_and_Label_generator_valid(img_path)
	IoU1 = 0
	IoU2 = 0
	MIoU = 0
	PA = 0
	with torch.no_grad():
		for iters in tqdm(range(iterations)):
			inputs, labels, bmps = next(g_data)

			inputs = inputs.cuda()
			labels = labels.cuda()
			bmps = bmps.cuda()


			optimizer.zero_grad()
			outputs = net(inputs, bmps)

			lab = labels.detach().cpu().squeeze().numpy()
			out = outputs.detach().cpu().squeeze().numpy()

			DICE, IOU, PA, FP, FN = caculate(out, lab, (not bool(iters)))

			strIndex = str(epoch) + '_valid_' + str(iters)
			show([inputs, labels, outputs], strIndex)

			# outputs.reshape(1, -1)
			# labels.reshape(1, -1)
			valid_loss += criterion(outputs, labels)

		valid_loss_mean = valid_loss / iterations
		print('           Valid Epoch: {}\t Total Loss: {:.6f}\t Average Loss: {:.6f}\t FP: {}\t FN: {}\t FN+FP: {}\t DICE: {:.6f}\t IOU: {:.6f}\t PA: {:.6f}'.format(epoch, valid_loss.item(), valid_loss_mean.item(), FP, FN, FN+FP, IOU, DICE, PA))
		logger.info('         Valid Epoch: {}\t Total loss: {:.6f}\t Average Loss: {:.6f}\t FP: {}\t FN: {}\t FN+FP: {}\t DICE: {:.6f}\t IOU: {:.6f}\t  PA: {:.6f}'.format(epoch, valid_loss.item(), valid_loss_mean.item(), FP, FN, FN+FP, IOU, DICE, PA))


criterion = nn.BCELoss(weight=None, reduction='mean')
optimizer = optim.Adam(net.parameters(), lr = 0.0001)
valid_path = "./ds_ct/valid_ct"
positive_path = "./ds_ct/train_ct/positive"
negative_path = "./ds_ct/train_ct/nopositive"
model_path = "./checkpoint"
log_path = "./log"

# valid_path = "/mnt/hdd1/alluser/zzh/fullData/valid"
# positive_path = "/mnt/hdd1/alluser/zzh/fullData/train/def"
# negative_path = "/mnt/hdd1/alluser/zzh/fullData/train/no_def"

def caculate_FLOPs_and_Params():
	net = segnet(1,1)
	input = torch.randn(1, 1, 672, 752)
	flops, params = profile(net, inputs=(input, ))
	print('flops: ', flops, ' params: ', params)
	return flops, params

def calFlop(path):
	net = segnet(1,1)
	checkpoint = torch.load(path, map_location='cpu' )
	net.load_state_dict(checkpoint['model'])
	stat(net, (3, 1408, 256))

def main(epochs = 101):
	img_Path = negative_path + "/img"
	l = os.listdir(img_Path)
	iterations = 400#len(l)
	print('img_Path: ', img_Path, 'iterations: ', iterations)
	if os.path.exists(model_path):
		dir_list = os.listdir(model_path)
		if len(dir_list) > 0:
			dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
			print('dir_list: ', dir_list)
			last_model_name = model_path + '/' + dir_list[-1]

			checkpoint = torch.load(last_model_name)
			net.load_state_dict(checkpoint['model'])
			# params = net.state_dict().keys()
			# for i, j in enumerate(params):
			# 	print(i, j)
			last_epoch = checkpoint['epoch']
			loss = checkpoint['loss']
			print('load epoch {} succeed! loss: {:.6f} '.format(last_epoch, loss))
		else:
			last_epoch = 0
			print('no saved model, start a new train.')

	else:
		last_epoch = 0
		print('no saved model, start a new train.')


	for epoch in range(last_epoch+1, epochs+1):
		ret, loss = train(net = net, epoch=epoch, iterations=iterations, loss_stop=0.01, positive_path=positive_path, negative_path=negative_path)
		state = {'model':net.state_dict(),'epoch':epoch, 'loss':loss}
		model_name = model_path + '/model_epoch_' + str(epoch) + '.pth'
		torch.save(state, model_name)
		valid(net, epoch, valid_path)

		del_models(model_path)
		if ret:
			break
	print("train.....done.")


if __name__ == '__main__':  
	# calFlop('./checkpoint/model_epoch_99.pth')
	# caculate_FLOPs_and_Params()
	logger = get_logger(log_path)
	# #flops, params = caculate_FLOPs_and_Params()
	# #logger.info('----> flops: {:.6f}, params: {:.6f}'.format(flops, params))
	main()
	# checkpoint = torch.load('./checkpoint/model_epoch_15.pth')
	# net.load_state_dict(checkpoint['model'])
	# model = repvgg_model_convert(net)
	#calFlop('model_epoch_deploy.pth')
