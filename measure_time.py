from fast_algs import fast_algs
# from utils import *
import numpy as np
import os
import matplotlib.pyplot as plt

def pitch_order_comp_time(methods):
	# setup
	maxArOrder = 3
	nData = 512
	pitchBounds = [1.5/nData, 0.4]
	nMc = 50
	# nMc = 2
	samplingFreq = 1
	maxPitchOrderList = range(1,16)
	# maxPitchOrderList = range(1,4)
	nPitchOrders = len(maxPitchOrderList)
	nMethods = len(methods)
	timingMtx = np.empty((nPitchOrders, nMc, nMethods)) * np.nan

	if os.path.exists(os.getcwd() + '/results/time_vs_pitch_order.npz'):
		loaded = np.load(os.path.join(os.getcwd() + '/results/', 'time_vs_pitch_order.npz'))
		curr_pitch = loaded['curr_pitch']+1
		print(curr_pitch)
		timingMtx = loaded['timingMtx']
	else:
		curr_pitch = 0

	for ii in range(curr_pitch, nPitchOrders):
		maxPitchOrder = maxPitchOrderList[ii]
		for jj in range(nMethods):
			if methods[jj] == 'Fast-Exact':
				EstObj = fast_algs(nData, maxPitchOrder, maxArOrder, pitchBounds, 1, 'E', True)
			elif methods[jj] == 'Fast-Approx1':
				EstObj = fast_algs(nData, maxPitchOrder, maxArOrder, pitchBounds, 1, 'A1', True)
			elif methods[jj] == 'Naive':
				EstObj = fast_algs(nData, maxPitchOrder, maxArOrder, pitchBounds, 1, 'N', True)
			
			# make sure that a pitch is estimated
			EstObj.setValidPitchOrders(np.arange(1,maxPitchOrder+1))
			for kk in range(nMc):
				dataVector = np.random.randn(nData)

				estPitch = EstObj.estimate(dataVector)
				timingMtx[ii,kk,jj] = EstObj.compTime
				if (kk + 1) % 5 == 0 and jj == 2:
					print("{0} monte carlo simulations out of 50".format(kk + 1))
		np.savez(os.path.join(os.getcwd() + '/results/', 'time_vs_pitch_order.npz'), curr_pitch=ii, timingMtx=timingMtx,
				 nData=nData, maxArOrder=maxArOrder, maxPitchOrderList=maxPitchOrderList, nMc=nMc)
		print("finished {0} pitch simulations out of 15".format(ii + 1))
	

def plot_pitch_order_comp_time(methods):
	loaded = np.load(os.path.join(os.getcwd()+'/results/', 'time_vs_pitch_order.npz'))
	timingMtx = loaded['timingMtx']
	maxPitchOrderList = loaded['maxPitchOrderList']
		
	timing2plot = np.min(timingMtx, axis=1).squeeze()	
		
	fig = plt.figure(0)
	plt.semilogy(maxPitchOrderList, timing2plot, '-o', linewidth=2)
	plt.xlabel('Max. pitch order')
	plt.ylabel('Computation time [s]')
	plt.grid(True)
	plt.legend(methods)
	plt.savefig('results/time_vs_pitch_order')
	plt.show()

def Ar_order_comp_time(methods):
	# setup
#	np.random.seed(10)
	maxPitchOrder = 3
	nData = 512
	pitchBounds = [1.5/nData, 0.4]
	nMc = 50
	# nMc = 2
	samplingFreq = 1
	maxArOrderList = range(1,16)
	# maxArOrderList = range(1, 4)
	nArOrders = len(maxArOrderList)
	nMethods = len(methods)
	timingMtx = np.empty((nArOrders, nMc, nMethods)) * np.nan

	if os.path.exists(os.getcwd() + '/results/time_vs_AR_order.npz'):
		loaded = np.load(os.path.join(os.getcwd() + '/results/', 'time_vs_AR_order.npz'))
		curr_ar = loaded['curr_ar']+1
		timingMtx = loaded['timingMtx']
	else:
		curr_ar = 0

	for ii in range(curr_ar, nArOrders):
		maxArOrder = maxArOrderList[ii]
		for jj in range(nMethods):
			if methods[jj] == 'Fast-Exact':
				EstObj = fast_algs(nData, maxPitchOrder, maxArOrder, pitchBounds, 1, 'E', True)
			elif methods[jj] == 'Fast-Approx1':
				EstObj = fast_algs(nData, maxPitchOrder, maxArOrder, pitchBounds, 1, 'A1', True)
			elif methods[jj] == 'Naive':
				EstObj = fast_algs(nData, maxPitchOrder, maxArOrder, pitchBounds, 1, 'N', True)
			
			# make sure that a pitch is estimated
			EstObj.setValidPitchOrders(np.arange(1,maxPitchOrder+1))
			for kk in range(nMc):
				dataVector = np.random.randn(nData)
				estPitch = EstObj.estimate(dataVector)
				timingMtx[ii,kk,jj] = EstObj.compTime
				if (kk + 1) % 5 == 0 and jj == 2:
					print("{0} monte carlo simulations out of 50".format(kk + 1))

		np.savez(os.path.join(os.getcwd() + '/results/', 'time_vs_AR_order.npz'), curr_ar=ii, timingMtx=timingMtx, nData=nData, maxArOrderList=maxArOrderList, maxPitchOrder=maxPitchOrder, nMc=nMc)
		print("finished {0} AR simulations out of 15".format(ii + 1))
	

def plot_Ar_order_comp_time(methods):
	loaded = np.load(os.path.join(os.getcwd()+'/results/', 'time_vs_AR_order.npz'))
	timingMtx = loaded['timingMtx']
	maxArOrderList = loaded['maxArOrderList']
		
	timing2plot = np.min(timingMtx, axis=1).squeeze()	
		
	fig = plt.figure(0)
	plt.semilogy(maxArOrderList, timing2plot, '-o', linewidth=2)
	plt.xlabel('Max. AR order')
	plt.ylabel('Computation time [s]')
	plt.grid(True)
	plt.legend(methods)
	plt.savefig('results/time_vs_AR_order')
	plt.show()
	
def segment_length_comp_time(methods):
	# setup
	maxArOrder = 3
	maxPitchOrder = 3
	# nMc = 2
	nMc = 50
	samplingFreq = 1
	nDataList = [2**i for i in range(6,13)]
	# nDataList = [2**i for i in range(6,9)]
	nDataLengths = len(nDataList)
	nMethods = len(methods)
	timingMtx = np.empty((nDataLengths, nMc, nMethods)) * np.nan

	if os.path.exists(os.getcwd() + '/results/time_vs_segment_length.npz'):
		loaded = np.load(os.path.join(os.getcwd() + '/results/', 'time_vs_segment_length.npz'))
		curr_data_len = loaded['curr_data_len']+1
		timingMtx = loaded['timingMtx']
	else:
		curr_data_len = 0

	for ii in range(curr_data_len, nDataLengths):
		nData = nDataList[ii]
		pitchBounds = [1.5/nData, 0.4]
		for jj in range(nMethods):
			if methods[jj] == 'Fast-Exact':
				EstObj = fast_algs(nData, maxPitchOrder, maxArOrder, pitchBounds, 1, 'E', True)
			elif methods[jj] == 'Fast-Approx1':
				EstObj = fast_algs(nData, maxPitchOrder, maxArOrder, pitchBounds, 1, 'A1', True)
			elif methods[jj] == 'Naive':
				EstObj = fast_algs(nData, maxPitchOrder, maxArOrder, pitchBounds, 1, 'N', True)
			
			# make sure that a pitch is estimated
			EstObj.setValidPitchOrders(np.arange(1,maxPitchOrder+1))
			for kk in range(nMc):
				dataVector = np.random.randn(nData)
				estPitch = EstObj.estimate(dataVector)
				timingMtx[ii,kk,jj] = EstObj.compTime
				if (kk + 1) % 5 == 0 and jj == 2:
					print("{0} monte carlo simulations out of 50".format(kk + 1))
		np.savez(os.path.join(os.getcwd() + '/results/', 'time_vs_segment_length.npz'), curr_data_len = ii, timingMtx=timingMtx,
			 nDataList=nDataList, maxArOrder=maxArOrder, maxPitchOrder=maxPitchOrder, nMc=nMc)
		print("finished {0} segment length simulations out of 7".format(ii + 1))

def plot_segment_length_comp_time(methods):
	loaded = np.load(os.path.join(os.getcwd()+'/results/', 'time_vs_segment_length.npz'))
	timingMtx = loaded['timingMtx']
	nDataList = loaded['nDataList']
		
	timing2plot = np.min(timingMtx, axis=1).squeeze()	
		
	fig = plt.figure(0)
	plt.semilogy(nDataList, timing2plot, '-o', linewidth=2)
	plt.xlabel('Segment length [samples]')
	plt.ylabel('Computation time [s]')
	plt.grid(True)
	plt.legend(methods)
	plt.savefig('results/time_vs_segment_length')
	plt.show()
