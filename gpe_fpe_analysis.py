import numpy as np
import os
from fast_algs import fast_algs
from scipy.signal import lfilter
import scipy.io as sio
import matplotlib.pyplot as plt

def genRndArParams(order, rootAbsMin, rootAbsMax):
	nRealRoots = order % 2
	nComplexRootPairs = (order-nRealRoots)//2
	# generate the real roots
	realRoots = np.sign(np.random.randn(nRealRoots))*((rootAbsMax-rootAbsMin)*np.random.rand(nRealRoots)+rootAbsMin)
	# generate the complex roots
	complexRoots = ((rootAbsMax-rootAbsMin)*np.random.rand(nComplexRootPairs)+rootAbsMin)*np.exp(1j*np.pi*np.random.rand(nComplexRootPairs))
	# compute AR coefficients
	complexRoots_conj = np.conj(complexRoots)
	b = np.poly(np.concatenate((realRoots,complexRoots,complexRoots_conj)))
	arParameters = -b[1:]
	return arParameters


def generateData(nData, snrDb, pitchOrder, arOrder, pitchBounds):
	# gererate periodic signal
	minPitch = pitchBounds[0]
	maxPitch = pitchBounds[1]/pitchOrder
	truePitch = np.random.rand()*(maxPitch-minPitch)+minPitch
	sinusAmps = np.ones((pitchOrder,)) * pow(0.5, np.arange(pitchOrder))
	sinusPhases = 2*np.pi*np.random.rand(pitchOrder)	
	A = (1j*2*np.pi*truePitch*np.arange(nData).T).reshape(nData,1)
	B = np.arange(1,pitchOrder+1).reshape(1,pitchOrder)
	cisMtx = np.exp(A @ B)
	perSignal = np.real(cisMtx @ (sinusAmps*np.exp(1j*sinusPhases)))

	# generate AR-parameters
	rootAbsMin = 0.5
	rootAbsMax = 0.90
	arParams = genRndArParams(arOrder, rootAbsMin, rootAbsMax)
	arVect = np.concatenate(([1], -arParams))

	# pre-filter the harmonic signal
	perSignal = lfilter(arVect, [1], perSignal)

	# compute the value of the excitation variance
	perPower = perSignal.T @ perSignal / nData
	unitWgn = np.random.randn(nData)
	exVar = 10**(-snrDb/10)*perPower/(unitWgn.T @ unitWgn / nData)
	noisySignal = perSignal + np.sqrt(exVar)*unitWgn

	# post-filter the signal to get the data vector with AR-noise
	dataVector = lfilter([1], arVect, noisySignal)
	return dataVector, truePitch, sinusAmps, arParams, exVar


def computeAsympCrlb(nData, freqs, sinusAmps, arParams, exVar, samplingFreq=1):
	pitchOrder = len(sinusAmps)
	arOrder = len(arParams)
	arPsdVals = exVar / (np.abs(1 - np.exp(-1j*2*np.pi*freqs[:,np.newaxis]*np.arange(1,arOrder+1)/samplingFreq) @ arParams)**2)
	aCrlb = samplingFreq**2*24/((2*np.pi)**2*nData**3*np.sum(np.arange(1,pitchOrder+1)**2*sinusAmps**2/arPsdVals))
	return aCrlb

def computeF0ErrorMetrics(f0_true, f0_est, threshold):
    n_cases, n_mc, n_methods = f0_est.shape
    fpe_mtx = np.full((n_cases, n_mc, n_methods), np.nan)
    gpe_mtx = np.zeros((n_cases, n_mc, n_methods))
    for ii in range(n_cases):
        for jj in range(n_mc):
            jj_f0_true = f0_true[ii, jj]
            for kk in range(n_methods):
                f0_error = jj_f0_true - f0_est[ii, jj, kk]
                if abs(f0_error) > threshold * jj_f0_true:
                    gpe_mtx[ii, jj, kk] = 1
                else:
                    fpe_mtx[ii, jj, kk] = f0_error
    gpe_mtx = 100 * np.mean(gpe_mtx, axis=1).squeeze()
    fpe_mtx = np.sqrt(np.nanmean(fpe_mtx**2, axis=1)).squeeze()
    return fpe_mtx, gpe_mtx	


def gpe_fpe_vs_snr(methods):
	# setup
	np.random.seed(10)
	maxArOrder = 3
	maxPitchOrder = 6
	# maxPitchOrder = 4
	nData = 512
	nMc = 10000
	# nMc = 50
	samplingFreq = 1
	snrDbList = [0,5,10]
	# snrDbList = [0]
	nSnrs = len(snrDbList)
	maxPitch = 0.4
	minPitchList =(np.arange(1, 8.25, 0.25) / nData)
	nPitchBounds = len(minPitchList)
	nMethods = len(methods)
	f0Mtx = np.empty((nPitchBounds, nMc, nMethods+1, nSnrs)) * np.nan
	crlbMtx = np.empty((nPitchBounds, nMc, nSnrs)) * np.nan

	for rr in range(nSnrs):
		snrDb = snrDbList[rr]
		for ii in range(nPitchBounds):
			pitchBounds = [minPitchList[ii], maxPitch]
			for jj in range(nMc):	
				dataVector, truePitch, sinusAmps, arParams, exVar = generateData(nData, snrDb, maxPitchOrder, maxArOrder, pitchBounds)
				freqs = truePitch*np.arange(1,len(sinusAmps)+1)
				crlbMtx[ii,jj,rr] = computeAsympCrlb(nData, freqs, sinusAmps, arParams, exVar, samplingFreq)
				f0Mtx[ii,jj,0,rr] = truePitch										# first position of the true pitch, the second of the pitch of the method
				for kk in range(nMethods):
					if methods[kk] == 'Fast-Exact':
						EstObj = fast_algs(nData, maxPitchOrder, maxArOrder, pitchBounds, 1, 'E')
					elif methods[kk] == 'Naive':
						EstObj = fast_algs(nData, maxPitchOrder, maxArOrder, pitchBounds, 1, 'N')
					elif methods[kk] == 'Fast-Approx1':
						EstObj = fast_algs(nData, maxPitchOrder, maxArOrder, pitchBounds, 1, 'A1')
					# make sure that a pitch is estimated
					EstObj.setValidPitchOrders(np.arange(1,maxPitchOrder+1))
					# estimate the pitch
					estPitch = EstObj.estimate(dataVector, 1e-6)
					f0Mtx[ii,jj,kk+1,rr] = estPitch
			if (ii+1)%2 == 0:
				print("finished {0} pitch simulations out of 29".format(ii+1))

	np.savez(os.path.join(os.getcwd() + '/results/', 'gpe_vs_snr.npz'), f0Mtx=f0Mtx, crlbMtx=crlbMtx, minPitchList=minPitchList, snrDbList=snrDbList, nData=nData, maxArOrder=maxArOrder, maxPitchOrder=maxPitchOrder, nMc=nMc)



def plot_gpe_fpe_vs_snr(methods):
	loaded = np.load(os.path.join(os.getcwd() + '/results/', 'gpe_vs_snr.npz'))
	f0Mtx = loaded['f0Mtx']
	crlbMtx = loaded['crlbMtx']
	minPitchList = loaded['minPitchList']
	snrDbList = loaded['snrDbList']
	nData = loaded['nData']

	nSnrs = len(snrDbList)

	# Create a single figure with three subplots arranged vertically
	fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

	for rr in range(nSnrs):
		fpeMtx, gpeMtx = computeF0ErrorMetrics(f0Mtx[:, :, 0, rr], f0Mtx[:, :, 1:, rr], 0.2)
		if len(methods) == 1:
			fpeMtx = np.concatenate(
				(fpeMtx[:, np.newaxis], (np.sqrt(np.mean(crlbMtx[:, :, rr], axis=1)))[:, np.newaxis]), axis=1)
		else:
			fpeMtx = np.concatenate((fpeMtx, (np.sqrt(np.mean(crlbMtx[:, :, rr], axis=1)))[:, np.newaxis]), axis=1)
		# Use the current subplot
		ax = axs[rr]
		ax.set_title(f'SNR = {snrDbList[rr]} dB')
		ax.plot(minPitchList * nData, gpeMtx, '-o', linewidth=2)
		ax.set_ylabel('GPE [%]')
		ax.grid(True)
		ax.legend(methods)

	# Set a common x-axis label
	ax.set_xlabel('Tf0 min. [cycles/segment]')

	# Save and display the entire figure
	plt.tight_layout()
	plt.savefig('results/gpe_vs_snr.png')
	plt.show()


def gpe_vs_f0_min(methods):
	# setup
	np.random.seed(10)
	maxArOrder = 3
	# maxPitchOrder = 4
	maxPitchOrder = 6
	nData = 512
	# nMc = 50
	nMc = 10000
	pitchBounds = [6/nData, 0.4]
	samplingFreq = 1
	snrDbList = range(-10,11)	#dB
	# snrDbList = range(-10,-6)	#dB
	nSnrs = len(snrDbList)
	nMethods = len(methods);
	f0Mtx = np.empty((nSnrs, nMc, nMethods+1)) * np.nan
	crlbMtx = np.empty((nSnrs, nMc)) * np.nan

	if os.path.exists(os.getcwd() + '/results/gpe_fpe_vs_snr.npz'):
		loaded = np.load(os.path.join(os.getcwd() + '/results/', 'gpe_fpe_vs_snr.npz'))
		curr_snr = loaded['curr_snr']+1
		f0Mtx = loaded['f0Mtx']
		crlbMtx = loaded['crlbMtx']
	else:
		curr_snr = 0

	for ii in range(curr_snr, nSnrs):
		snrDb = snrDbList[ii]
		for jj in range(nMc):
			dataVector, truePitch, sinusAmps, arParams, exVar = generateData(nData, snrDb, maxPitchOrder, maxArOrder, pitchBounds)
			freqs = truePitch*np.arange(1,len(sinusAmps)+1)
			crlbMtx[ii,jj] = computeAsympCrlb(nData, freqs, sinusAmps, arParams, exVar, samplingFreq)
			f0Mtx[ii,jj,0] = truePitch									# first position of the true pitch, the second of the pitch of the method
			for kk in range(nMethods):
				if methods[kk] == 'Fast-Exact':
					EstObj = fast_algs(nData, maxPitchOrder, maxArOrder, pitchBounds, 1, 'E')
				elif methods[kk] == 'Naive':
					EstObj = fast_algs(nData, maxPitchOrder, maxArOrder, pitchBounds, 1, 'N')
				elif methods[kk] == 'Fast-Approx1':
					EstObj = fast_algs(nData, maxPitchOrder, maxArOrder, pitchBounds, 1, 'A1')
				# make sure that a pitch is estimated
				EstObj.setValidPitchOrders(np.arange(1,maxPitchOrder+1))
				# estimate the pitch
				estPitch = EstObj.estimate(dataVector, 1e-6)
				f0Mtx[ii,jj,kk+1] = estPitch
			if (jj+1) % 1000 == 0:
				print("{0} monte carlo simulations out of 10000".format(jj + 1))
		np.savez(os.path.join(os.getcwd() + '/results/', 'gpe_fpe_vs_snr.npz'), curr_snr = ii, f0Mtx=f0Mtx,
			 crlbMtx=crlbMtx, snrDbList=snrDbList, nData=nData, maxArOrder=maxArOrder, maxPitchOrder=maxPitchOrder,
			 nMc=nMc)
		print("finished {0} snrs simulations out of 21".format(ii + 1))
	
	
def plot_gpe_vs_f0_min(methods):
	loaded = np.load(os.path.join(os.getcwd() + '/results/', 'gpe_fpe_vs_snr.npz'))
	f0Mtx = loaded['f0Mtx']
	crlbMtx = loaded['crlbMtx']
	snrDbList = loaded['snrDbList']

	fpeMtx, gpeMtx = computeF0ErrorMetrics(f0Mtx[:,:,0], f0Mtx[:,:,1:], 0.2)
	if len(methods) == 1:
		fpeMtx = np.concatenate((fpeMtx[:,np.newaxis], (np.sqrt(np.mean(crlbMtx, axis=1)))[:,np.newaxis]), axis=1)
	else:
		fpeMtx = np.concatenate((fpeMtx, (np.sqrt(np.mean(crlbMtx, axis=1)))[:,np.newaxis]), axis=1)
		
	fig = plt.figure(0)
	plt.subplot(2, 1, 1)
	plt.semilogy(snrDbList, fpeMtx, '-o', linewidth=2)
	plt.xlabel('SNR [dB]')
	plt.ylabel('FPE [cycles/sample]')
	plt.grid(True)
	plt.legend(methods + ['CRLB'])

	plt.subplot(2, 1, 2)
	plt.plot(snrDbList, gpeMtx, '-o', linewidth=2)
	plt.xlabel('SNR [dB]')
	plt.ylabel('GPE [%]')
	plt.grid(True)
	plt.legend(methods)
	plt.savefig('results/gpe_fpe_vs_snr')
	plt.show()
