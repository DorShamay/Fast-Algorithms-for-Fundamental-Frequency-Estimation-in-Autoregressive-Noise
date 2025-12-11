from utils import *
import numpy as np
from timeit import timeit

class fast_algs:
	def __init__(self, nData, maxPitchOrder, maxArOrder, pitchBounds, samplingFreq, alg, benchmarkAlg=False):
		self.gammaTpH=np.nan;
		self.gammaTmH=np.nan;

		if alg == 'E':
			self.alg = 'E'
		elif alg == 'A1':
			self.alg = 'A1'
		elif alg == 'N':
			self.alg = 'N'
		else:
			raise ValueError('You have not selected a valid algorithm (either E, A1, or N).')

		minNDft = 5*nData*maxPitchOrder
		self.nDft = 2**np.ceil(np.log2(minNDft))
		self.samplingFreq = samplingFreq; # Hz
		self.pitchResolution = self.samplingFreq/self.nDft
		self.pitchBounds = pitchBounds
		self.maxPitchOrder = maxPitchOrder
		self.validPitchOrders = list(range(1, maxPitchOrder+1))
		self.maxArOrder = maxArOrder
		self.nData = nData
		self.startIndex = -(self.nData+self.maxArOrder-1)/2

		self.benchmarkAlg = benchmarkAlg
		if pitchBounds[0] < samplingFreq/self.nData:
			print('The lower pitch bound is set lower than 1 periods/segment. Inaccurate results will most likely be produced!')
		if maxPitchOrder == 0:
			raise ValueError('The maximum pitch order is set to 0, and this is not supported. Please use the lpc function instead.')

		self.fullPitchGrid, self.dftRange = computePitchGrid(self.nDft, self.pitchBounds, self.maxPitchOrder, self.samplingFreq)			# utils
		if self.alg == 'E' or self.alg == 'A':
			self.gammaTpH, self.gammaTmH = self.dataIndependentStep(self.nData+self.maxArOrder)

	def setValidPitchOrders(self, validPitchOrders):
		self.validPitchOrders = sorted(validPitchOrders)
	
	def estimate(self, dataVector, refinementTol=None):
#		pass
		# store the data
		self.dataVector = dataVector
		# zero-pad the data vector (we use the autocorrelation method)
		dataVector = np.concatenate((dataVector, np.zeros(self.maxArOrder)))
		if refinementTol is None:
			refinementTol = self.samplingFreq/self.nDft
		# Compute the objective on a grid
		self.objValNoPitch, self.objValPitch = self.computeObjOnGrid(dataVector)
		# Estimate the model order
		self.estArOrder, self.estPitchOrder = self.estimateModelOrders()
		# estimate the pitch, harmonic amplitudes and phases, the AR parameters, and the excitation variance
		self.estPitch, self.estSinusAmps, self.estSinusPhases, self.estArParams, self.estExVar = self.estimatePitchArParameters(dataVector, self.estPitchOrder, self.estArOrder, refinementTol)
		# return the pitch
		return self.estPitch
		
	def computeModelledSpectrum(self, nDft=None):
		if self.objValNoPitch is None:
			raise ValueError('Please run the estimator (self.estimate) first!')
		if nDft is None:
			nDft = self.nDft
		# AR spectrum
		arPsd = self.estExVar / (self.samplingFreq * abs(np.fft.fft(np.concatenate(([1], -self.estArParams.squeeze())), nDft, axis=0)) ** 2)
		# pitch spectrum
		pitchPsd = np.zeros((nDft, 1))
		pitchDftIdx = round(self.estPitch * nDft / self.samplingFreq)
		pitchPsd[1 + pitchDftIdx * np.arange(1, self.estPitchOrder + 1) - 1] = len(self.dataVector) * self.estSinusAmps ** 2 / self.samplingFreq
		pitchPsd[nDft - 1 - pitchDftIdx * np.arange(1, self.estPitchOrder + 1) - 1] = len(self.dataVector) * self.estSinusAmps ** 2 / self.samplingFreq
		pitchPsd = pitchPsd / 4
		# total spectrum
		modelledSpectrum = pitchPsd.squeeze() + arPsd
		freqVector = self.samplingFreq * np.arange(nDft) / nDft
		return modelledSpectrum, arPsd, pitchPsd, freqVector
		
	def computePrewhitenedSpectrum(self, nDft=None):
		if nDft is None:
			nDft = self.nDft
		freqVector = self.samplingFreq*np.arange(nDft)/nDft
		prewhitenedData = np.fft.ifft(np.fft.fft(self.dataVector, nDft)*
			np.fft.fft(np.concatenate(([1], -self.estArParams.squeeze())), nDft))
		preWhitenedSpectrum = abs(np.fft.fft(prewhitenedData, nDft))**2/(self.samplingFreq*len(self.dataVector))
		return preWhitenedSpectrum, freqVector
	
	def computeObjOnGrid(self, dataVector):
		if self.alg == 'E':
			objValNoPitch, objValPitch = self.computeArPitchFastExactObj(dataVector)
			if self.benchmarkAlg:
				f = lambda: self.computeArPitchFastExactObj(dataVector)
				self.compTime = timeit(f, number=10)
				# self.compTime = timeit(f, number=4)
		elif self.alg == 'A1':
			objValNoPitch, objValPitch = self.computeArPitchFastApprox1Obj(dataVector)
			if self.benchmarkAlg:
				f = lambda: self.computeArPitchFastApprox1Obj(dataVector)
				self.compTime = timeit(f, number=10)
				# self.compTime = timeit(f, number=4)
		elif self.alg == 'N':
			objValNoPitch, objValPitch = self.computeArPitchExactObj(dataVector)
			if self.benchmarkAlg:
				f = lambda: self.computeArPitchExactObj(dataVector)
				self.compTime = timeit(f, number=10)
				# self.compTime = timeit(f, number=4)
		return objValNoPitch, objValPitch
		
		
	def computeArPitchFastExactObj(self, dataVector):
		# no periodic signal is present
		covVct = computeCovVct(dataVector, self.maxArOrder, self.nData)
		objValNoPitch, cholVct = genSchurToeplitzAlg(covVct)
		# periodic signal is present
		dftData = np.fft.fft(dataVector, int(self.nDft))

		for q in range(1, self.maxPitchOrder + 1):
			qShiftedDftIdx = q * np.arange(int(self.dftRange[q - 1][0]), int(self.dftRange[q - 1][1] + 1)) - int(
				self.dftRange[q - 1][0]) + 1
			nPitches = len(qShiftedDftIdx)
			f0Idx = np.array(range(0, nPitches))
			if q == 1:
				# initialise array
				objValPitch = np.empty((self.maxArOrder + 1, nPitches, self.maxPitchOrder))
				objValPitch[:] = np.nan
				neededDftBins = np.arange(int(self.dftRange[q - 1][0]),
										  int(max((np.arange(1, self.maxPitchOrder + 1) * self.dftRange[:, 1])) + 1))

				A = -1j * ((np.arange(0, self.maxArOrder + 1))[:, np.newaxis] + self.startIndex)
				B = 2 * np.pi * neededDftBins[np.newaxis, :] / self.nDft
				C = np.ones((self.maxArOrder + 1, 1))
				D = dftData[neededDftBins][np.newaxis, :]
				Yhat = np.exp(A @ B) * (C @ D)

				dftDataTpH = np.empty((self.maxPitchOrder, nPitches, self.maxArOrder + 1))
				dftDataTpH[:] = np.nan
				dftDataTmH = np.empty((self.maxPitchOrder, nPitches, self.maxArOrder + 1))
				dftDataTmH[:] = np.nan
				if self.maxArOrder > 0:
					qRho = covVct[1:][:, np.newaxis] @ np.ones((1, nPitches))
					# Cholesky factor with stacked columns
					cholVct = cholVct[:, np.newaxis] @ np.ones((1, nPitches))
			else:
				# remove unneeded entries
				if self.maxArOrder > 0:
					qRho = qRho[:, f0Idx]
					cholVct = cholVct[:, f0Idx]
			lambdaTpH = np.empty((self.maxArOrder + 1, nPitches))
			lambdaTpH[:] = np.nan
			lambdaTmH = np.empty((self.maxArOrder + 1, nPitches))
			lambdaTmH[:] = np.nan
			for p in range(0, self.maxArOrder + 1):
				dftDataTpH[q - 1, f0Idx, p] = np.real(Yhat[p, qShiftedDftIdx - 1])
				dftDataTmH[q - 1, f0Idx, p] = -np.imag(Yhat[p, qShiftedDftIdx - 1])
				if p == 0 and q == 1:
					if self.gammaTpH[p].ndim < 2:
						self.gammaTpH[p] = self.gammaTpH[p][np.newaxis, :]
						self.gammaTmH[p] = self.gammaTmH[p][np.newaxis, :]
				lambdaTpH[p, :] = np.sum(self.gammaTpH[q - 1][0:q, f0Idx] * dftDataTpH[0:q, f0Idx, p], 0) / np.sqrt(
					self.nData)
				lambdaTmH[p, :] = np.sum(self.gammaTmH[q - 1][0:q, f0Idx] * dftDataTmH[0:q, f0Idx, p], 0) / np.sqrt(
					self.nData)
				if p == 0:
					if q == 1:
						objValPitch[0, :, 0] = objValNoPitch[0] - lambdaTpH[0, :] ** 2 - lambdaTmH[0, :] ** 2
					else:
						objValPitch[0, f0Idx, q - 1] = objValPitch[0, f0Idx, q - 2] - lambdaTpH[0, :] ** 2 - lambdaTmH[
																											 0, :] ** 2
				else:  # p > 0
					qRho[p - 1, :] = qRho[p - 1, :] - lambdaTpH[0, :] * lambdaTpH[p, :] - lambdaTmH[0, :] * lambdaTmH[p,
																											:]
			if self.maxArOrder > 0:
				# update Cholesky decomposition using two rank one downdating steps
				cholVct = cholDowndating(cholVct, lambdaTpH[1:, :])
				cholVct = cholDowndating(cholVct, lambdaTmH[1:, :])
				# recursively compute the objective using forward substitution
				objValPitch[1:p + 2, f0Idx, q - 1] = (np.ones((self.maxArOrder, 1)) @ objValPitch[0, f0Idx, q - 1][np.newaxis,:]) - orderRecursiveForwardSubstitution(cholVct, qRho)

		return objValNoPitch, objValPitch
	
	def computeArPitchFastApprox1Obj(self, dataVector):
		# no periodic signal is present
		objValNoPitch, rho = ldr(dataVector, self.maxArOrder, self.nData)
		# periodic signal is present
		phi = 2 * abs((np.fft.fft(dataVector, int(self.nDft))) ** 2) / (self.nData * (self.nData + self.maxArOrder))
		for q in range(1, self.maxPitchOrder + 1):
			qDftIdx = q * np.array(range(int(self.dftRange[q - 1][0]), int(self.dftRange[q - 1][1]) + 1))
			qShiftedDftIdx = qDftIdx - int(self.dftRange[q - 1][0])
			nPitches = len(qDftIdx)
			f0Idx = np.array(range(0, nPitches))
			if q == 1:
				# initialise array
				objValPitch = np.empty((self.maxArOrder + 1, nPitches, self.maxPitchOrder))
				objValPitch[:] = np.nan
				objValPitch[0, :, 0] = objValNoPitch[0] - phi[qDftIdx]
				neededDftBins = np.arange(int(self.dftRange[q - 1][0]),
										  int(max((np.arange(1, self.maxPitchOrder + 1) * self.dftRange[:, 1])) + 1))
				Phi = np.cos(
					np.arange(1, self.maxArOrder + 1).reshape(-1, 1) * 2 * np.pi * neededDftBins / self.nDft) * (
								  np.ones((self.maxArOrder, 1)) @ phi[neededDftBins][np.newaxis, :])
				rho = rho @ np.ones((1, nPitches)) - Phi[:, qShiftedDftIdx]
			else:
				# the +1 is to compensate for MATLAB's indexing
				objValPitch[0, f0Idx, q - 1] = objValPitch[0, f0Idx, q - 2] - phi[qDftIdx]
				rho = rho[:, f0Idx] - Phi[:, qShiftedDftIdx]
			# we run the recursion in the reversed version beta to avoid one flipud
			if self.maxArOrder > 0:
				betaRev = rho[0, :] / objValPitch[0, f0Idx, q - 1]
				objValPitch[1, f0Idx, q - 1] = objValPitch[0, f0Idx, q - 1] * (1 - betaRev ** 2)
				betaRev = betaRev[np.newaxis, :]
				for p in range(1, self.maxArOrder):
					nu = (rho[p, :] - np.sum(betaRev * rho[:p, :], axis=0)) / objValPitch[p, f0Idx, q - 1]
					nu = nu[np.newaxis, :]
					objValPitch[p + 1, f0Idx, q - 1] = objValPitch[p, f0Idx, q - 1] * (1 - nu ** 2)
					if p < self.maxArOrder - 1:
						betaRev = np.concatenate((nu, betaRev - np.flipud(betaRev) * (np.ones((p, 1)) @ nu)), axis=0)

		return objValNoPitch, objValPitch
		
	def computeArPitchExactObj(self, dataVector):
		for ii in range(0, self.maxPitchOrder + 1):
			if ii == 0:
				iiPitchGrid = np.empty((1, 1)) * np.nan
				objValNoPitch = np.empty((self.maxArOrder + 1, 1)) * np.nan
			else:
				# the +1 is to compensate for MATLAB's indexings
				iiPitchGrid = self.fullPitchGrid[int(self.dftRange[ii - 1, 0]):int(self.dftRange[ii - 1, 1]) + 1]
				nPitches = len(iiPitchGrid)
				if ii == 1:
					objValPitch = np.empty((self.maxArOrder + 1, nPitches, self.maxPitchOrder)) * np.nan

			for jj in range(0, self.maxArOrder + 1):
				if ii == 0:
					objValNoPitch[jj], _ = computeObjectiveNaively(dataVector, iiPitchGrid, ii, jj, self.startIndex,
																   self.nData)
				else:
					objValPitch[jj, 0:nPitches, ii - 1], _ = computeObjectiveNaively(dataVector, iiPitchGrid, ii, jj,
																					 self.startIndex, self.nData)
		return objValNoPitch, objValPitch
	
	def estimateModelOrders(self):
		self.bayesFactor = pitchArBicModelComparison(self.nData, self.objValNoPitch, self.objValPitch, self.validPitchOrders)
		idxPitch, idxAr = np.unravel_index(np.argmax(self.bayesFactor), self.bayesFactor.shape)
		estPitchOrder = idxPitch
		estArOrder = idxAr
		return estArOrder, estPitchOrder
	
	def estimatePitchArParameters(self, dataVector, pitchOrder, arOrder, refinementTol):
		if pitchOrder == 0:
			estPitch = float('nan')
		else:
			pitchIdx = np.nanargmin(self.objValPitch[arOrder,:, pitchOrder-1])
			coarsePitchEst = self.fullPitchGrid[int(self.dftRange[pitchOrder-1,0]+pitchIdx)]
			# refine the pitch estimate if needed
			if refinementTol < self.samplingFreq/self.nDft:
				# add refinement here if necessary!!!
				pitchLimits = coarsePitchEst/self.samplingFreq+[-1, 1]/self.nDft
				
				costFunction = lambda f0: computeObjectiveNaively(dataVector, f0, pitchOrder, arOrder, self.startIndex, self.nData)
				(f0l, f0u) = fibonacciSearch(costFunction, pitchLimits[0], pitchLimits[1], refinementTol)
				estPitch = (f0l+f0u)*self.samplingFreq/2
			else:
				estPitch = coarsePitchEst
		estExVar, estLinParams = computeObjectiveNaively(dataVector, estPitch/self.samplingFreq, pitchOrder, arOrder, self.startIndex, self.nData)
		if arOrder == 0:
			estArParams = np.zeros(0)
		else:
			estArParams = estLinParams[:arOrder]
		if pitchOrder == 0:
			estSinusAmps = np.zeros(0)
			estSinusPhases = np.zeros(0)
		else:
			estSinusAmps, estSinusPhases = self.estimatePitchAmpsAndPhases(estArParams, estLinParams[arOrder:], estPitch)
				
		return estPitch, estSinusAmps, estSinusPhases, estArParams, estExVar
	
	def estimatePitchAmpsAndPhases(self, estArParams, shapedPitchLinearParameters, estPitch):
		arOrder = len(estArParams)
		pitchOrder = len(shapedPitchLinearParameters)//2
		shapedPitchComplexLinearParameters = np.reshape(shapedPitchLinearParameters, (pitchOrder, 2)) @ [1, -1j]
		if arOrder == 0:
			pitchComplexLinearParameters = shapedPitchComplexLinearParameters
		else:
			pitchComplexLinearParameters = shapedPitchComplexLinearParameters[:, np.newaxis]/(1-np.exp(-1j*(2*np.pi*estPitch/self.samplingFreq)*np.arange(1, pitchOrder+1)[:, np.newaxis] @ np.arange(1, arOrder+1)[np.newaxis, :]) @ estArParams)

		estSinusPhases = np.angle(pitchComplexLinearParameters)
		estSinusAmps = np.abs(pitchComplexLinearParameters)
		
		return estSinusAmps, estSinusPhases

	def dataIndependentStep(self, nData):
		ccVectors = computeRealSinusoidalCorrVector(nData, self.fullPitchGrid[int(self.dftRange[0, 0]):int(self.dftRange[0, 1])+1] / self.samplingFreq, self.maxPitchOrder)
		gammaTpH, gammaTmH = computeGamma(self.maxPitchOrder, self.dftRange, ccVectors)
		return gammaTpH, gammaTmH

