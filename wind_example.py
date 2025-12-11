import numpy as np
from scipy.io import wavfile
import math
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import os
from fast_algs import fast_algs

def create_noisy_speech_files():
	# set resampling frequency and load speech signal
	resamplingFreq = 16000
	speechSamplingFreq, rawSpeech = wavfile.read('wind_data/female_speech(48k).wav')
	speech = np.interp(np.arange(0, len(rawSpeech), speechSamplingFreq/resamplingFreq), np.arange(0, len(rawSpeech)), rawSpeech)
	speechPower = np.sum(speech**2)

	# load and resample noise signal
	noiseSamplingFreq, rawNoise = wavfile.read('wind_data/mic2_10cm.wav')
	noiseLong = np.interp(np.arange(0, len(rawNoise), noiseSamplingFreq/resamplingFreq), np.arange(0, len(rawNoise)), rawNoise)

	# truncate noise to have the same length as speech
	noise = noiseLong[:len(speech)]

	noisePower = np.sum(noise**2)

	# define desired SNR values in dB
	snrListDb = np.concatenate((np.arange(-10, 30, 5), [100]))

	# generate audio files for each SNR value
	for ii in range(len(snrListDb)):
		# calculate gain to achieve desired SNR
		noiseGain = np.sqrt(10**(-snrListDb[ii]/10) * speechPower / noisePower)
		
		# generate noisy speech and normalize it
		noisySpeech = speech + noiseGain * noise
		if np.max(np.abs(noisySpeech)) >= 1:
			noisySpeech = noisySpeech / (np.max(np.abs(noisySpeech))+2.2204e-16)
		
		# write noisy speech to file
		filename = 'wind_data/speechInWind_{}dB.wav'.format(snrListDb[ii])
		wavfile.write(filename, resamplingFreq, noisySpeech)


def speech_wind_example():
	# load the data
	samplingFreq, rawData = wavfile.read('wind_data/speechInWind_5dB.wav')
	cleanSamplingFreq, rawCleanData = wavfile.read('wind_data/speechInWind_100dB.wav')
	resamplingFreq = 16000 # Hz

	resampledData = rawData
	resampledCleanData = rawCleanData
	
	# set up the segment-by-segment processing (no overlap)
	nData = len(resampledData)
	segmentTime = 0.025 # seconds
	segmentLength = round(segmentTime*resamplingFreq)
	nSegments = math.floor(nData/segmentLength)
	
	# set up the analysis
	maxArOrder = 10
	maxPitchOrder = 15
	pitchBounds = [60, 400] # Hz
	
	# do the analysis
	methods = ['Clean', 'Fast-Exact', 'Exact-non-AR']
	nMethods = len(methods)
	estArOrder = np.empty((nMethods, nSegments)) * np.nan
	estPitchOrder = np.empty((nMethods, nSegments)) * np.nan
	estPitchHz = np.empty((nMethods, nSegments)) * np.nan
	exampleSegmentNo = 50											# depends on the length of the input audio file
	exampleNDft = 2048
	
	for ii in range(nMethods):
		print('Processing method {0} of {1}'.format(str(ii+1), str(nMethods)))
		if methods[ii] == 'Clean':
			Est = fast_algs(segmentLength, maxPitchOrder, maxArOrder, pitchBounds, resamplingFreq, 'E')
			obsData = resampledCleanData
		elif methods[ii] == 'Fast-Exact':
			Est = fast_algs(segmentLength, maxPitchOrder, maxArOrder, pitchBounds, resamplingFreq, 'E')
			obsData = resampledData
		elif methods[ii] == 'Exact-non-AR':
			Est = fast_algs(segmentLength, maxPitchOrder, 0, pitchBounds, resamplingFreq, 'E')
			obsData = resampledData
		
		
		Est.setValidPitchOrders(np.array([0] + list(range(3, maxPitchOrder+1))))
		idx = np.array(range(0, segmentLength))
		
		for jj in range(nSegments):
			print('Processing segment {0} of {1}'.format(str(jj+1), str(nSegments)))
			dataSegment = obsData[idx]
			# estimate parameters and orders
			estPitchHz[ii,jj] = Est.estimate(dataSegment)
			estPitchOrder[ii,jj] = Est.estPitchOrder
			estArOrder[ii,jj] = Est.estArOrder
			if (methods[ii] =='Fast-Exact') and (jj == exampleSegmentNo-1):
				modelledSpectrum, _, _, _ = Est.computeModelledSpectrum(exampleNDft)
				preWhitenedSpectrum, freqVector = Est.computePrewhitenedSpectrum(exampleNDft)
				perEst = abs(np.fft.fft(dataSegment, exampleNDft))**2/(Est.samplingFreq*segmentLength)
			idx = idx + segmentLength
		
	
	np.savez(os.path.join('results/', 'wind_params.npz'), segmentTime=segmentTime, samplingFreq=samplingFreq, nSegments=nSegments, rawData=rawData, segmentLength=segmentLength, resamplingFreq=resamplingFreq, estPitchOrder=estPitchOrder, estArOrder=estArOrder, estPitchHz=estPitchHz, methods=methods, freqVector=freqVector, perEst=perEst, preWhitenedSpectrum=preWhitenedSpectrum, modelledSpectrum=modelledSpectrum)
	

def plot_speech_wind_example():
	loaded = np.load(os.path.join('results/', 'wind_params.npz'))
	segmentTime = loaded['segmentTime']
	samplingFreq = loaded['samplingFreq']
	nSegments = loaded['nSegments']
	rawData = loaded['rawData']
	segmentLength = loaded['segmentLength']
	resamplingFreq = loaded['resamplingFreq']
	estPitchHz = loaded['estPitchHz']
	methods = loaded['methods']
	freqVector = loaded['freqVector']
	perEst = loaded['perEst']
	preWhitenedSpectrum = loaded['preWhitenedSpectrum']
	modelledSpectrum = loaded['modelledSpectrum']

	# spectrogram settings
	segmentLengthRawData = round(segmentTime*samplingFreq)
	win = np.hanning(segmentLengthRawData)
	nOverlap = round(segmentLengthRawData*3/4)
	nFft = 4*segmentLengthRawData

	# do the plotting
	timeMidSegments = (np.arange(nSegments)+1)*segmentTime-segmentTime/2
	f, t, S = spectrogram(rawData, fs=samplingFreq, window=win, nperseg=segmentLengthRawData, noverlap=nOverlap, nfft=nFft)
	spec = 10*np.log10(S**2/(samplingFreq*segmentLength))
	plt.figure(1)
	im = plt.imshow(spec, aspect='auto', origin='lower', extent=[t[0], t[-1], f[0], f[-1]], cmap='gray', clim=[-280, -130])
	plt.xlabel('Time [s]')
	plt.ylabel('Freq. [Hz]')
	plt.ylim([0, resamplingFreq/2])
	plt.colorbar(im)
	plt.savefig('results/spectrogram.png', dpi=300, bbox_inches='tight')

	# plot estimated pitch frequency
	plt.figure(2)
	plt.plot(timeMidSegments, estPitchHz[0, :], '-', linewidth=2)
	plt.plot(timeMidSegments[np.newaxis,:].T, estPitchHz[1:, :].T, 'o', linewidth=2)
	plt.legend(methods)
	plt.xlabel('Time [s]')
	plt.ylabel('Freq. [Hz]')
	plt.savefig('results/estimated_pitch_frequency.png')

	# plot periodogram, prewhitened periodogram, and modelled spectrum
	plt.figure(3)
	plt.plot(freqVector, 10*np.log10(perEst), linewidth=2)
	plt.plot(freqVector, 10*np.log10(preWhitenedSpectrum), linewidth=2)
	plt.plot(freqVector, 10*np.log10(modelledSpectrum), linewidth=2)
	plt.legend(['Periodogram', 'Prewhitened periodogram', 'Modelled spectrum'])
	plt.xlabel('Freq. [Hz]')
	plt.ylabel('PSD [dB/Hz]')
	plt.xlim([0, resamplingFreq/2])
	plt.ylim([-100, -30])
	plt.savefig('results/periodogram.png')
	plt.show()
		
		
		
		
		
		
		
		


