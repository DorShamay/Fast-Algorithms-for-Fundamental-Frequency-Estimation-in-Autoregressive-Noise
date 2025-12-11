import numpy as np
import copy
from scipy.linalg import toeplitz
import math

def computePitchGrid(nDft, pitchBounds, maxPitchOrder, samplingFreq):
	fullPitchGrid = samplingFreq*(np.arange(nDft))/nDft
	minDftIdx = max(0, round(pitchBounds[0]*nDft/samplingFreq))
	maxDftIdx = round(min(nDft/2-1, pitchBounds[1]*nDft/samplingFreq))
	dftRange = np.empty((maxPitchOrder, 2)) * np.nan
	dftRange[0,:] = [minDftIdx,maxDftIdx]
	for ii in range(1, maxPitchOrder):
		maxDftIdx = round(min(nDft/(2*(ii+1))-1, pitchBounds[1]*nDft/samplingFreq))
		dftRange[ii,:] = [minDftIdx, maxDftIdx]
	return fullPitchGrid, dftRange

def computeRealSinusoidalCorrVector(nData, pitchGrid, pitchOrder):
	nPitches = len(pitchGrid)
	orderPitchMtx = np.arange(1, 2*pitchOrder+1, dtype=float).reshape(-1,1)*np.pi*pitchGrid.reshape(1,-1)
	crossCorrVectors = np.concatenate([nData*np.ones((1, nPitches)), np.sin(nData*orderPitchMtx)/np.sin(orderPitchMtx)])/2
	return crossCorrVectors
	
def computeGamma(pitchOrder, dftRange, crossCorrelationVectors):
	nPitches = dftRange[0,1] - dftRange[0,0] + 1
	nPitches = int(nPitches)
	gammaTpH = []
	gammaTmH = []
	if pitchOrder == 1:
		a1 = crossCorrelationVectors[1, :]
		a1 = a1[np.newaxis, :]
		a2 = a1
	else:
		a1 = crossCorrelationVectors[1:pitchOrder, :] + np.concatenate((np.zeros((1,nPitches)), crossCorrelationVectors[2:pitchOrder, :]), axis=0)
		a2 = crossCorrelationVectors[1:pitchOrder, :] - np.concatenate((np.zeros((1,nPitches)), crossCorrelationVectors[2:pitchOrder, :]), axis=0)
	for q in range(1, pitchOrder+1):
		nPitches = np.diff(dftRange[q-1,:])+1
		nPitches = int(nPitches)
		validPitchIndices = np.array(range(0, nPitches))

		if q == 1:
			psi1, phi1, gammaOld1 = computeGammaSingleSinus(crossCorrelationVectors[:, validPitchIndices], a1[0, validPitchIndices], True)
			gammaNew1 = gammaOld1
			psi2, phi2, gammaOld2 = computeGammaSingleSinus(crossCorrelationVectors[:, validPitchIndices], a2[0, validPitchIndices], False)
			gammaNew2 = gammaOld2
		elif q == 2:
			R1, alpha1, gammaNew1 = computeGammaTwoSinus(
				crossCorrelationVectors[0:5, validPitchIndices],
				psi1[validPitchIndices],
				gammaOld1[validPitchIndices], True)
			R2, alpha2, gammaNew2 = computeGammaTwoSinus(
				crossCorrelationVectors[0:5, validPitchIndices],
				psi2[validPitchIndices],
				gammaOld2[validPitchIndices], False)
		else:
			if len(phi1.shape) == 1:
				phi1 = phi1[np.newaxis, :]
				phi2 = phi2[np.newaxis, :]
				psi1 = psi1[np.newaxis, :]
				psi2 = psi2[np.newaxis, :]
				gammaOld1 = gammaOld1[np.newaxis, :]
				gammaOld2 = gammaOld2[np.newaxis, :]
			
			R1, phi1, psi1, alpha1, gammaOld1, gammaNew1 = computeGammaMultipleSinus(
				R1[0:-1, validPitchIndices], q,
				crossCorrelationVectors[0:2*q+1, validPitchIndices],
				a1[q-2, validPitchIndices],
				phi1[:, validPitchIndices],
				psi1[:, validPitchIndices],
				gammaOld1[:, validPitchIndices],
				gammaNew1[:, validPitchIndices],
				alpha1[validPitchIndices], True)
			R2, phi2, psi2, alpha2, gammaOld2, gammaNew2 = computeGammaMultipleSinus(
				R2[0:-1, validPitchIndices], q,
				crossCorrelationVectors[0:2*q+1, validPitchIndices],
				a2[q-2, validPitchIndices],
				phi2[:, validPitchIndices],
				psi2[:, validPitchIndices],
				gammaOld2[:, validPitchIndices],
				gammaNew2[:, validPitchIndices],
				alpha2[validPitchIndices], False)
		if q == 1:
			gammaTpH.append(gammaNew1 / np.sqrt(gammaNew1))
			gammaTmH.append(gammaNew2 / np.sqrt(gammaNew2))
		else:
			gammaTpH.append(gammaNew1 / ((np.ones((q,1)) @ (np.sqrt(gammaNew1[q-1,:]).reshape(1,nPitches)))))
			gammaTmH.append(gammaNew2 / ((np.ones((q,1)) @ (np.sqrt(gammaNew2[q-1,:]).reshape(1,nPitches)))))
	return gammaTpH, gammaTmH	
		

def computeGammaSingleSinus(crossCorrelationVectors, a, hankelMatrixIsAdded):
	R = computeRowsOfToeplitzHankelMatrix(1, 1, crossCorrelationVectors, hankelMatrixIsAdded)
	psi = 1/R[0, :]
	gamma = psi
	phi = a*gamma
	return psi, phi, gamma

def computeGammaTwoSinus(crossCorrelationVectors, psi, gamma, hankelMatrixIsAdded):
	nPitches = len(psi)
	R = computeRowsOfToeplitzHankelMatrix(2, 2, crossCorrelationVectors, hankelMatrixIsAdded)
	alpha = R[0,:]*gamma
	gamma = np.vstack((-R[0,:]*psi, np.ones(nPitches)))/(np.ones(2).reshape(2,1) @ (R[1,:]-pow(R[0,:],2)*psi).reshape(1,nPitches))
	return R, alpha, gamma

def computeGammaMultipleSinus(ROld, iOrder, crossCorrelationVectors, a, phi, psi, gammaOld, gammaNew, alphaOld, hankelMatrixIsAdded):
	nPitches = len(a)
	RNew = computeRowsOfToeplitzHankelMatrix(iOrder, iOrder, crossCorrelationVectors, hankelMatrixIsAdded)
	lambda_ = a-np.sum(ROld*phi,axis=0)
	mu = -np.sum(ROld*psi,axis=0)
	if len(lambda_.shape) == 1:
		lambda_ = lambda_[np.newaxis, :]
		mu = mu[np.newaxis, :]
		alphaOld = alphaOld[np.newaxis, :]
		a = a[np.newaxis, :]
	phi = np.vstack((phi,np.zeros((1,nPitches))))+(np.ones((iOrder-1,1)) @ lambda_)*gammaNew
	psi = np.vstack((psi,np.zeros((1,nPitches))))+(np.ones((iOrder-1,1)) @ mu)*gammaNew
	alphaNew = np.sum(RNew[:-1,:]*gammaNew,axis=0)
	b = (np.ones((iOrder-1,1)) @ (alphaOld-alphaNew))*gammaNew+\
	np.vstack((np.zeros((1,nPitches)),gammaNew[:iOrder-2,:]))+\
	np.vstack((gammaNew[1:,:],np.zeros((1,nPitches))))-\
	np.vstack((gammaOld[:iOrder-2,:],np.zeros((1,nPitches))))+\
	(np.ones((iOrder-1,1)) @ (psi[-1,:])[np.newaxis, :])*phi-\
	(np.ones((iOrder-1,1)) @ (phi[-1,:])[np.newaxis, :])*psi
	nu = np.sum(RNew[:-1,:]*b,axis=0)/gammaNew[-1,:]
	gammaOld = gammaNew
	gammaNew = np.empty((iOrder,nPitches)) * np.nan
	gammaNew[iOrder-1,:] = 1/(nu+RNew[iOrder-1,:])
	gammaNew[:iOrder-1,:] = (np.ones((iOrder-1,1)) @ (gammaNew[iOrder-1,:]/gammaOld[-1,:])[np.newaxis, :])*b
	return RNew, phi, psi, alphaNew, gammaOld, gammaNew
			
	
def computeRowsOfToeplitzHankelMatrix(rowNumber, nColumns, crossCorrelationVectors, hankelMatrixIsAdded):
	if rowNumber == 1:
		toeplitzRows = crossCorrelationVectors[:nColumns,:]
	else:
		toeplitzRows = np.vstack((np.flip(crossCorrelationVectors[1:rowNumber,:], axis=0), crossCorrelationVectors[:nColumns-rowNumber+1,:]))
	hankelRows = crossCorrelationVectors[(np.arange(nColumns)) + rowNumber + 1, :]

	if hankelMatrixIsAdded:
		rowMatrix = toeplitzRows + hankelRows
	else:
		rowMatrix = toeplitzRows - hankelRows
	return rowMatrix
	
def computeCovVct(dataVector, maxLag, nData):
	covVct = np.array(range(maxLag+1))*np.nan
	dataVector = dataVector[np.newaxis, :]
	end = len(dataVector)
	for ii in range(maxLag+1):
		if ii == 0:
			covVct[ii] = (dataVector[:,ii:] @ dataVector.T)/nData
		else:
			covVct[ii] = (dataVector[:,ii:] @ dataVector[:,:-ii].T)/nData
	return covVct

def genSchurToeplitzAlg(covVct):
	arOrder = len(covVct)-1
	predErrVar = copy.deepcopy(covVct)
	cholVct = np.array(range(int(arOrder*(arOrder+1)/2)))*np.nan
	u = predErrVar[:]/math.sqrt(predErrVar[0])
	v = u[1:]
	cholIdx = np.array(range(arOrder))
	for p in range(arOrder):
		cholVct[cholIdx[0]:cholIdx[-1]+1] = u[:-1]
		nu = v[0]/u[0]
		det = (1-nu**2)
		predErrVar[p+1] = predErrVar[p]*det
		if p < arOrder-1:
			sqrtDet = math.sqrt(det)
			u = (u[:-1]-nu*v)/sqrtDet
			v = sqrtDet*v[1:]-nu*u[1:]
			cholIdx = cholIdx[:-1]+arOrder-p
	return predErrVar, cholVct

def cholDowndating(cholVct, v):
	copyv = copy.deepcopy(v)
	nRows = copyv.shape[0]
	copycholVct = copy.deepcopy(cholVct)
	cholIdx = np.array(range(nRows))
	for p in range(1,nRows+1):
		nu = np.ones((nRows-p+1,1))@(copyv[0,:]/copycholVct[cholIdx[0],:])[np.newaxis, :]
		sqrtDet = np.sqrt(1-nu**2)
		copycholVct[cholIdx,:] = (copycholVct[cholIdx,:] - nu*copyv)/sqrtDet
		if p < nRows:
			copyv = sqrtDet[1:,:]*copyv[1:,:]-nu[1:,:]*copycholVct[cholIdx[1:],:]
			cholIdx = cholIdx[:-1]+(nRows-p+1)
	return copycholVct	
			

def orderRecursiveForwardSubstitution(cholVct, qRho):
	maxArOrder, nFreqs = qRho.shape
	resVar = np.empty((maxArOrder,nFreqs))
	resVar[:] = np.nan
	cholIdx = np.empty((maxArOrder, maxArOrder)).astype(int)
	linParam = np.empty((maxArOrder,nFreqs))
	linParam[:] = np.nan
	for p in range(1, maxArOrder + 1):
		if p == 1:
			cholIdx[:,0] = np.arange(maxArOrder)
			linParam[0,:] = qRho[0,:] / cholVct[cholIdx[0,0],:]
			resVar[0,:] = linParam[0,:]**2
		else:
			cholIdx[p-1:,p-1] = cholIdx[-1,p-2] + np.arange(1, maxArOrder - p + 2)
			linParam[p-1,:] = (qRho[p-1,:] - np.sum(cholVct[cholIdx[p-1,:p-1],:] * linParam[:p-1,:], axis=0)) / cholVct[cholIdx[p-1,p-1],:]
			resVar[p-1,:] = resVar[p-2,:] + linParam[p-1,:]**2
	return resVar

def pitchArBicModelComparison(nData, objValNoPitch, objValPitch, validPitchOrders):
	maxArOrder = len(objValNoPitch)-1
	maxPitchOrder = objValPitch.shape[2]
	logBayesFactor = -float('inf') * np.ones((maxPitchOrder+1, maxArOrder+1))
	for ii in validPitchOrders:
		for jj in range(maxArOrder+1):
			if ii == 0:
				try:
					estExVar = min(objValNoPitch[jj])
				except:
					estExVar = objValNoPitch[jj]
			else:
				estExVar = min(objValPitch[jj,:,ii-1])
			if estExVar < 0:
				estExVar = float('inf')
			logBayesFactor[ii][jj] = -nData*np.log(estExVar)/2-(ii+jj/2)*np.log(nData)
	# normalise the Bayes' factors
	bayesFactor = np.exp(logBayesFactor - np.max(np.max(logBayesFactor)))
	bayesFactor = bayesFactor / np.sum(np.sum(bayesFactor))
	return bayesFactor

def computeObjectiveNaively(dataVector, pitchGrid, pitchOrder, arOrder, startIndex, nData):
	dataPower = (dataVector.T @ dataVector)/nData
	maxArOrder = len(dataVector) - nData
	timeIndices = np.arange(0, nData + maxArOrder).reshape(-1, 1) + startIndex
	if arOrder == 0:
#		print("arOrder is {0} while maxArOrder is {1}".format(arOrder, maxArOrder))
		arDataMatrix = np.zeros((nData+maxArOrder, 0))
	elif arOrder < maxArOrder:
#		print("arOrder is {0} while maxArOrder is {1}".format(arOrder, maxArOrder))
		arDataMatrix = toeplitz(np.concatenate((np.zeros(1), dataVector[:-1])), np.zeros(arOrder))
	else:
		arDataMatrix = toeplitz(np.concatenate((np.zeros(1), dataVector[:-1])), np.zeros(arOrder))

	if np.any(np.isnan(pitchGrid)) and arDataMatrix.shape[1] == 0:
	# no pitch and AR part
		objectiveGrid = dataPower
		estLinParams = np.nan
	elif np.any(np.isnan(pitchGrid)): # no pitch
		estLinParams = np.linalg.lstsq(arDataMatrix, dataVector, rcond=None)[0]
		objectiveGrid = dataPower - (dataVector.T @ arDataMatrix @ estLinParams) / nData
	else:
		try:
			nPitches = len(pitchGrid)
		except:
			nPitches = 1
		objectiveGrid = np.empty(nPitches)
		objectiveGrid[:] = np.nan
		estLinParams = np.empty((arOrder + 2 * pitchOrder, nPitches))
		estLinParams[:] = np.nan
		for ii in range(nPitches):
			try:
				pitchgrid = pitchGrid[ii]
			except:
				pitchgrid = pitchGrid
			cplxMtx = np.exp(1j * 2 * np.pi * timeIndices * pitchgrid * np.arange(1, pitchOrder + 1))
			
			sinusoidalMatrix = np.column_stack((np.real(cplxMtx[:, 0:1]), np.imag(cplxMtx[:, 0:1])))
			for i in range(1, cplxMtx.shape[1]):
				sinusoidalMatrix = np.hstack((sinusoidalMatrix, np.real(cplxMtx[:, i:i+1]), np.imag(cplxMtx[:, i:i+1])))
			# compute the linear parameters
			systemMtx = np.column_stack((arDataMatrix, sinusoidalMatrix))
			estLinParams[:, ii] = np.linalg.lstsq(systemMtx, dataVector, rcond=None)[0]
			# compute the objective
			objectiveGrid[ii] = dataPower - (dataVector.T @ systemMtx @ estLinParams[:, ii]) / nData
	return objectiveGrid, estLinParams

def fibonacciSearch(objective_function, lower_initial_bound, upper_initial_bound, required_interval):
	validate_objective_function(objective_function);
	validate_bound(lower_initial_bound);
	validate_bound(upper_initial_bound);
	initial_interval = upper_initial_bound - lower_initial_bound
	if initial_interval <= 0:
		raise ValueError('The lower bound must be smaller than the upper bound.')
	validate_interval(required_interval)
#	Compute the list of Fibonacci numbers
	fibonacci_no_list = compute_fibonacci_numbers(initial_interval, required_interval)
#	Perform the Fibonacci search
	lower_final_bound, upper_final_bound = narrow_bounds(objective_function, lower_initial_bound, upper_initial_bound, fibonacci_no_list)
	return lower_final_bound, upper_final_bound

def validate_objective_function(objective_function):
	if not callable(objective_function):
		raise ValueError("The input argument must be a function handle")

def validate_bound(bound):
	if not isinstance(bound, (int, float)) or not math.isfinite(bound):
		raise ValueError("The bounds must be real-valued scalars.")
		
def validate_interval(interval):
	if not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
		raise ValueError('The required interval must be a positive real-valued scalar.')

def fibonacci(index):
	# If any index elements are +/- infinite, complex-valued, not an integer, or larger than abs(70), issue an error
	index_element_is_non_integer = (type(index) != int) or abs(index - np.round(index)) > 1e-16
	index_element_is_too_large = abs(index) > 70
	index_is_invalid = index_element_is_non_integer or index_element_is_too_large
	if np.any(index_is_invalid):
		raise ValueError('An infinite, complex-valued, non-integer, or too large (index > 70) index has been given.')
	else:
		golden_ratio = 0.5 + np.sqrt(5) / 2
		fibonacci_no = np.round((golden_ratio**index-(1-golden_ratio)**index) / np.sqrt(5))
	return fibonacci_no

def fibonacciIndex(positiveNumber):
	fibonacci70 = 190392490709135
	if (type(positiveNumber) != np.float64) or (math.isinf(positiveNumber)) or (positiveNumber<2) or (positiveNumber>fibonacci70):
		raise ValueError('A infinite, complex-valued, too small (< 2) Fibonacci number has been given.')
	else:
		goldenRatio = 0.5 + math.sqrt(5)/2
		indexVector = math.floor(math.log(positiveNumber * math.sqrt(5) + 0.5) / math.log(goldenRatio))
		fibonacciNo = fibonacci(indexVector)
		if (abs(fibonacciNo - positiveNumber) >= 1e-16) and (not math.isnan(positiveNumber)):
			if fibonacciNo < positiveNumber:
				indexVector += 1
	return indexVector
	
#Compute the list of Fibonacci numbers so that the initial interval is narrowed down to an interval not greater than the required interval
def compute_fibonacci_numbers(initialInterval, requiredInterval):
	# The factor of two ensures that the Fibonacci search produces a final interval smaller than requiredInterval instead of 2*requiredInterval.
	intervalRatio = initialInterval / requiredInterval
	if intervalRatio < 2:
		raise ValueError('fibonacciSearch:argChk', 'The final interval must be smaller than half the initial interval.')

	nFibonacci = fibonacciIndex(intervalRatio)
	if nFibonacci > 70:
		raise ValueError('fibonacciSearch:argChk', 'The required final interval requires a Fibonacci number of index greater than 70. This cannot be computed reliably.')

	fibonacciNoList = [fibonacci(x) for x in range(2, nFibonacci + 1)]
	return fibonacciNoList
	
#Run the Fibonacci search algorithm to narrow down the lower and upper bound for the minimiser to within a tolerance of at most +/-requiredInterval/2. 	
def narrow_bounds(objectiveFunction, lowerBound, upperBound, fibonacciNoList):
    nFibonacci = len(fibonacciNoList)
    startInterval = upperBound - lowerBound
    iInterval = startInterval * fibonacciNoList[nFibonacci-2] / fibonacciNoList[nFibonacci-1]
    variableLowerVal = upperBound - iInterval
    funcLowerVal = objectiveFunction(variableLowerVal)[0][0]
    variableUpperVal = lowerBound + iInterval
    funcUpperVal = objectiveFunction(variableUpperVal)[0][0]
    nMaxIterations = nFibonacci - 2
    # Run the first nFibonacci-2 iterations
    for iIteration in range(1, nMaxIterations+1):
        if iIteration < nMaxIterations:
            iInterval = iInterval * fibonacciNoList[nFibonacci - iIteration - 2] / fibonacciNoList[nFibonacci - iIteration - 1]
        else:
            # To avoid that the interior points are the same after the last iteration, we multiply by slightly more than 0.5.
            iInterval = 1.01 * iInterval / 2
        if funcLowerVal > funcUpperVal:
            # The minimum is in the interval [variableLowerVal;upperBound]
            lowerBound = variableLowerVal
            variableLowerVal = variableUpperVal
            variableUpperVal = lowerBound + iInterval
            funcLowerVal = funcUpperVal
            funcUpperVal = objectiveFunction(variableUpperVal)[0][0]
        else:
            # The minimum is in the interval [lowerBound;variableUpperVal]
            upperBound = variableUpperVal
            variableUpperVal = variableLowerVal
            variableLowerVal = upperBound - iInterval
            funcUpperVal = funcLowerVal
            funcLowerVal = objectiveFunction(variableLowerVal)[0][0]
        # If the final search tolerance is within the precision of the computer, then stop
        if variableLowerVal > variableUpperVal:
            break
    # In the last iteration, only the final bounds are used so we do not waste computational resources on updating the other parameters
    if funcLowerVal > funcUpperVal:
        lowerBound = variableLowerVal
    else:
        upperBound = variableUpperVal
    return lowerBound, upperBound

def ldr(data_vector, ar_order, n_data):
    # buffers
    rho = np.full((ar_order, 1), np.nan)
    pred_err_var = np.full((ar_order + 1, 1), np.nan)
    nu = np.full((ar_order, 1), np.nan)
    # p == 0
    pred_err_var[0] = (data_vector.T @ data_vector) / n_data
    # p == 1
    if ar_order > 0:
        rho[0] = (data_vector[1:][np.newaxis, :] @ data_vector[:-1][:, np.newaxis]) / n_data
        nu[0] = rho[0] / pred_err_var[0]
        pred_err_var[1] = pred_err_var[0] * (1 - nu[0] ** 2)

        # we run the recursion in the reversed version beta to avoid one flipud
        beta_rev = nu[0][:, np.newaxis]
        for p in range(2, ar_order+1): # p >= 1
            rho[p-1] = (data_vector[p:][np.newaxis, :] @ data_vector[:-p][:, np.newaxis]) / n_data
            nu[p-1] = (rho[p-1] - beta_rev @ rho[:p-1]) / pred_err_var[p-1]
            pred_err_var[p] = pred_err_var[p-1] * (1 - nu[p-1] ** 2)
            if p < ar_order:
                beta_rev = np.hstack((np.zeros((1,1)), beta_rev)) + nu[p-1] * np.hstack((np.ones((1,1)), -beta_rev[::-1]))

    return pred_err_var, rho
