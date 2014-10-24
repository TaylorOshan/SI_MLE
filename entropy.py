from scipy.stats.stats import pearsonr
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import prettytable as pt
import sys

def sysDesc(data, trips, sep, origins, destinations):
    """
    calculate system/descriptive statistics of model results
    """

    numOrigins = len(data[origins].unique())
    numDestinations = len(data[destinations].unique())
    pairs = len(data)
    obsInt = np.sum(data[trips])
    predInt = np.sum(data['SIM_Estimates'])
    avgDist = round(np.sum(data[sep])*1.00000/pairs)
    avgDistTrav = round((np.sum(data[trips]*data[sep]))*1.00000/np.sum(data[trips])*1.00000)
    obsMeanTripLen = (np.sum(data[trips]*data[sep]))*1.00000/obsInt*1.000000
    predMeanTripLen = (np.sum(data['SIM_Estimates']*data[sep]))/predInt
    #Calculating the Asymmetry Index kills run time for large datasets due to all the loops
	#aSymSum = 0
    #for o in data[origins].unique():
        #for d in data[destinations].unique():
            #if o != d:
                #aSymSum += abs((data[trips][(data[origins] == o) & (data[destinations] == d)].values) - (data[trips][(data[origins] == d) & (data[destinations] == o)].values))
    #if numOrigins == numDestinations:
        #aSymInd = 50.0000*aSymSum[0]/(np.sum(data[trips]))
    aSymInd = 'N/A'
    #three likelihood statistics
    percentDev = round(((np.sum(abs(data[trips]-data['SIM_Estimates'])))/np.sum(data[trips]))*100, 3)
    intMean = round(np.sum(data[trips]/pairs), 1)
    percentDevMean = round(((np.sum(abs(data[trips]-intMean)))/np.sum(data[trips]))*100, 3)
    percentDevRed = (abs((percentDev-percentDevMean))/percentDevMean)*100
    pij = data[trips]/np.sum(data[trips])
    phatij = data['SIM_Estimates']/np.sum(data[trips])
    infoGain = np.sum(pij*np.log((pij/phatij)))
    sij = (pij+phatij)/2
    psiStat = np.sum(pij*np.log(pij/sij)) + np.sum(phatij*np.log(phatij/sij))
    MDI = 2*np.sum(data[trips])*psiStat
    #why is MDI only calculated once? skipped
    srmse = ((np.sum((data[trips]-data['SIM_Estimates'])**2)/pairs)**.5)/(np.sum(data[trips])/pairs)
    maxEntropy = round(np.log(pairs), 4)
    predEntropy = round(-np.sum(phatij*np.log(phatij)), 4)
    obsEntropy = round(-np.sum(pij*np.log(pij)), 4)
    diffPredEnt = round(maxEntropy - predEntropy, 4)
    diffObsEnt = round(maxEntropy - obsEntropy, 4)
    diffEntropy = round(predEntropy - obsEntropy, 4)
    entropyRS = round(diffPredEnt/diffObsEnt, 4)
    varPredEnt = round(((np.sum(phatij*(np.log(phatij)**2))-predEntropy**2)/obsInt) + ((pairs-1)/(2*obsInt**2)), 11)
    varObsEnt = round(((np.sum(pij*np.log(pij)**2)-obsEntropy**2)/obsInt) + ((pairs-1)/(2*obsInt**2)), 11)
    tStatEnt = round((predEntropy-obsEntropy)/((varPredEnt+varObsEnt)**.5), 4)
    #bhat = ((np.sum(data.Data))*(np.sum(data.SIM_Estimates))/(pairs-(np.sum(data.Data*data.SIM_Estimates))))/((np.sum(data.SIM_Estimates)**2)/(pairs-(np.sum(data.SIM_Estimates**2))))
    #print bhat
    #sebhat = (np.sum((data.Data-data.SIM_Estimates)**2/(pairs-2)))/(((np.sum(data.SIM_Estimates**2))-(np.sum(data.Data**2))/pairs)**.5)
    #top = ((np.sum((data.Data-data.SIM_Estimates)**2))/(pairs-2))
    #bottom = ((np.sum(data.SIM_Estimates)**2 - ((np.sum(data.Data)**2) / pairs))**.5)
    #print top, bottom, top/bottom
    #print sebhat
    #tbhat = (bhat - 1)/(top/bottom)
    #print tbhat
    return numOrigins, numDestinations, pairs, obsInt, predInt, avgDist, avgDistTrav, obsMeanTripLen, predMeanTripLen, aSymInd, percentDev, percentDevMean, percentDevRed, pij, phatij, infoGain, psiStat, MDI, srmse, maxEntropy, predEntropy, obsEntropy, diffPredEnt, diffObsEnt, diffEntropy, entropyRS, varPredEnt, varObsEnt, tStatEnt

def llStats(PV, data, params, factors, trips, sep, cost, model, constraints, knowns, estimates, initialParams):
    """
    calculate log-likelihood statistics for model
    """

	#calc the ll value of the fitted model with all params set to MLE's
    ll = np.sum((data.Data/np.sum(data.Data))*np.log((data.SIM_Estimates/np.sum(data.SIM_Estimates))))
    #for each parameter, set value to initial value and the rest to their MLE
    newlls = []
    lambs = []
    newPV = PV
    for x, param in enumerate(PV):
        newPV[x] = 0
        #calc new ll value for param
        buildLLFunctions(newPV, data, params, factors, trips, sep, cost, model, constraints, knowns)
        data = estimateFlows(data, sep, cost, model, factors)
        newll = np.sum((data.Data/np.sum(data.Data))*np.log((data.SIM_Estimates/np.sum(data.SIM_Estimates))))
        newlls.append(newll)
        #calc lambda (relative likelihood statistic) for the param
        lamb = 2*np.sum(data.Data)*(ll-newll)
        lambs.append(lamb)
        newPV = PV
    #set all params to zero
    for x, param in enumerate(PV):
        newPV[x] = 0
    #then calc the ll value with all params set to zero
    buildLLFunctions(newPV, data, params, factors, trips, sep, cost, model, constraints, knowns)
    data = estimateFlows(data, sep, cost, model, factors)
    llZero = np.sum((data.Data/np.sum(data.Data))*np.log((data.SIM_Estimates/np.sum(data.SIM_Estimates))))
    N = len(data)
    z = len(params)
    rho = 1 - (ll/llZero)
    adjRho = 1 - ((ll/(N-z))/(llZero/N))
    llMean = np.sum((data.Data/np.sum(data.Data))*np.log((np.sum(data.SIM_Estimates)/len(data.SIM_Estimates))/np.sum(data.SIM_Estimates)))
    return ll, newlls, lambs, llZero, rho, adjRho, llMean

def peStats(PV, data, params, factors, trips, sep, cost, model, constraints, knowns, estimates):
    """
    calculate parameter estimate statistics - standard errors
    """
    if len(PV) == 1:
        firstD = buildLLFunctions(PV, data, params, factors, trips, sep, cost, model, constraints, knowns)
        recalc = buildLLFunctions(PV+.001, data, params, factors, trips, sep, cost, model, constraints, knowns)
        diff = firstD[0]-recalc[0]
        secondD = -(1/(diff/.001))
        data[params[0]] = PV[0]
        return [np.sqrt(secondD)]
    elif len(PV) > 1:
        counter = 0
        varMatrix = np.zeros((len(PV),len(PV)))
        for x, param in enumerate(PV):
            firstD = buildLLFunctions(PV, data, params, factors, trips, sep, cost, model, constraints, knowns)
            varParams = list(PV)
            varParams[x] = varParams[x] + .001
            varMatrix[x] = buildLLFunctions(varParams, data, params, factors, trips, sep, cost, model, constraints, knowns)
            varMatrix[x] = (firstD-varMatrix[x])/(.001)

        return np.sqrt(-np.linalg.inv(varMatrix).diagonal())

def checkParams(factors, initParams):
    """
    check to make sure there are initial parameters for all factors
    """

    variables = []
    for key in factors.keys():
        if key not in ['origins', 'destinations']:
            sys.exit('Only acceptable keys for factors are "origns" and/or "destinations"')
        for factor in factors[key]:
            variables.append(factor)
    factors = set(variables)
    params = set(initParams.keys())
    params.discard('beta')
    if len(factors.symmetric_difference(params)) > 0:
        sys.exit('The initial paramter keys and the factor names must be symmetrical (excluding beta)')

def setup(data, trips, sep, cost, factors, constraints, prodCon, attCon, initialParams, Oi, Dj, totalFlows):
    """
    set up all initial variables and balancing factors for mle calibration
    """

    #The following setup is for within all models

    #There is always a beta parameter so set it to user's initial value and add to param list
    data['beta'] = initialParams['beta']
    params = ['beta']

    #This is the observed data for which we want to derive parameters
    if cost == 'exp':
        knowns = data[sep]
    elif cost == 'pow':
        knowns = np.log(data[sep])
    else:
        sys.exit(sys.exit("The distance/cost function must be either 'pow' or 'exp'."))

    #For doubly constrained model
    if (prodCon == True) & (attCon == True):

        #Variables for constants and deriving them
        data["Bj"] = 1.0
        data["Ai"] = 1.0
        data["OldAi"] = 10.000000000
        data["OldBj"] = 10.000000000
        data["diff"] = abs((data["OldAi"] - data["Ai"])/data["OldAi"])

        #Calc total outflows and inflows
        if Oi:
            data["Oi"] = data[Oi]
        else:
            Oi = data.groupby(data[constraints['production']]).aggregate({trips: np.sum})
            data["Oi"] = Oi.ix[pd.match(data[constraints['production']], Oi.index)].reset_index()[trips]

        if Dj:
            data["Dj"] = data[Dj]
        else:
            Dj = data.groupby(data[constraints['attraction']]).aggregate({trips: np.sum})
            data["Dj"] = Dj.ix[pd.match(data[constraints['attraction']], Dj.index)].reset_index()[trips]


    #For Production Constrained model
    if (prodCon == True) & (attCon == False):

        #Calc total outflows
        if factors == None:
            if not Dj:
                Dj = data.groupby(data[totalFlows]).aggregate({trips: np.sum})
                data["Dj"] = Dj.ix[pd.match(data[totalFlows], Dj.index)].reset_index()[trips].sort_index()

            else:
                data["Dj"] = data[Dj]

        if not Oi:
            Oi = data.groupby(data[constraints['production']]).aggregate({trips: np.sum})
            data["Oi"] = Oi.ix[pd.match(data[constraints['production']], Oi.index)].reset_index()[trips]
        else:
            data['Oi'] = data[Oi]


    #For Attraction Constrained model
    if (prodCon == False) & (attCon == True):

        #Calc total inflows
        if factors == None:
            if not Oi:
                Oi = data.groupby(data[totalFlows]).aggregate({trips: np.sum})
                data["Oi"] = Oi.ix[pd.match(data[totalFlows], Oi.index)].reset_index()[trips]
            else:
                data["Oi"] = data[Oi]
        if not Dj:
            Dj = data.groupby(data[constraints['attraction']]).aggregate({trips: np.sum})
            data["Dj"] = Dj.ix[pd.match(data[constraints['attraction']], Dj.index)].reset_index()[trips]
        else:
            data["Dj"] = data[Dj]


    #For Unconstrained Model
    if (prodCon == False) & (attCon == False):
        for factor in factors['origins']:
            #Include that information in the model
            knowns = knowns+np.log(data[factor])
            #Add to params list
            params.append(str(factor))
            #variable param vector
            data[str(factor) + 'Param'] = initialParams[factor]
        for factor in factors['destinations']:
            #Include that informatio in the model
            knowns = knowns+np.log(data[factor])
            #Add to params list
            params.append(str(factor))
            #variable param vector
            data[str(factor) + 'Param'] = initialParams[factor]

    #For all models besides unconstrained - is probably redundant and can be refactored

    #If there are additional factors we will include that observed data, add it to param list, and add a data vector for the param
    if factors != None:
        if attCon != False:
            for factor in factors['origins']:
                #Include that information in the model
                knowns = knowns+np.log(data[factor])
                #Add to params list
                params.append(str(factor))
                #variable param vector
                data[str(factor) + 'Param'] = initialParams[factor]
        if prodCon != False:
            for factor in factors['destinations']:
                #Include that informatio in the model
                knowns = knowns+np.log(data[factor])
                #Add to params list
                params.append(str(factor))
                #variable param vector
                data[str(factor) + 'Param'] = initialParams[factor]

    #Observed information is sum of trips multiplied by the log of known information
    observed = np.sum(data[trips]*knowns)

    #return observed info, data, knownn info, and params list
    return observed, data, knowns, params

def calcAi(data, sep, cost, factors, model):
    """
    calculate Ai balancing factor
    """

    #add distance data with appropriate functional form
    if cost == 'exp':
        Ai = np.exp(data[sep]*data["beta"])
    elif cost == 'pow':
        Ai = (data[sep]**data["beta"])
    else:
        sys.exit("The distance/cost function must be either 'pow' or 'exp'.")

    #Add factors
    if factors != None:
        for factor in factors['destinations']:
            Ai = Ai*(data[factor]**data[factor + 'Param'])

    else:
        Ai = Ai*data['Dj']

    #If model is doubly constrained add destination balancing factor
    if model == 'dConstrained':
        Ai = Ai*data["Bj"]


    data["Ai"] = Ai

def calcBj(data, sep, cost, factors, model):
    """
    calculate Bj balancing factor
    """

    #add distance data with appropriate functional form
    if cost == 'exp':
        Bj = np.exp(data[sep]*data["beta"])
    elif cost == 'pow':
        Bj = (data[sep]**data["beta"])
    else:
        sys.exit("The distance/cost function must be either 'pow' or 'exp'.")

    #Add factors
    if factors != None:
        for factor in factors['origins']:
            Bj = Bj*(data[factor]**data[factor + 'Param'])

    else:
        Bj = Bj*data['Oi']

    #If model is doubly constrained add origin balancing factor
    if model == 'dConstrained':
        Bj = Bj*data["Ai"]

    data["Bj"] = Bj

def balanceFactors(data, sep, cost, factors, constraints, model):
    """
    calculate balancing factors and balance the balancing factors if doubly constrained model
    """
    its = 0
    cnvg = 1
    while cnvg > .0001:
        its = its + 1
        #If model is prod or doubly constrained
        if model != 'attConstrained':
            calcAi(data, sep, cost, factors, model)
            AiBF = (data.groupby(data[constraints['production']].name).aggregate({"Ai": np.sum}))
            AiBF["Ai"] = 1/AiBF["Ai"]
            updates = AiBF.ix[pd.match(data[constraints['production']], AiBF.index), "Ai"]
            data["Ai"] = updates.reset_index(level=0, drop=True) if(updates.notnull().any()) else data["Ai"]
            #If model is prod constrained stop here - dont need to balance
            if model == 'prodConstrained':
                break
            if its == 1:
                data["OldAi"] = data["Ai"]
            else:
                data["diff"] = abs((data["OldAi"] - data["Ai"])/data["OldAi"])
                data["OldAi"] = data["Ai"]
        #If model is att or doubly constrained
        if model != 'prodConstrained':
            calcBj(data, sep, cost, factors, model)
            BjBF = data.groupby(data[constraints['attraction']].name).aggregate({"Bj": np.sum})
            BjBF["Bj"] = 1/BjBF["Bj"]
            updates = BjBF.ix[pd.match(data[constraints['attraction']], BjBF.index), "Bj"]
            data["Bj"] = updates.reset_index(level=0, drop=True) if(updates.notnull().any()) else data["Bj"]
            if its == 1:
                #If model is att constrained stop here - dont need to balance
                if model == 'attConstrained':
                    break
                data["OldBj"] = data["Bj"]
            else:
                data["diff"] = abs((data["OldBj"] - data["Bj"])/data["OldBj"])
                data["OldBj"] = data["Bj"]
        cnvg = np.sum(data["diff"])
        #print cnvg, its
    return data

def estimateFlows(data, sep, cost, model, factors):
    """
    estimate predicted flows multiplying individual model terms
    """

    #add distance data with appropriate functional form
    if cost == 'exp':
        decay = np.exp(data[sep]*data['beta'])
    elif cost == 'pow':
        decay = (data[sep]**data['beta'])
    else:
        sys.exit("The distance/cost function must be either 'pow' or 'exp'.")

    #For each type of model add in appropriate balancing factors and the factors

    if model == 'dConstrained':
        data["SIM_Estimates"] = data["Oi"]*data["Ai"]*data["Dj"]*data["Bj"]*decay

        if factors != None:
            for key in factors.keys():
                for factor in factors[key]:
                    data["SIM_Estimates"] = data["SIM_Estimates"]*(data[factor]**data[str(factor) + 'Param'])

    elif model == 'prodConstrained':
        data["SIM_Estimates"] = data["Oi"]*data["Ai"]*decay
        if factors != None:
            for factor in factors['destinations']:
                data["SIM_Estimates"] = data["SIM_Estimates"]*(data[factor]**data[str(factor) + 'Param'])
        else:
            data["SIM_Estimates"] = data["SIM_Estimates"]*data['Dj']

    elif model == 'attConstrained':
        data["SIM_Estimates"] = data["Dj"]*data["Bj"]*decay
        if factors != None:
            for factor in factors['origins']:
                data["SIM_Estimates"] = data["SIM_Estimates"]*(data[factor]**data[str(factor) + 'Param'])
        else:
            data["SIM_Estimates"] = data["SIM_Estimates"]*data['Oi']


    elif model == 'unConstrained':
        data["SIM_Estimates"] = decay
        if factors != None:
            for key in factors.keys():
                for factor in factors[key]:
                    data["SIM_Estimates"] = data["SIM_Estimates"]*(data[factor]**data[str(factor) + 'Param'])

    return data

def estimateCum(data, knowns):
    """
    calculate sum of all estimated flows and log of parameters being estimated (log likelihood)
    """

    return np.sum(data["SIM_Estimates"]*knowns)

#Function to construct log-likelihood functions for each parameter being estimated
def buildLLFunctions(PV, data, params, factors, trips, sep, cost, model, constraints, knowns, peM=False):
    """
    build log-likelihood functions for each parameter being estimated - used in optimization/calibration and statistics
    """

    #assign single param values to pandas dataframe vector
    for x, param in enumerate(params):
        if param != 'beta':
            data[str(param) + 'Param'] = PV[x]
        else:
            data[param] = PV[x]

    #if not calculating multiple standard errors on parameters and the model is not unconstrained then rebalance factors
    if peM == False and model != 'unConstrained':
        data = balanceFactors(data, sep, cost, factors, constraints, model)


    #build individual function compnents
    def buildFunction(common, data, trips, param, factors, beta=False):
        #build  factors for unconstreained model - probably redundant
        if model == 'unConstrained':
            first = True
            count = 1
            last = 0
            for key in factors.keys():
                last += len(factors[key])
            for key in factors.keys():
                for factor in factors[key]:
                    if first == True and count == last:
                        f = 'data["'+ str(factor) + '"]**PV[' + str(count) + ']'
                        first = False
                        count+=1
                    elif first == True and count != last:
                        f = 'data["'+ str(factor) + '"]**PV[' + str(count) + ']*'
                        first = False
                        count+=1
                    elif first == False and count != last:
                        f += 'data["'+ str(factor) + '"]**PV[' + str(count) + ']*'
                        count+=1
                    else:
                        f += 'data["'+ str(factor) + '"]**PV[' + str(count) + ']'
                        count+=1

        #for other models
        else:
            if factors != None:
                first = True
                count = 1
                #print factors.keys()
                for key in factors.keys():
                    for factor in factors[key]:
                        last = len(factors[key])
                        #print last
                        if first == True and count == last:
                            f = 'data["'+ str(factor) + '"]**PV[' + str(count) + ']'
                            first = False
                            count+=1
                        elif first == True and count != last:
                            f = 'data["'+ str(factor) + '"]**PV[' + str(count) + ']*'
                            first = False
                            count+=1
                        elif first == False and count != last:
                            f += 'data["'+ str(factor) + '"]**PV[' + str(count) + ']*'
                            count+=1
                        else:
                            f += 'data["'+ str(factor) + '"]**PV[' + str(count) + ']'
                            count+=1

        #If there are other factors use this routine to put together factors, distance, and log of known data values
        if factors != None:

            if cost == 'exp':
                decay = np.exp(data[sep]*PV[0])
            elif cost == 'pow':
                decay = (data[sep]**PV[0])
            else:
                sys.exit("The distance/cost function must be either 'pow' or 'exp'.")


            if beta == True:
                if cost == 'exp':
                    return np.sum(common*eval(f)*decay*data[param]) - np.sum(data[trips]*data[param])
                else:
                    return np.sum(common*eval(f)*decay*np.log(data[param])) - np.sum(data[trips]*np.log(data[param]))
            else:
                return np.sum(common*eval(f)*decay*np.log(data[param])) - np.sum(data[trips]*np.log(data[param]))

        #otherwise use this routine
        else:

            if cost == 'exp':
                decay = np.exp(data[sep]*PV[0])
            elif cost == 'pow':
                decay = (data[sep]**PV[0])
            else:
                sys.exit("The distance/cost function must be either 'pow' or 'exp'.")


            if beta == True:
                if cost == 'exp':
                    return np.sum(common*decay*data[param]) - np.sum(data[trips]*data[param])
                else:
                    return np.sum(common*decay*np.log(data[param])) - np.sum(data[trips]*np.log(data[param]))
            else:
                return np.sum(common*decay*np.log(data[param])) - np.sum(data[trips]*np.log(data[param]))


    #for each model type add the log-likelihood function for each parameter being estimated to a list
    #beta is done first and separately using the beta=True input
    functions = []

    if model == 'dConstrained':
        common = data['Ai']*data['Oi']*data['Bj']*data['Dj']
        func = buildFunction(common, data, trips, sep, factors, beta=True)
        functions.append(func)
        if factors != None:
            for key in factors.keys():
                for factor in factors[key]:
                    func = buildFunction(common, data, trips, factor, factors)
                    functions.append(func)


    if model == 'prodConstrained':
        common = data['Ai']*data['Oi']
        if factors == None:
            common = common*data['Dj']
        func = buildFunction(common, data, trips, sep, factors, beta=True)
        functions.append(func)
        if factors != None:
            for key in factors.keys():
                for factor in factors[key]:
                    func = buildFunction(common, data, trips, factor, factors)
                    functions.append(func)

    if model == 'attConstrained':
        common = data['Bj']*data['Dj']
        if factors == None:
            common = common*data['Oi']
        func = buildFunction(common, data, trips, sep, factors, beta=True)
        functions.append(func)
        if factors != None:
            for key in factors.keys():
                for factor in factors[key]:
                    func = buildFunction(common, data, trips, factor, factors)
                    functions.append(func)

    if model == 'unConstrained':
        common = 1
        func = buildFunction(common, data, trips, sep, factors, beta=True)
        functions.append(func)
        if factors != None:
            for key in factors.keys():
                for factor in factors[key]:
                    func = buildFunction(common, data, trips, factor, factors)
                    functions.append(func)


    return functions

def run(observed, data, origins, destinations, knowns, params, trips, sep, cost, factors, constraints, model, initialParams):
    """
    run the main routine which estimates parameters using mle
    """

    #print 'Model selected: ' + model
    #only run this function if model is not unconstrained - no balancing factors
    if model != 'unConstrained':
        data = balanceFactors(data, sep, cost, factors, constraints, model)
    #multiply model terms to get estimates
    data = estimateFlows(data, sep, cost, model, factors)
    #multiply estimates by log of known data for optimization
    estimates = estimateCum(data, knowns)
    its = 0

    #To avoid potential errors
    if abs(estimates-observed) != 0:

        #While optimization convergence is not met
        while abs(estimates - observed) > 1:
            #make list of single param values from pandas dataframe vector for each param
            paramSingle = []
            for param in params:
                if param != 'beta':
                    paramSingle.append(data[str(param) + 'Param'].ix[0])
                else:
                    paramSingle.append(data[param].ix[0])

            #run an iteration of scipy optimization solver
            updates = fsolve(buildLLFunctions, paramSingle, (data, params, factors, trips, sep, cost, model, constraints, knowns))
            #print updates, abs(estimates - observed)

            #format updates param values
            for x, each in enumerate(params):
                updates[x] = round(updates[x], 7)

            #re-balance and calculate estimates
            if model != 'unConstrained':
                data = balanceFactors(data, sep, cost, factors, constraints, model)
            data = estimateFlows(data, sep, cost, model, factors)
            estimates = estimateCum(data, knowns)

            its += 1

            #If more than 100 optimization runs than exit - no convergence
            if its > 100:
                break
        print "After " + str(its) + " runs, beta is : " + str(data["beta"].ix[0])
        print '1'
        #To ensure values of finished optimization are preserved after statistics are calculated - can change these values
        if 'Ai' in data.columns.names:
            Ai = data.Ai.values.copy()
        if 'Bj' in data.columns.names:
            Bj = data.Bj.values.copy()
        ests = data.SIM_Estimates.values.copy()

        new = ''
        cor = 0

        try:
            for x, each in enumerate(params):
                updates[x] = round(updates[x], 7)
            finalParams = updates.copy()
            #calculate statistics and output of model
            data['absoluteError'] = data.SIM_Estimates - data.Data
            data['percentError'] = (data.absoluteError/data.Data) * 100
            numOrigins, numDestinations, pairs, obsInt, predInt, avgDist, avgDistTrav, obsMeanTripLen, predMeanTripLen, aSymInd, percentDev, percentDevMean, percentDevRed, pij, phatij, infoGain, psiStat, MDI, srmse, maxEntropy, predEntropy, obsEntropy, diffPredEnt, diffObsEnt, diffEntropy, entropyRS, varPredEnt, varObsEnt, tStatEnt = sysDesc(data, trips, sep, origins, destinations)
            variance = peStats(updates, data, params, factors, trips, sep, cost, model, constraints, knowns, estimates)
            ll, newlls, lambs, llZero, rho, adjRho, llMean = llStats(updates, data, params, factors, trips, sep, cost, model, constraints, knowns, estimates, initialParams)

            if 'Ai' in data.columns.names:
                data.Ai = Ai
            if 'Bj' in data.columns.names:
                data.Bj = Bj
            data.SIM_Estimates = ests
            cor = pearsonr(data.SIM_Estimates, data.Data)[0]
            descStats = pt.PrettyTable(["Statistic", "Value"])
            descStats.align["Statistic"] = 'l'
            descStats.align["Value"] = 'l'
            descStats.padding_width = 1
            descStats.add_row(["Observed Mean Trip Length", str(obsMeanTripLen)])
            descStats.add_row(["Predicted Mean Trip Length", str(predMeanTripLen)])
            descStats.add_row(["# of Origin-Destination Pairs", str(pairs)])
            descStats.add_row(["Total Observed Interaction", str(obsInt)])
            descStats.add_row(["Total Predicted Interaction", str(predInt)])
            descStats.add_row(["Asymmetry Index", str(aSymInd)])
            #print descStats
            paramEsts = pt.PrettyTable(["Statistic", "Value"])
            paramEsts.align["Statistic"] = 'l'
            paramEsts.align["Value"] = 'l'
            paramEsts.padding_width = 1
            for x,param in enumerate(params):
                paramEsts.add_row([param+" Parameter Estimate", str(finalParams[x])])
            for x,param in enumerate(params):
                paramEsts.add_row(["Standard Error of " + param, str(variance[x])])
            paramEsts.add_row(["Log-Likelihood with All Params", str(ll)])
            for x,param in enumerate(params):
                paramEsts.add_row(["Log-Likelihood without "+ param, str(newlls[x])])
                paramEsts.add_row(["Lambda LL Statistic for "+ param, str(lambs[x])])
            #print paramEsts
            goodnessFit = pt.PrettyTable(["Statistic", "Value"])
            goodnessFit.align["Statistic"] = 'l'
            goodnessFit.align["Value"] = 'l'
            goodnessFit.padding_width = 1
            goodnessFit.add_row(["R-Squared", str(cor*cor)])
            goodnessFit.add_row(["T-Statistic of R-Squared", "Not computed"])
            goodnessFit.add_row(["% Deviation of Observed from Mean", str(percentDevMean)])
            goodnessFit.add_row(["% Deviation of Predicted from Observed", str(percentDev)])
            goodnessFit.add_row(["% Reductin in Deviation ", str(percentDevRed)])
            goodnessFit.add_row(["Ayeni S Information Statistic (PSI)", str(psiStat)])
            goodnessFit.add_row(["Minimum Discriminant Information Stat", str(MDI)])
            goodnessFit.add_row(["SRMSE Statistic", str(srmse)])
            goodnessFit.add_row(["Max Entropy for " + str(pairs) + " Cases", str(maxEntropy)])
            goodnessFit.add_row(["The Entropy of Predicted Data", str(predEntropy)])
            goodnessFit.add_row(["The Entropy of Observed Data", str(obsEntropy)])
            goodnessFit.add_row(["Max Entropy - Predicted Data Entropy", str(diffPredEnt)])
            goodnessFit.add_row(["Entropy of Predicted - Entropy of Observed", str(diffEntropy)])
            goodnessFit.add_row(["Entropy Ratio Statistic", str(entropyRS)])
            goodnessFit.add_row(["Variance of Entropy of Predicted Data", str(varPredEnt)])
            goodnessFit.add_row(["Variance of Entropy of Observed Data", str(varObsEnt)])
            goodnessFit.add_row(["T-Statistic for Absolute Entropy Difference", str(tStatEnt)])
            goodnessFit.add_row(["Information Gain Statistic", str(infoGain)])
            goodnessFit.add_row(["Rho-Squared Statistic", str(rho)])
            goodnessFit.add_row(["Adjusted Rho-Squared Statistic", str(adjRho)])
            goodnessFit.add_row(["Likelihood Value of Mean Model", str(llMean)])
            new += '\n'
            new += '\nModel type: ' + str(model)
            new += '\nWith ' + str(numOrigins) + ' origins and ' + str(numDestinations) + ' destinations.'
            new += '\n'
            new += '\nAfter ' + str(its) + ' iterations of the calibration routine,'
            new += '\nWith a cost/distance function of: ' +  str(cost)
            new += '\n'
            new += '\nThe number of origin-destination pairs considered = ' + str(pairs)
            new += '\n'
            new += "\nSystem Descriptive Statistics\n"
            new += descStats.get_string()
            new += '\n\n'
            new += "Parameter Estimates and Associated Statistics\n"
            new += paramEsts.get_string()
            new += '\n\n'
            new += "Goodness-of-fit Statistics\n"
            new += goodnessFit.get_string()

        except:
            new += 'Please try new initial parameters - optimization cannot converge to reasonable estimate'
        return data, cor, new
    else:
        new = 'Estimated values are equal to observed data flows. Ensure model type is not production/attraction constrained with only one origin/destination representing the total out/in flow'
        return data, 0, new




