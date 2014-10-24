import pandas as pd
import numpy as np
import entropy
import sys

class calibrate:
    """
    Calibration class is a set of functions for estimating factors or parameters of SI models

    """

    def __init__(self, data, origins, destinations, trips, sep, dataForm='adj', diagFilter = True, cost='pow', factors=None, constraints={}, Oi=None, Dj=None, totalFlows=None):
        """
        initialize a model to be calibrated

        Parameters
        ----------
        data :          pandas data frame or csv with header composed of columns of necessary data
        origins:        str, value of column name for origins
        destinations:   str, value of column name for destinations
        trips:          str, value of column name for observed flows in data object
        sep:            str, deafults to 'pow', value of column name for distance values in data object
        dataForm:       not yet being used but will be to allow matrix data format
        diagFilter:     boolean, defaults to True, to filter intra-zonal flows
        cost:           str, either 'exp' for exponential or 'pow' for power distance function
        factors:        dict, defaults to None,  whose keys can be only 'origins' and/or 'destinations' and whose values are each a list of strings for names of columns in data object
        constraints:    dict, deaults to empty dict, whose keys can only 'production' and/or 'attraction' and whose values strings for the name of columns representing origins and/or destinations respectively
        Oi:             str, defaults to None, to provide total outflow column rather than calculate from data
        Dj:             str, defaults to None, to provide totoal inflow column rather than calculate from the data
        totalFlows:     str, defaults to None, to provide a field to tally total in/out flows if none is provided (only for GUI extension)
        """

        self.data = data
        self.origins = origins
        self.destinations = destinations
        self.cost = cost
        self.dataForm = dataForm
        self.constraints = constraints
        self.factors = factors
        self.trips = trips
        self.sep = sep
        self.Oi = Oi
        self.Dj = Dj
        self.totalFlows = totalFlows

        self.data[[origins, destinations]] = self.data[[origins, destinations]].astype(str)

        if diagFilter == True:
            if self.dataForm == 'adj':
                self.data = self.data[self.data[self.origins] != self.data[self.destinations]].reset_index(level = 0, drop = True)
            else:
                print 'Need to implement method to filter matrix for inter-zone data'

        self.prodCon = False
        self.attCon = False
        if 'production' in self.constraints.keys():
            self.prodCon = True
        if 'attraction' in self.constraints.keys():
            self.attCon = True


    def mle(self, initialParams):
        """
        calibrate max-entropy gravity model using maximum lilelihood parameter estimation

        Parameters
        ----------
        initialParams: dict, whose keys must be either 'beta' or equal to factors which have been specificed for parameter estimation and whose values are an integer
        """
        self.method = 'mle'
        self.initialParams = initialParams

        observed, data, knowns, params = entropy.setup(self.data, self.trips, self.sep, self.cost, self.factors,self.constraints, self.prodCon, self.attCon, self.initialParams, self.Oi, self.Dj, self.totalFlows)

        if self.factors != None:
            entropy.checkParams(self.factors, self.initialParams)

        if (self.prodCon == True) & (self.attCon == True):
            self.model = 'dConstrained'
        elif (self.prodCon == True) & (self.attCon == False):
            self.model = 'prodConstrained'
        elif (self.prodCon == False) & (self.attCon == True):
            self.model = 'attConstrained'
        elif (self.prodCon == False) & (self.attCon == False):
            self.model = 'unConstrained'

        self.results, cor, sumStr = entropy.run(observed, data, self.origins, self.destinations, knowns, params, self.trips, self.sep, self.cost, self.factors, self.constraints, self.model, self.initialParams)
        self.results.rsquared = cor**2
        self.results.sumStr = sumStr

        return self
