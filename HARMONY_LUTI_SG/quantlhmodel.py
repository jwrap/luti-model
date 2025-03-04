"""
QUANTLHModel.py

Lakshmanan and Hansen form model. Used as a base for other models.
Uses an attractor, Aj, Population Ei and cost matrix Cij.
"""

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.optimize import minimize

from HARMONY_LUTI_SG.globals import *



class QUANTLHModel:
    """
        constructor
        @param n number of residential zones (MSOA)
        @param m number of school point zones
    """

    ## TODO: include bikes
    def __init__(self, m,n):
        # constructor
        self.m = m
        self.n = n
        self.Ei = np.zeros(m)
        self.Aj = np.zeros(n)
        self.cij_pt = np.zeros(1)
        self.cij_road = np.zeros(1)
        self.cij_bike = np.zeros(1) # prepare for bikes
        self.OD_pt = np.zeros(1)
        self.OD_road = np.zeros(1)
        self.OD_bike = np.zeros(1) # prepare for bikes
        self.OBS_cbar_pt = 0
        self.OBS_cbar_road = 0
        self.OBS_cbar_bike = 0

    # end def constructor

    ###########################################

    """
        setPopulationEi
        Given a data frame containing one column with the zone number (i) and one column
        with the actual data values, fill the Ei population property with data so that
        the position in the Ei numpy array is the zonei field of the data. 
        The code is almost identical to the setAttractorsAj method.
        NOTE: max(i) < self.m
    """
    ## TODO: replace zonei with correct column

    def setPopulationEi(self, df, zoneiColName,dataColName):
        df2 = df.sort_values(by = [zoneiColName])
        self.Ei = df2[dataColName].to_numpy()
        assert len(self.Ei) == self.m, "FATAL: setPopulationEi length Ei =" + str(len(self.Ei)) + "MUST equal model definition size of m=" + str(self.m) + "ERROR from quantlhmodel.py setPopulationEi()"

    ##############################################

    """
        setAttractorsAj
        Given a data frame containing one column with the zone number (j) and one column
        with the actual data values, fill the Aj attractors property with data so that
        the position in the Aj numpy array is the zonej field of the data.
        The code is almost identical to the setPopulationEi method.
        NOTE: max(j) < self.n
    """

    ## TODO: replace zonei with correct column name
    ## TODO: alternatively, change name in data to "zonei"

    def setAttractorsAj(self, df, zonejColName, dataColName):
        df2 = df.sort_values(by = [zonejColName])
        self.Aj = df2[dataColName].to_numpy()
        assert len(self.Aj) == self.n, "FATAL: setAttractorsAj length Aj=" + str(len(self.Aj)) + " MUST equal model definition size of n=" + str(self.n) + "ERROR from quantlhmodel.py setAttractorsAj()"

    ###############################################

    """
    setCostsMatrix
    Assign the cost matrix for the model to use when it runs.
    NOTE: this MUST match the m x n order of the model and be a numpy array
    """

    def setCostMatrixCij(self, cij_pt, cij_road, cij_bike):
        assert cij_pt.shape == cij_road.shape,"FATAL: setCostsMatrix cij_pt is not the same shape as cij_road"
        i, j = cij_pt.shape  # Hopefully cij_road.shape = cij_pt.shape
        assert i == self.m, "FATAL: setCostsMatrix cij matrix is the wrong size, cij.m=" + str(
            i) + " MUST match model definition of m=" + str(self.m)
        assert j == self.n, "FATAL: setCostsMatrix cij matrix is the wrong size, cij.n=" + str(
            j) + " MUST match model definition of n=" + str(self.n)
        self.cij_pt = cij_pt
        self.cij_road = cij_road
        self.cij_bike = cij_bike


    """
        setObsMatrix
        Assign the cost matrix for the model to use when it runs.
        NOTE: this MUST match the m x n order of the model and be a numpy array
    """

    ## TODO: include bikes

    def setODMatrix(self, filepath_OD_pt, filepath_OD_road): # OD_bike
        OD_pt = pd.read_csv(filepath_OD_pt, header=None)
        OD_road = pd.read_csv(filepath_OD_road, header=None)
        i, j = OD_pt.shape ## OD_road and OD_bike should have the same form
        assert i == self.m, "FATAL: setCostsMatrix cij matrix is the wrong size, cij.m=" + str(
            i) + " MUST match model definition of m=" + str(self.m) + "ERROR from quantlhmodel.py setODMatrix()"
        assert j == self.n, "FATAL: setCostsMatrix cij matrix is the wrong size, cij.n=" + str(
            j) + " MUST match model definition of n=" + str(self.n) + "ERROR from quantlhmodel.py setODMatrix()"
        self.OD_pt = OD_pt
        self.OD_road = OD_road
        # self.OD_bike = OD_bike

    ################################################


    """
    computeCBar
    Compute average trip length TODO: VERY COMPUTATIONALLY INTENSIVE - FIX IT
    @param Sij trips matrix containing the flow numbers between zone (i) and schools (j)
    @param cij trip times between i and j
    """
    def computeCBar(self, Sij, cij):
        CNumerator = np.sum(Sij * cij)
        CDenominator = np.sum(Sij)
        cbar = CNumerator / CDenominator
        return cbar
    ######################################################

    """
            setObsCbar
            Assign the observed cbar values for model calibration.
    """

    ## TODO: include bikes
    # how do I calibrate for bikes
    # answer: take observed values from MZMV

    def setObsCbar(self, OBS_cbar_pt, OBS_cbar_road, OBS_cbar_bike):
        self.OBS_cbar_pt = OBS_cbar_pt
        self.OBS_cbar_road = OBS_cbar_road
        self.OBS_cbar_bike = OBS_cbar_bike

    ################################################

    """
        computeCBar
        Compute average trip length TODO: VERY COMPUTATIONALLY INTENSIVE - FIX IT
        @param Sij trips matrix containing the flow numbers between MSOA (i) and schools (j)
        @param cij trip times between i and j
    """

    def computeCbar(self, Sij, cij):
        CNumenator = np.sum(Sij * cij)
        CDenominator = np.sum(Sij)
        cbar = CNumenator/CDenominator

        return cbar

    #####################################################

    def calculateDj(self, Tij):
        (M, N) = np.shape(Tij)
        Dj = np.zeros(N)
        Dj = Tij.sum(axis=0)
        return Dj

    """
    run Model run2modes
    Lakshamanan Hansen Model Two Modes
    """

    def runTwoModes(self, logger, Beta):
        # run model
        # i = employment zone
        # j = residential zone
        # Ei = number of jobs per zone
        # Aj = attractor of HH (floor space)
        # cij_mode = travel cost per mode
        # Modes: PT and individual
        # Beta: beta values for 2 modes - from literature
        # Observed cbar values: OBS_cbar_pt, OBS_cbar_road
        # Travel cost per mode: cij_pt, cij_road
        # SObs_k: observed flows per mode
        # e: Scaling factor per zone i and mode k

        # Initialisation of parameters
        n_modes = 2  ## Number of modes
        cij_k = [self.cij_pt, self.cij_road]  # list of cost matrices
        SObs_k = [self.OD_pt, self.OD_road]

        CBarObs = [self.OBS_cbar_pt, self.OBS_cbar_road]

        # Set up Beta for modes 0, 1 to 1.0 as starting point
        Beta = [1] * n_modes
        e_i = [[] for k in range(n_modes)]
        for k in range(n_modes):
            e_i[k] = [1] * self.m

        # set up correction factor
        c = 1 / 1000

        logger.warning("Calibrating the model...")
        iteration = 0
        converged = False
        max_iterations = 250
        while not converged:
            iteration += 1
            # print("Iteration:", iteration)

            # Initialise variables:
            Sij = [[] for i in range(n_modes)]  # initialise Sij with a number of empty lists equal to n_modes

            # hold copy of pre multiplied copies of -Beta_k * cij[k] for each mode
            ExpMBetaCijk = [[] for k in range(n_modes)]
            for kk in range(n_modes):
                ExpMBetaCijk[kk] = np.exp(-Beta[kk] * cij_k[kk])

            ## this is the main model loop to calculate Sij[k] trip numbers for each mode
            for k in range(n_modes):  # mode loop
                Sij[k] = np.zeros(self.m * self.n).reshape(self.m, self.n)
                for i in range(self.m):
                    # denom = np.dot(self.Aj , ExpMBetaCijk[k][i]) ## dot product, not *
                    ## take dot because only one mode is considered "competition" (Lak. Hansen, 1965)
                    denom = 0
                    for kk in range(n_modes):
                        # test = self.Aj * ExpMBetaCijk[kk][i] ## check which format denom has
                        denom += np.sum(self.Aj * ExpMBetaCijk[kk][i])
                    # calculate using observed flows
                    # introduce scaling factor e_i (Lak, Hansen 1965)
                    Sij2 = e_i[k][i] * (self.Ei * (
                            self.Aj * ExpMBetaCijk[k][i] / denom) + c)  ## use c to get e_i in case P or A is 0
                    Sij[k][i] = Sij2  # put answer slice back in return array

            # end for k

            # now we see how well it worked and modify the two betas accordingly
            # we have no Oi or Dj to check against, so it can only be CBar and totals
            """
            # Calibration with CBar values
            # Calculate mean predicted trips and mean observed trips (this is CBar)
            if calibration == 1:
                converged = True
                CBarPred = np.zeros(n_modes)
                delta = np.ones(n_modes)
                for k in range(n_modes):
                    CBarPred[k] = self.computeCbar(Sij[k], cij_k[k])
                    delta[k] = np.absolute(CBarPred[k] - CBarObs[k]) ## the aim is to minimize delta[n]
                    # beta mode here and convergence check
                    if delta[k] / CBarObs[k] > 0.05: # adjust tolerance
                        Beta[k] = Beta[k] * CBarPred[k] / CBarObs[k]
                        converged = False
                # end for k
                # commuter sum blocks
            """
            # DjObs : vector with dimension = number of destinations
            DjObs = [[] for i in range(n_modes)]
            for k in range(n_modes):
                DjObs[k] = np.zeros(self.n)
            for k in range(n_modes):
                DjObs[k] += SObs_k[k].sum(axis=1)
            #
            DjPred = [[] for i in range(n_modes)]
            DjPredSum = np.zeros(n_modes)
            DjObsSum = np.zeros(n_modes)
            delta = np.zeros(n_modes)

            # Calibration with Observed flows
            for k in range(n_modes):
                DjPred[k] = self.calculateDj(Sij[k])  # sum trips over row
                DjPredSum[k] = np.sum(DjPred[k])
                DjObsSum[k] = np.sum(DjObs[k])
                # gradient descent search on beta
                delta[k] = DjPredSum[k] - DjObsSum[k]
                # delta check on beta stopping condition for convergence
            converged = True

            for k in range(n_modes):
                tolerance = abs(delta[k] / DjObsSum[k])
                if tolerance > 0.1:
                    e_i[k][e_i[k] == 0] = 1.0  # catch div by 0 error
                    e_i[k] = SObs_k[k] / Sij[k]
                    #         for i in range(self.m):
                    #             if delta[k] > 0: # decrease gradient search on production rate
                    #                 e_i[k][i] = e_i[k][i] * DjPred[k][i] / DjObs[k][i]
                    #             if delta[k] < 0: # increase gradient search on production rate
                    #                 e_i[k][i] = e_i[k][i] * DjObs[k][i] / DjPred[k][i]
                    converged = False

            CBarPred = np.zeros(n_modes)
            # Calculate CBar
            for k in range(n_modes):
                CBarPred[k] = self.computeCBar(Sij[k], cij_k[k])

            TotalSij_pt = Sij[0].sum()
            TotalSij_road = Sij[1].sum()
            # TotalSij_bike = Sij[2].sum # prepare for bikes

            # Debug block
            # for k in range(0, n_modes):
            #     print("Beta",k, "=", Beta[k])
            #     print("delta", k, "=", delta[k])
            # end for k
            # print("delta", delta[0], delta[1]) # should be a k loop
            # print("beta", Beta[0], Beta[1])
            logger.warning(
                "iterations = {0:d} beta pu = {1:.6f} beta pr={2:.6f} cbar pred pu={3:.1f} ({4:.1f}) cbar pred pr={5:.1f} ({6:.1f})"
                .format(iteration, Beta[0], Beta[1], CBarPred[0], CBarObs[0], CBarPred[1], CBarObs[1]))

            if iteration == max_iterations:
                logger.warning(
                    "Maximum iterations reached. Returning beta pu = {0:.6f} and beta pr = {1:.6f} with predicted cbar pu = {2:.1f} ({3:.1f}) and cbar pr = {4:.1f} ({5:.1f})"
                    .format(Beta[0], Beta[1], CBarPred[0], CBarObs[0], CBarPred[1], CBarObs[1]))
                break

        return Sij, e_i, CBarPred, CBarObs  # Note that Sij = [Sij_k=0 , Sij_k=1] and CBarPred = [CBarPred_0, CBarPred_1]

    def runThreeModes(self, logger):
        # run model
        # i = employment zone
        # j = residential zone
        # Ei = number of jobs per zone
        # Aj = attractor of HH (floor space)
        # cij_mode = travel cost for "mode"
        # Modes: public and private transportation
        # Beta = Beta values for 2 modes - this is also output
        # Observed cbar values: OBS_cbar_pt, OBS_cbar_road
        # Travel cost per mode: cij_pt, cij_road

        # Initialisation of parameters
        n_modes = 3  # Number of modes
        cij_k = [self.cij_pt, self.cij_road, self.cij_bike]  # list of cost matrices

        CBarObs = [self.OBS_cbar_pt, self.OBS_cbar_road,self.OBS_cbar_bike]

        # Set up Beta for modes 0, 1 to 1.0 as a starting point
        Beta = np.ones(n_modes)

        logger.warning("Calibrating the model...")
        iteration = 0
        converged = False
        while not converged:
            iteration += 1
            # print("Iteration: ", iteration)

            # Initialise variables:
            Sij = [[] for i in range(n_modes)]  # initialise Sij with a number of empty lists equal to n_modes

            # hold copy of pre multiplied copies of -Beta_k * cij[k] for each mode
            ExpMBetaCijk = [[] for k in range(n_modes)]
            for kk in range(n_modes):
                ExpMBetaCijk[kk] = np.exp(-Beta[kk] * cij_k[kk])

                # this is the main model loop to calculate Sij[k] trip numbers for each mode k
            for k in range(n_modes):  # mode loop
                Sij[k] = np.zeros(self.m * self.n).reshape(self.m, self.n)
                for i in range(self.m):
                    denom = 0
                    for kk in range(n_modes):
                        denom += np.sum(self.Aj * ExpMBetaCijk[kk][i, :])
                    Sij2 = self.Ei[i] * (self.Aj * ExpMBetaCijk[k][i, :] / denom)
                    Sij[k][i, :] = Sij2  # put answer slice back in return array
            # end for k

            # now we see how well it worked and modify the two betas accordingly
            # we have no Oi or Dj to check against, so it can only be CBar and totals

            # Calculate mean predicted trips and mean observed trips (this is CBar)
            converged = True
            CBarPred = np.zeros(n_modes)
            delta = np.ones(n_modes)
            for k in range(n_modes):
                CBarPred[k] = self.computeCBar(Sij[k], cij_k[k])
                delta[k] = np.absolute(CBarPred[k] - CBarObs[k])  # the aim is to minimise delta[0]+delta[1]+...
                # beta mode here and convergence check
                if delta[k] / CBarObs[k] > 0.05:
                    Beta[k] = Beta[k] * CBarPred[k] / CBarObs[k]
                    converged = False
            # end for k

            # commuter sum blocks
            TotalSij_pt = Sij[0].sum()
            TotalSij_road = Sij[1].sum()
            TotalSij_bike = Sij[2].sum()
            TotalEi = self.Ei.sum()  # total jobs = pu+pr above

            # Debug block
            # for k in range(0,n_modes):
            #    print("Beta", k, "=", Beta[k])
            #    print("delta", k, "=", delta[k])
            # end for k
            # print("delta", delta[0], delta[1]) #should be a k loop
            logger.warning(
                "iteration= {0:d} beta pu={1:.3f} beta pr={2:.3f} beta_bike ={3:.3f} cbar pred pu={4:.1f} ({5:.1f})  cbar pred pr={6:.1f} ({7:.1f}) cbar pred b = ({8:.1f}) ({9:.1f})"
                .format(iteration, Beta[0], Beta[1], Beta[2], CBarPred[0], CBarObs[0], CBarPred[1], CBarObs[1], CBarPred[2], CBarObs[2]))
            # print("TotalSij_pt={0:.1f} TotalSij_road={1:.1f} Total={2:.1f} ({3:.1f})"
            #       .format(TotalSij_pt, TotalSij_road, TotalSij_pt + TotalSij_road, TotalEi))

        # end while ! converged

        return Sij, Beta, CBarPred, CBarObs


        ##################################################

    """
        run Model run2modes_NoCalibration
        Quant model for two modes of transport without calibration
        @returns Sij predicted flows between i and j
    """

    ## TODO: adust to hold bikes

    def run3modes_NoCalibration(self, Beta):
        n_modes = len(Beta)
        cij_k = [self.cij_pt, self.cij_road, self.cij_bike]
        # print("Running model for ", n_modes, "modes.")
        # Initialise variables
        Sij = [[] for i in range(n_modes)] # initialise Sij with a number of empty lists equal to n_modes

        # hold copy of pre multiplied copies of -Beta_k * cij[k] for each mode
        ExpMBetaCijk = [[] for k in range(n_modes)]
        for kk in range(n_modes):
            ExpMBetaCijk[kk] = np.exp(-Beta[kk] * cij_k[kk])

        # this is the main model loop to calculate Sij[k] trip numbers for each mode k

        for k in range(n_modes): # mode loop
            Sij[k] = np.zeros(self.m * self.n).reshape(self.m, self.n)
            for i in range(self.m):
                denom = 0
                for kk in range(n_modes):
                    denom += np.sum(self.Aj * ExpMBetaCijk[kk][i, :])
                Sij2 = self.Ei[i] * (self.Aj * ExpMBetaCijk[k][i, :] / denom)
                Sij[k][i, :] = Sij2 # put answer slice back in return array


        CBarPred = np.zeros(n_modes)
        for k in range(n_modes):
            CBarPred[k] = self.computeCbar(Sij[k], cij_k[k])

        return Sij, CBarPred

    ###############################################################

    """
    computeProbabilities2modes
    Compute the probability of a flow from an to any (i.e. all) of the possible point zones
    """

    def computeProbabilities3modes(self, Sij):
        n_modes = 3
        probSij = [[] for i in range(n_modes)]

        for k in range(n_modes):
            probSij[k] = np.arange(self.m * self.n, dtype=np.float).reshape(self.m, self.n)
            for i in range(self.m):
                sum = np.sum(Sij[k][i])
                if sum <= 0:
                    sum = 1  # catch for divide by zero - just let the zero probs come through to the final matrix
                probSij[k][i,] = Sij[k][i] / sum

        return probSij

    #######################################################################

    ## TODO: include bikes
    @staticmethod
    def loadODData(inputs):
        # load observed trips for each mode:

        # import the csv OD public as Pandas Dataframe
        OD_pt_df = pd.read_csv(inputs["ODPublic"], header = None)
        # Convert the dataframe to numpy matrix
        OD_pt = OD_pt_df.to_numpy()

        # Import the csv OD private as Pandas DataFrame
        OD_road_df = pd.read_csv(inputs["ODPrivate"], header=None)
        # Convert to numpy matrix
        OD_road = OD_road_df.to_numpy()

        return OD_pt, OD_road