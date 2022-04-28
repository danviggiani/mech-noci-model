# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:02:48 2019

Updated on Apr 28 2022

@authors: Dan Viggiani, Jack P. Callaghan; University of Waterloo

"""

#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import time
import neuro_dict



#%%
""" 1. CONSTANTS SECTION """


SIM_T = 10     # Simulation time in seconds
RES = 0.0002        # Resolution in seconds
Q = int(SIM_T / RES)    # Number of samples in the simulation
P = 500           # Number of iterations


# Takes in a percentile (between 0 and 1) and returns a PPT in MPa
# Constants use a Gumbel Distribution
ppt_mu = 0.842689
ppt_beta = 0.282371
PPT = lambda x: -ppt_beta * np.log(-np.log(x)) + ppt_mu

# Test Neuron Mechanical Sensitivity
PPTv = PPT(np.random.uniform(low = 0.001, high = 0.999, size = P)) # in MPa


# Sensitivity Analysis Variables
PSI_NT = np.array((1.042, 1.0)) # Neuron sensitivity terms [Ad, C]
TAU = (0.008, 0.200, 1.7446)    # Time constants in seconds [rapid, slow, ultra-slow]
GAMMA_Ad = (0.7048, 0.2286, 0.0666) # Kernel shape terms for the Ad neuron [rapid, slow, ultra-slow]
GAMMA_C = (0.60, 0.25, 0.15)        # Kernel shape terms for the C neuron [rapid, slow, ultra-slow]
S_Ad = (17.7430, -0.5334, 29.201)   # Scaling term for the Ad neuron [s0, s1, s2]
S_C = (27.530, -2.014, -4.831)      # Scaling term for the Ad neuron [s0, s1, s2]
K0p = 0.08  # Excitatory PSP scaling term
K0m = -0.01 # Inhibitory PSP scaling term


# Index of neurons simulated imported from the neuro_dict.py file
nidx = neuro_dict.NIDX


# Adjacency Matrix for neural connectivity and connection strength
# Each connection is explicitly labelled with an excitatory (K0p) or inhibitory (K0m) constant
Arch = np.zeros([len(nidx),len(nidx)])
Arch[0,4] = K0p
Arch[0,5] = K0p
Arch[1,3] = K0p
Arch[1,5] = K0m
Arch[2,3] = K0p
Arch[2,5] = K0m
Arch[2,6] = K0m 
Arch[3,7] = K0p 
Arch[4,3] = K0p 
Arch[5,3] = K0m 
Arch[5,4] = K0m 
Arch[6,4] = K0m 


# Neuron Constants from Gerstner et al (2014), modified for a resting membrane potential of 0 mV
Gk = 35.0      # Potassium channel conductance mS/cm^2 (smaller number, more APs)
GNa = 40.0     # Sodium channel conductance mS/cm^2 (bigger number, more APs)
Gl = 0.3       # Leaky channel conductance mS/cm^2
Vk = -12.0     # Potassium eq potential mV
VNa = 120.0    # Sodium eq potential mV
Vl = -1.056    # Leaky ion eq potential mV
Cm = 1.0       # Membrane capacitance in uF/cm^2

# Initial values
M0 = 0.04975 
H0 = 0.62246
N0 = 0.10509

# "A" functions are for the open state of each channel, "B" functions are for the closed state
AM = lambda x: (0.182 * (x - 30.0)) / (1 - np.exp(-(x - 30.0) / 9.0))
BM = lambda x: (-0.124 * (x - 30.0)) / (1 - np.exp((x - 30.0) / 9.0))
AH = lambda x: 0.25 * np.exp(-(x + 25.0) / 12.0)
BH = lambda x: (0.25 * np.exp((x - 3.0) / 6.0)) / np.exp((x + 25.0) / 12.0)
AN = lambda x: (0.02 * (x - 40.0)) / (1 - np.exp(-(x - 40.0) / 9.0))
BN = lambda x: (-0.002 * (x - 40.0)) / (1 - np.exp((x - 40.0) / 9.0))                   



#%%
""" 2. MECHANICAL SECTION """ 

def InterFxn(x, x0, l0):
    """
    A moveable, scalable hyperbolic tangent function to smoothly transition between two curves
    A way to smoothly move between 0 and 1
    
    Inputs:
        "x" is the main 'input' variable to the function, usually this will be a time vector
        "x0" is the value of "x" where it'll output 0.5; the midpoint of the function
        "l0" is a scaling constant, a bigger number means a sharper transition, should always be positive
        
    Outputs:
        A hyperbolic tangent function smoothly transitioning from 0 to 1
    """
    
    return 0.5 * (1 + np.tanh(l0 * (x - x0)))
    

def LinPPT(ppt, dP = 0.2, l0 = 5.0):
    """
    Makes a linear pressure pain threshold function
    Linearly increases from zero to a pressure pain threshold, truncates based on SIM_T and Q
    
    Inputs:
        "ppt" is the desired pressure pain threshold in MPa, represents the peak pressure
        "dP" is the rate of pressure increase and decrease, defaults to 0.2 MPa/s
        "k" is a constant dtermining how sharp the turnaround is at the peak pressure value, bigger numbers indicate a sharper point, defaults to 5
        
    Outputs:
        Time-varying pressures starting at 0 MPa and increasing to "ppt" at "dP" MPa/s, 
        the output is length "Q" + 1, descends back to zero if time permits
    
    """

    t = np.linspace(0, SIM_T, Q + 1)    # Time from 0 to SIM_T in seconds
    
    # Linear and Smooth-interpolation function
    linf = lambda x, a, b: a * x + b
    
    # Components:
    #   l1 is the linear increasing function from 0 to "ppt"
    #   l2 is the linear decreasing funciton from "ppt" beack down to 0
    #   i12 is the interpolation function between l1 and l2
    #   i20 is the interpolation function between l2 and zero
    l1 = linf(t, dP, 0)
    l2 = linf(t, -dP, 2 * ppt)
    i12 = InterFxn(t, ppt / dP, l0)
    i20 = InterFxn(t, 2 * ppt / dP, l0)
    
    return l1 + i12 * (l2 - l1) + i20 * (-l2)


#%%
""" 3. SENSITIVTY SECTION """


def FrontPad(x):
    """
    A helper function for "Mech2Current" that puts length(x) - 1 zeros in front of an input array "x"
    
    Inputs:
        "x" is vector of some length n
        
    Outputs:
        The array "x" with n - 1 zeros in front of it
    """
    return np.pad(x, (len(x) - 1, 0), 'constant', constant_values = (0, 0))


def Mech2Current(stress, t, sens, is_myelinated = True, scale = 1.0):
    """
    Taken from Gerling et al (2018), coverts a stress over time to a current over time
    Developed to include rapid-adapting (r), slow-adapting (s), and ultra-slow-adapting (u) touch data
    Checks if the input is myelinated or not, and uses different weightings so the unmyelinated neurons have longer-lasting effects
    
    Requires "FrontPad"
    
    Inputs:
        "stress" is a vector containing stress data in MPa
        "t" is a vector of each time point in milliseconds "stress"; must be the same length as "stress"
        "sens" is the Neuron object's sensitivity in MPa
        "is_myelinated" is a boolean checking whether the neuron is myelinated or not to pick curve constants, defaults to True
        "scale" is a float scaling factor to gain the outputted current; is multiplicative; defaults to 1
        
    Outputs:
        The function I(t) from Gerling et al 2018, Equation 4, scaled up by "scale"
    
    """
    
    if is_myelinated:
        gr, gs, gu = GAMMA_Ad
        g_sum = sum(GAMMA_Ad)
        s0, s1, s2 = S_Ad

        
    else:
        gr, gs, gu = GAMMA_C
        g_sum = sum(GAMMA_C)
        s0, s1, s2 = S_C

    
    gr, gs, gu = gr / g_sum, gs / g_sum, gu / g_sum # Re-normalizing constants
    
    tr, ts, tu = TAU # Time constants in seconds
    ks_peak, ks_steady = 0.87, 0.13 # Things related to the slow-adapting shape
    
    
    expf = lambda x, tau: np.exp(- x / tau)
    
    # Kernel function from Gerling et al., (2018)
    kernelt = (gr * expf(t, tr)
               + gs * (ks_peak * expf(t, ts) + ks_steady)
               + gu * expf(t, tu))
    
    # Time-derivative of stress
    dstress = np.gradient(stress, edge_order = 2)
    
    # Scaling parameter s that's a function of the peak stress in the exposure derived at a mean dstress of 0.2 MPa/s
    mx = np.max(stress)
    s = s0 * np.log(mx - s1) + s2

    
    # Pads the stress-derivative, convolves it with respect to the kernel function, and scales up the output
    return (scale * s / sens) * np.convolve(kernelt, FrontPad(dstress), mode = 'valid')
    

def AB2Current(stress):
    """
    A way to have the AB peripheral neuron not generate unreasonably large input currents
    
    Inputs:
        "stress" is a vector containing stress data in MPa
        
    Outputs:
        An input current suitable for the AB neuron
    """
    k = 2.45 # Scales the function up and down
    d = 59.2 # Squishes/Stretches along the x-axis
    b = 1.25 # Value at stress = 0, set to be right below the current limit of 1.26 nA
    
    stress[stress < 0 ] = 0
    
    return k * np.log(d * stress + 1) + b


#%%    
""" 4. NEURONAL SECTION """


def FireNeuron(pre_syn, post_syn, current_time, current_q, output_weight):
    """
    A function that causes action potentials in Neuron objects
    Calling this function is what sends an AP from a given Neuron to one of its Post-Synaptic Neurons
    Both Neurons should be part of the same network and draw from a common adjancency matrix
    
    Inputs:
        "pre_syn" is the pre-synaptic Neuron object; this one fires the action potential
        "post_syn" is the post-synaptic Neuron object; this one recieves the action potential
        "current_time" is the time in seconds when the pre-synaptic Neuron object fires, used to compute travel time
        "current_q" is the frame number matching "current_time", used for indexing
        "output_weight" is a float matching the strength of the connection between the two neurons; positive numbers are for excitatory connections, negative numbers are for inhibitory ones
        
    Outputs:
        Doesn't actually output anything, just directly changes the post-synaptic neuron's ion channels
        This uses the "PSP" method in the presynaptic neuron
    
    """
    
    
    # Updates the time of both neurons
    pre_syn.UpdateTime(current_time)
    post_syn.UpdateTime(current_time)
    
    # Adjusts the time vector using the following algorithm:
    # t0 is the current_time, T is the end of the simulation (simulation_time), apt is the presynaptic neuron's action potential time, t1 = t0 + apt
    # Generates a time vector going from t1 to T
    apt = pre_syn.aptime
    tte = pre_syn.simulation_time - (current_time + apt)
    qv = pre_syn.MakeQVector(current_q + int(apt / RES), pre_syn.n_samples)
    
    pspt = np.linspace(0, tte, len(qv))

    
    # Applies the result of the action potential to the postsynaptic neuron
    if output_weight < 0:
        post_syn.inhibit[qv] += pre_syn.PSP(output_weight, pspt)
    else:
        post_syn.excite[qv] += pre_syn.PSP(output_weight, pspt)
    
    # Adjusts the presynaptic neuron's state
    pre_syn.tready = pre_syn.refractory_period       


# A function that takes a given neuron and then appropriately activates all the other neurons it attaches to.
# "pre_syn" is a single neuron object, "list_post_syn" should be a list of neuron objects
# "output_weights" is a list or array the same length as "list_post_syn" with the appropriate weight for each of those Neurons            
def ActionPotentials(pre_syn, list_post_syn, current_time, current_q, output_weights):
    """
    A wrapper function for "FireNeuron"
    This takes a single pre-synaptic Neuron object ad activates all of its post-synaptic Neurons
    "list_post_syn" and "output_weights" need to have the same number of elements
    All Neurons and output weights should be drawn from the same adjacency matrix
    
    Inputs:
        "pre_syn" is the pre-synpatic Neuron object; this is the one firing the action potential
        "list_post_syn" is a list of Neuron objects or a single Neuron object; these are all the Neurons that "pre_syn" connects to that would recieve the action potential
        "current_time" is the time in seconds when the pre-synaptic Neuron object fires, used to compute travel time
        "current_q" is the frame number matching "current_time", used for indexing
        "output_weights" is a list of floats or a single float that indicates the connection type (excitatory/inhibitory) and strength for each Neuron in "list_post_syn"
        
    Outputs:
        This funciton has no outputs, it directly alters ion channels in "list_post_syn" through the "FireNeuron" function using the "PSP" method
    """

    if isinstance(list_post_syn, list):
        nPst = len(list_post_syn)
    
        for ps in range(nPst):
            FireNeuron(pre_syn, list_post_syn[ps], current_time, current_q, output_weights[ps])
    
    else:
        FireNeuron(pre_syn, list_post_syn, current_time, current_q, output_weights)


def PreAP(total_neuron_list, A, pre_no):
    """
    A function that gets the necessary information to call "ActionPotentials" and "FireNeuron" and puts it in the right format
    
    Inputs:
        "total_neuron_list" is the list of all the neurons in the current network
        "A" is the adjacency matrix documenting the connections anf strength of each connection
        "pre_no" is the index number that corresponds to the presynaptic neuron; the neuron that will fire an action potential
    
    Outputs:
        "pre_syn" is the presynaptic neuron object
        "post_neurons" is a list of the postsynaptic neurons that "pre_syn" connects to taken from "total_neuron_list"
        "post_weights" is a list of floats extracted from "A" denoting the output weight of each neuron in "post_neuron"
    
    I believe the syntax makes the outputs of "post_neurons" and "post_weights" not a list is there's only a single postsynaptic neuron
    Regardless, both "post_neurons" and "post_weights" will be of the same length
    """
    
    # Identifies the neuron in question
    pre_syn = total_neuron_list[pre_no]
    
    # Assigns the post-synaptic neuron list
    ipn = pre_syn.postsynaptic
    post_neurons = [total_neuron_list[i] for i in ipn]
    
    # Extracts the appropriate output_weights from the adjancency matrix A
    pwa = A[pre_no,:]
    post_weights = [pwa[i] for i in ipn]
    
    return pre_syn, post_neurons, post_weights



def ConnectNeurons(total_neuron_list, pre_syn, post_syn, weight):
    """
    A function that connects two neurons together
    This was previously a "Builder" class with a single method called "Attach" that got called twice for each connection
    Reworked to just be a function called once per connection
    
    Inputs:
        "total_neuron_list" the list of all neuron objects in the network
        "pre_syn" is the index number of the presynaptic neuron
        "post_syn" is the index number of the postsynaptic neuron
        "weight" is the weight of the connection between "pre_syn" and "post_syn"
        
    Outputs:
        This function has no outputs, it modifies the "presynaptic" and "postsynaptic" atttributes of the "post_syn" and "pre_syn" Neurons in the network respectively
    """
    
    # Identifies the pre- and postsynpatic  neurons
    n_pre = total_neuron_list[pre_syn]
    n_post = total_neuron_list[post_syn]
    
    # Puts the postsynaptic index in the presynaptic neuron
    n_pre.postsynaptic = n_pre.postsynaptic + [post_syn]
    
    # Puts the presynaptic index in the postsynaptic neuron then attributes the connection weight to the postsynaptic neuron
    n_post.presynaptic = n_post.presynaptic + [pre_syn]
    n_post.SetWeights(weight)


class Neuron:
    """
    The main class describing all the neurons in the neural model
    These act as Hodgkin-Huxley neurons with some simplifications to allow for action potential travel and post-synaptic voltage summation
    Instance Variables should be specified upon creation, use global variables to affect Class Variables
    Derived Attributes use the above variables to initialize and are predominantly acted upon during simulations
    
    Uses times ("t") and frames ("q"), where q/t is the sample rate or temporal resolution of the object
    
    Instance Variables:
        "name" a string describing the neuron, defaults to 'None'
        "diameter" the axon diameter in micrometres, affects "cvel", defaults to 1 um
        "axon_length" the axon length in millimetres, affects "aptime" and "apq", defaults to 10 mm
        "refractory_period" the minimum time between successive action potentials in millseconds, defaults to 5 ms
        "activation_threshold" the minimum voltage needed before an action potential is sent in millivolts, defaults to 35 mV
        "resting_Vm" is the resting membrane voltage of the neuron in millivolts, defaults to 0 mV
        "I_tonic" is the input current for tonic firing, defaults to 0 nA
        "sensitivity" is the minimum stress in MPa that causes the neuron to fire, defaults to 0.5 MPa
        "myelinated" is a boolean indicating if the neuron is myelinated or not, affects the "cvel", defaults to True
        "nociceptive" is a boolean indicating if the neuron is nociceptive or not, defaults to True, doesn't do anything yet
        "primary_afferent" is a boolean indicating if the neuron originates in the periphery (start of the network) or not, defaults to True, used to determine which neurons are affected by tissue loads
        "is_brainstem" is a boolean indicating if the neuron is the brainstem (end of the network) or not, defaults to False, doesn't do anything yet
        "presynaptic" is a list of indices within the network indicating the presynaptic neurons that could activate this neuron, defaults to an empty list
        "postsynaptic" is a list of indices within the network indicating the postsynaptic neurons that could be activated by this neuron, defaults to an empty list
        "psweights" is a list of floats indicating the strength of each connect, has the same length of "presynaptic"; the post-synpaptic neuron currently tracks the connectivity strength
    
    Class Variables: 
        "simulation_time" is the total time in milliseconds this Neuron will run for, defaults to the global setting "SIM_T"
        "n_samples" is the number of samples within "simulation_time", defaults to the global setting "Q" + 1
        "ic0" is an array of size [3,] containing the initial values of the potassium (n), sodium activation (m), and sodium inactivation (h) channel constants, defaults to the global settings "N0", "M0", and "H0"
    
    Derived Attributes:
        "membrane_potential" is an [n_samples,] array for tracking the membrane voltage at each sample, initializes to "resing_Vm"
        "input_current" is an [n_samples,] array keeping track of the instantaneous input current into the cell, initializes to "I_tonic"
        "inhibit" is an [n_samples,] array for recieving inhibitory PSPs from presynatptic neurons, initializes as zeros
        "excite" is an [n_samples,] array for recieving excitatory PSPs from presynatptic neurons, initializes as zeros
        "ion_channels" is an [n_samples, 3] array holding the instantaneous state of each of the "n", "m", amd "h" ion channels, initializes to "ic0"
        "psp_out" is an [n_samples,] array indicating the frames when the neuron fired an action potential as a 1, all other frames are stored as a 0
        "tready" is a counter variable indicating the number of milliseconds before the neuron can fire an action potential
        "active_range" is a variable-length array of times that updates each time the neuron takes part in some event, initializes as a range from 0 to "SIM_T" with "n_samples" elements, the starting point changes as time passes
        "cvel" is the neuron's conduction velocity in m/s, derived from "diameter" and "myelinated"
        "aptime" is the time in seconds (to four decimal places) needed for an action potential to travel the neuron's axon, is assumed constant for all connections that neuron makes, derived from "cvel" and "axon_length"
        "apq" is the number of frames corresponding to "aptime"
        
    Methods:
        "SetWeights" adds the in puts weights to the existing weights in a list
        "UpdateTime" changes the "active_range" of the neuron based on the inputted times
        "MakeTimeVector" makes a vector with the units of milliseconds spanning the inputted start and end time points at a frame resolution
        "MakeQVector" makes a vector of frames/indices spanning the inputted start and end frames
        "PSP" generates excitatory or inhibitory post-synaptic potentials to affect ion channel states; communication functions
        "HH" solves the Hodgkin-Huxley set of ODEs with a Newton-Euler approximation at a frame by frame step
        "Spike" sends out an action potential from the neuron at the inputted time 
        "GetAPFreq" does a shitty firing rate computation by counting peaks and dividing by "simulation_time" converted into seconds 
        "CheckReady" determines if the variable "tready" is 0 (is ready) or non-zero (is not ready)
        "ResetNeuron" called on each frame, reduces "tready" by the appropriate amount each frame (1 / "RES")
        "PlotVm" generates a plot of the neuron's membrane voltage over time
        "APTrain" determines whether the neuron contained a sufficient bunch of action potentials to qualify as a 'train'; if so gives some details on that train; if multiple options, only gives details on the most populous train
    """
        
    # Class Variables
    simulation_time = SIM_T
    n_samples = Q + 1
    ic0 = np.array([N0, M0, H0])
    
    def __init__(self, name = 'None', diameter = 1, axon_length = 10, refractory_period = 0.005, activation_threshold = 35, resting_Vm = 0, I_tonic = 0, sensitivity = 0.5, myelinated = True, nociceptive = True, primary_afferent = True, is_brainstem = False, presynaptic = [], postsynaptic = [], psweights = [], simulation_time = simulation_time, n_samples = n_samples, ic0 = ic0):
        
        # Neuron Properties       
        self.name = name   # What is this Neuron?
        self.diameter = diameter  # In micrometres
        self.axon_length = axon_length  # In millimetres
        self.refractory_period = refractory_period  # In seconds
        self.activation_threshold = activation_threshold  # In millivolts
        self.resting_Vm = resting_Vm # In millivolts; tracked separately to account for the instantaneous membrane potential changing
        self.I_tonic = I_tonic # In nA
        self.sensitivity = sensitivity # In megapascals; mechanical sensitivity
        
        # Current state of the neuron
        self.membrane_potential = np.ones(n_samples) * resting_Vm # In millivolts
        self.input_current = np.ones(n_samples) * I_tonic
        self.inhibit = np.zeros(n_samples) # The inhibitory inputs to the neuron; go into the HH method
        self.excite = np.zeros(n_samples) # The excitatory inputs to the neuron; go into the HH method
        self.ion_channels = ic0 * np.ones((n_samples, 3))
        self.psp_out = np.zeros(n_samples) # Indicates when this neuron has fired
        self.tready = 0 # In seconds, used to track the time since last fire using the refractory period
        self.active_range = np.arange(0, simulation_time, n_samples) # Start and end points in seconds
        
        # Architecture and connectivity        
        self.myelinated = myelinated
        self.nociceptive = nociceptive
        self.primary_afferent = primary_afferent
        self.is_brainstem = is_brainstem
        self.presynaptic = presynaptic
        self.postsynaptic = postsynaptic
        self.psweights = psweights
        
        # Conduction velocity in m/s as a function of axon diameter
        # Myelinated is taken from Hursh (1939)
        # Unmyelinated is taken from Kandel et al., (2013)
        if myelinated == True:
            self.cvel = 6 * diameter
        else:
            self.cvel = (16 * diameter + 20) / 13
        
        # Action potential transmission
        hapt = max(RES, self.axon_length / (self.cvel * 1000)) # I would be dividing a distance in mm by a velocity in m/s, so the correct of 1000 in the denominator ensures it's in seconds
        self.apq = int(hapt / RES)
        self.aptime = np.round(self.apq * RES, 4) # This backwards recomputation ensures that the aptime property is a multiple of RES
    
    
    def SetWeights(self, weight):
        """
        Sets the output weights of the presynaptic neuron onto the postsynaptic neuron; called by "ConnectNeurons" before the simulation starts
        Negative values are inhibitory, positive values are excitatory
        All are stored on the postsynaptic neuron and refer to the effects of its presynaptic neurons
        
        Inputs:
            "weight" is a float indicating the strength of the connection
            
        Outputs:
            None, just updates the list "psweights"
        
        """
        self.psweights = self.psweights + [weight]
    

    def UpdateTime(self, current_time):
        """
        Updates the lower bound of the active_range parameter based on "current_time"
        
        Inputs:
            "current_time" is a float between 0 and "simulation_time" indicating the time point in the simulation when something is happening
            
        Outputs:
            None, just updates "active_range"
        """
        self.active_range = np.arange(current_time, self.simulation_time + RES, RES)
        
    
    def MakeTimeVector(self, ts = 0, te = simulation_time):
        """
        Creates a range starting at the current time and ending at the end of the simulation time for feeding into the PSP method
        If the user specifies a starting time (ts) and an ending time (te), it uses those instead
        
        Inputs:
            "ts" is a number indicating the start time of the vector, defaults to 0
            "te" is a number indicating the end of the vector, defaults to "simulation_time"
            Must follow 0 <= ts < te <= simulation_time
            
        Outputs:
            An array running from "ts" to "te" with resolution "RES"
        """
        return np.arange(ts, te, RES)
    
    
    def MakeQVector(self, qs = 0, qe = n_samples):
        """
        Creates a range starting at the current sample (q) and ending at the last sample (Q) for feeding into the PSP method
        If the user specifies a starting sample (qs) and an ending sample (qe), it uses those instead
        
        Inputs:
            "qs" is a natural number indicating the start time of the vector, defaults to 0
            "qe" is a natural number indicating the end of the vector, defaults to "n_samples"
            Must follow 0 <= qs < qe <= n_samples
            
        Outputs:
            An array of whole numbers running from "qs" to "qe"
        """
        return np.arange(qs, qe, 1)
    
    
    def PSP(self, output_weight, t, is_neuropeptide = False):
        """
        Generates post-synaptic potentials using a fractional exponent function over a time vector "t"
        Described as "Communication Functions" in the companion paper
        General shape: k0 * exp(k1 * t + k2) / (1 + exp(k3 * t + k4))
        The constants vary based on whether the "output_weight" is positive or negative
        
        EPSP shapes: Lu and Perl (2005), Harrison et al (1988), Yoshimura and Jessell (1990) 
        IPSP shapes: Lu and Perl (2003), McCormick (1989), Yoshimura and Nishi (1995)
        
        Inputs:
            "output_weight" is a float indicating the strength of the connection between the pre and postsynaptic neurons
            "t" is a time vector outputted from "MakeTimeVector", this formulation assumes the time vector always starts at t=0 and gets positioned where it needs to by indexing
            "is_neuropeptide" is a boolean determining whether or not to use the neuropeptide shape
            
        Outputs:
            The PSP function of the time vector "t" scaled by "output_weight"
        """
        
        pspfxn = lambda x: k0 * (np.exp(k1 * t + k2))/(1 + np.exp(k3 * t + k4))
                
        if output_weight > 0:
            k0, k1, k2, k3, k4 = (0.9158, -43, 0.65, -820, 6.8)
            
        else:
            k0, k1, k2, k3, k4 = (0.8306, -7.5, 0.47, -300, 6.7)
        
        return output_weight * pspfxn(t)    
    

    def HH(self, q, dt = 0.05):
        """
        The heavy-lifting membrane voltage function that solves the Hodgkin-Huxley ODEs (Hodgkin and Huxley, 1952)
        Adapted to include an excitatory and inhibitory input in the ion channel state variables "m" and "n" respectively
        Calls upon the following neuron attributes: "membrane_potential", "input_current", "ion_channels", "excite", and "inhibit"
        
        Inputs:
            "q" is the current frame in the simulation
            "dt" is the time-step for the Newton-Euler approximation in ms
            
        Outputs:
            None, modifies the "membrane_potential" and "ion_channels" attributes directly
        """
        
        Vm = self.membrane_potential[q]
        n, m, h = self.ion_channels[q]       
        nStep = int((RES * 1000) / dt)
        
        I0 = self.input_current[q]
        ex = self.excite[q]
        ib = self.inhibit[q]
        
        # Does a multi-step Newton-Euler approximation
        # Iterates through multiple times each step for numerical stability
        for i in range(nStep):
             
            Ik = Gk * (n**4) * (Vm - Vk)
            INa = GNa * (m**3) * h * (Vm - VNa)
            Il = Gl * (Vm - Vl)
            
            dVm = (I0 - (Ik + INa + Il)) / Cm
            dn = (AN(Vm) - ib) * (1 - n) - BN(Vm) * n   # Includes inhibitory communication function       
            dm = (AM(Vm) + ex) * (1 - m) - BM(Vm) * m   # Includes excitatory communication function
            dh = AH(Vm) * (1 - h) - BH(Vm) * h
            
            Vm += dt * dVm
            n += dt * dn
            m += dt * dm
            h += dt * dh
            
        self.membrane_potential[q + 1] = Vm
        self.ion_channels[q + 1,:] = np.array([n, m, h])        
        
        
    def Spike(self, current_time, current_q, ow):
        """
        Basically a Method copying the "FireNeuron" function
        Makes an action potential at the current time and frame and applies it to itself
        
        Inputs:
            "current_time" is the current timepoint in seconds, should correspond to the "current_q"
            "current_q" is the current frame, should correspond to the "current_time"
            "ow" is a float to be applied to the "PSP" method; is the "output_weight" from the "FireNeuron" function
            
        Outputs:
            None, modifies the "inhibit" and "excite" attributes of the neuron
        """
        self.UpdateTime(current_time)
        
        tte = self.simulation_time - current_time
        qv = self.MakeQVector(current_q, self.n_samples)
        pspt = np.linspace(0, tte, len(qv))
    
        # Applies the result of the action potential to the postsynaptic neuron
        if ow < 0:
            self.inhibit[qv] += self.PSP(ow, pspt)
        else:
            self.excite[qv] += self.PSP(ow, pspt)
        
        # Adjusts the presynaptic neuron's state
        self.tready = self.refractory_period   
    

    def GetAPFreq(self):
        """
        Returns the mean firing frequency of the neuron through janky peak-counting and mean inter-peak intervals
        Uses the "find_peaks" function from scipy.signal library
        Should be called on a neuron with some sort of underlying current, otherwise its output doesn't really make much sense
        
        Inputs:
            None, just uses the neuron object's attributes: "membrane_potential" and "activation_threshold"
            
        Outputs:
            Returns the frequency of action potentials the 
        """
        vm = self.membrane_potential
        ht = self.activation_threshold # Threshold for saying if an AP happened, defaults to 50% of the max membrane potential
        
        # Computing the average inter-peak interval
        vm_pk, _ = find_peaks(vm, height = ht)
        avg_ipi = np.mean(np.diff(vm_pk))
        
        # Output in Hz:
        if np.isnan(avg_ipi):
            apf = 0.0
        else:
            apf = 1 / (RES * avg_ipi)
            
        return apf
    
    
    def CheckReady(self):
        """
        A simple checking method for a neuron's refractory period
        Should get called regularly
        
        Inputs:
            None, uses the "tready" attribute
            
        Outputs:
            Boolean; True if "tready" is 0, otherwise returns False
        """
        if self.tready > 0:
            return False     
        else:
            return True
        
        
    def ResetNeuron(self, tincr=RES):
        """
        A way to update the "tready" attribute every frame, reduces "tready" by a set amount to indicate that time has passed
        Is expected to be called every frame
        
        Inputs:
            "tincr" is the amount to decrease "tready" by instance; defaults to "RES" and shouldn't be changed
            
        Outputs:
            None, modifies "tready" directly
        """
        if self.tready > 0:
            self.tready = self.tready - tincr
            
            
    def PlotVm(self, xtra = True):
        """
        A method to automatically plot the neuron's "membrane_potential" attribute
        Uses "pyplot" from the matplotlib library
        
        Inputs:
            "xtra" is a boolean indicating whether to include axes labels and a legend; defaults to True
            
        Outputs:
            None, brings up the plot
        """
        
        st = self.simulation_time
        x = np.linspace(0, st, Q + 1)
        
        plt.plot(x, self.membrane_potential, '-k', label = self.name)
        
        if xtra:
            plt.xlabel('Time (s)')
            plt.ylabel('Membrane Potential (mV)')
            plt.show()
        
        
    def APTrain(self):
        """
        A way to auto-check if an action potential train occurred somewhere in that neuron's membrane potential
        An action potential train was defined as a window of at least 500 ms consisting of action potentials occurring at 21 Hz or greater
        Only identifies the AP train with the most action potentials in it for the moment
        
        Inputs:
            None, uses the "membrane_potential" attribute
            
        Outputs:
            "was_train" is a boolean indicating if a train existed or didn't
            "aptr_char" is a list holding:
                - The start time in s of the first AP in the train
                - The length of the action potential train in s
                - The number of action potenials in that train
                - The peak firing rate of that train
        """
        was_train = False
        vm = self.membrane_potential
        ht = self.activation_threshold
        
        min_dur = 0.5   # Duration criterion in s
        min_rate = 21   # Frequency criterion in Hz
        
        vmpk, _ = find_peaks(vm, height = ht)   # Index of all action potential peaks
        ipi = np.diff(vmpk, prepend = 0) * RES  # Interspike intervals in s
        n_tot = len(vmpk)                       # Total number of APs
        pkix = np.arange(n_tot)                 # An array to keep track of the APs
        
        # Checks for any action potentials
        if n_tot > 0:
            
            # Finds if any were more than 0.2 s apart from each other to ID the AP train with the most spikes
            chk_200 = ipi > 0.2
            
            if np.any(chk_200):
                pts = np.hstack((0, pkix[chk_200], n_tot))  # Assembles an array of the possible starting indices that divde the AP times into sections
                wx = np.argmax(np.diff(pts, prepend = 0))   # Finds which of the sections has the most APs
                aptr = vmpk[pts[wx - 1]:pts[wx]]            # sections out the part of the total list that had the most APs

            else:
                aptr = vmpk

            # Checks the maximum frequency and total duration requirements of the potential AP train
            first_ap = aptr[0] * RES                            # Start time of first peak (s)
            ap_duration = (aptr[-1] - aptr[0]) * RES            # Duration od AP train (s)
            n_ap = len(aptr)                                    # Number of APs in train
            if n_ap == 1:
                peak_rate = 0
            else:
                peak_rate = 1 / (np.min(np.diff(aptr)) * RES)   # Peak firing rate in Hz of train
            
            # Checking against criteria
            if ap_duration > min_dur and peak_rate > min_rate:
                was_train = True
    
        # Formatting outputs
        if was_train:
            aptr_char = [first_ap, ap_duration, n_ap, peak_rate]
        else:
            aptr_char = [0, 0, 0, 0]
        
        return was_train, aptr_char
        
  

def Initialize(NIdx, A):
    """
    Creates neuron objects and connects them using an index an adjacency matrix
    
    Uses the "ConnectNeurons" function
    Also calls upon the global variables "Q" and "SIM_T"
    
    Inputs:
        "NIdx" is a dictionary containing neuron properties for each neuron to be simulated through the "Simulate" function
        "A" is the adacency matrix describing how the neurons in "NIdx" are to be connected to each other
        
    Outputs:
        The first output ("NN") is a list of Neuron objects created according to the properties in "NIdx" and connected to each other based on "A"
        The second and third outputs, "qv" and "tv" respectively, are vectors with "Q"+1 elements each denoting the frame ("qv") and time ("tv")
        
    """
    # Initializes the frame and time vectors
    qv = np.arange(0, Q)
    tv = np.linspace(0, SIM_T, Q + 1)
    
    nNeurons = len(NIdx)

    NN = []

    for n in range(nNeurons):
        NN.append(Neuron(name = NIdx[n]['name'],
                         diameter = NIdx[n]['diam'],
                         axon_length = NIdx[n]['axonl'],
                         refractory_period = NIdx[n]['refractory_period'],
                         activation_threshold = NIdx[n]['activation_threshold'], 
                         I_tonic = NIdx[n]['I_tonic'],
                         sensitivity = NIdx[n]['sens'],
                         myelinated = NIdx[n]['myelinated'],
                         primary_afferent = NIdx[n]['primary_afferent'],
                         nociceptive = NIdx[n]['is_noci'],
                         is_brainstem = NIdx[n]['is_brainstem']))

    
    for n1 in range(nNeurons):
        for n2 in range(nNeurons):
            if not A[n1, n2] == 0:
                ConnectNeurons(NN, n1, n2, A[n1, n2])

    return NN, qv, tv


def Simulate(NN, A, qv, tv, stress, timer = False):
    """
    The simulation function for a network of neurons responding to a tissue stress
    Starts by applying the tissue stresses to each of the peripheral neurons to determine those input currents
    Then goes through the neuronal model frame-by-frame to determine their activations
    
    Uses the "Mech2Current", "AB2Current", "PreAP", and "ActionPotentials" functions
    
    Inputs:
        "NN" is a list of a Neuron objects representing a neuronal network, each neuron object needs to have a simulation q and simulation time matching the inputs "qv" and "tv"
        "A" is the adjacency matrix describing how the neurons in "NN" are connected to each other
        "qv" is the frame-rate vector counter used for indexing, starts at zero and goes to global big Q
        "tv" is the time vector keeping track of the seconds during the simulation, needs one more element than "qv"
        "stress" is the stress over time vector for the tissue in MPa, needs the same number of elements as "tv"
        
    Outputs:
        Doesn't output anything, just updates the membrane potentials (and other simulation variables) in each of the neurons in "NN"
    """
    
    st = time.time()
    
    nNeurons = len(NN)    
    
    # Identify peripheral neurons and apply tissue stresses to them   
    for ns in range(nNeurons):
        if NN[ns].primary_afferent:
            if NN[ns].nociceptive:
                NN[ns].input_current = Mech2Current(stress, tv, NN[ns].sensitivity, NN[ns].myelinated, PSI_NT[ns-1])
            else:
                if max(stress) == 0:
                    NN[ns].input_current = np.zeros(len(tv)) # Catch to silence the AB neuron in the absence of any inputs
                else:
                    NN[ns].input_current = AB2Current(stress)
    
 
    # Goes through frame-by-frame to solve the Hodgkin-Huxley ODEs for each neuron
    # Will fire action potentials when appropriate to activate other neurons in NN
    for q in qv:
        for ns in range(nNeurons):
            NN[ns].ResetNeuron()
            NN[ns].HH(q)
            
            if NN[ns].membrane_potential[q] > NN[ns].activation_threshold and NN[ns].CheckReady():
                
                NN[ns].psp_out[q] = 1
                
                pre_syn, lpostsyn, ow = PreAP(NN, A, ns)
                ActionPotentials(pre_syn, lpostsyn, tv[q], q, ow)
    
    et = time.time()
    
    if timer:
        print(et - st)

    
#%%
""" 5. MODEL SIMULATION SPACE """

Lambda = []
APChar = []

sig_p = 1.0

ss = time.time()

for i in range(P):
    pptv = PPTv[i]
    print((i, pptv))
    
    nidx[1]['sens'], nidx[2]['sens'] = (pptv, pptv)
            
    NN, q, t = Initialize(nidx, Arch)
    st = LinPPT(sig_p * pptv)
    
    Simulate(NN, Arch, q, t, st)
    li, apc = NN[7].APTrain()
    
    Lambda.append(li)
    APChar.append(apc)
       

APChar = pd.DataFrame(APChar, columns = ['First AP (s)', 'Duration (s)', 'n AP', 'Peak Rate (Hz)'])      

print(time.time() - ss)
print('Lambda-Percent: '+str(np.mean(Lambda)))
