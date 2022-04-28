# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:17:26 2022

@author: danvi
"""
import random

"""
Index of neurons simulated

For each neuron object:
    - "name" A string label for the user to see during plotting
    - "diam" Axon diameter in um
    - "axonl" Axon length in mm
    - "refractory_period" Refractory period in ms
    - "activation_threshold" Activation threshold in mV
    - "sens" Sensitivity in MPa, only called if "primary_afferent" is True
    - "I_tonic" Tonic current input in nA
    - "myelinated" Boolean; True if axon is myelinated
    - "primary_afferent" Boolean; True if aprimary afferent neuron
    - "is_noci" Boolean; True if it carries nociceptive information
    - "is_brainstem" Boolean; True if it ends the circuit (is in the brainstem)


"""

NIDX = [{'name' : 'AB', 'diam' : random.gauss(9.0, 1.5),    #N0, Primary AB
         'axonl' : random.randint(95, 102), 
         'refractory_period' : random.randint(5, 7) / 1000, 
         'activation_threshold' : random.gauss(30.0, 2.5),
         'sens' : 0.00001,'I_tonic' : 0, 'myelinated' : True, 
         'primary_afferent' : True, 'is_noci' : False, 'is_brainstem' : False},
        {'name' : 'Ad', 'diam' : random.gauss(3.5, 1.0),    #N1 Primary Ad
         'axonl' : random.randint(95, 102),
         'refractory_period' : random.randint(7, 12) / 1000,
         'activation_threshold' : random.gauss(30.0, 2.5), 
         'sens' : 0.5, 'I_tonic' : 0, 'myelinated' : True, 
         'primary_afferent' : True, 'is_noci' : True, 'is_brainstem' : False},   
        {'name' : 'Cn', 'diam' : random.gauss(0.3, 0.1),    #N2 Primary C
         'axonl' : random.randint(95, 102),
         'refractory_period' : random.randint(7, 12) / 1000,
         'activation_threshold' : random.gauss(30.0, 2.5),
         'sens' : 0.5, 'I_tonic' : 0, 'myelinated' : False,
         'primary_afferent' : True, 'is_noci' : True, 'is_brainstem' : False},    
        {'name' : 'PN', 'diam' : random.gauss(7.0, 1.5),    #N3 Projection Neuron
         'axonl' : random.randint(950, 1025),
         'refractory_period' : random.randint(5, 10) / 1000,
         'activation_threshold' : random.gauss(30.0, 2.5),
         'sens' : 0.5, 'I_tonic' : 0, 'myelinated' : True,
         'primary_afferent' : False, 'is_noci' : True, 'is_brainstem' : False},
        {'name' : 'VC', 'diam' : random.gauss(7.0, 1.5),    #N4 Vertical Cell
         'axonl' : random.randint(5, 40),
         'refractory_period' : random.randint(5, 10) / 1000,
         'activation_threshold' : random.gauss(30.0, 2.5),
         'sens' : 0.5, 'I_tonic' : 0, 'myelinated' : True,
         'primary_afferent' : False, 'is_noci' : True, 'is_brainstem' : False},
        {'name' : 'II', 'diam' : random.gauss(7.0, 1.5),    #N5 Secondary Inhibitor
         'axonl' : random.randint(5, 40),
         'refractory_period' : random.randint(5, 10) / 1000,
         'activation_threshold' : random.gauss(30.0, 2.5),
         'sens' : 0.5, 'I_tonic' : 0, 'myelinated' : True,
         'primary_afferent' : False, 'is_noci' : True, 'is_brainstem' : False},
        {'name' : 'TI', 'diam' : random.gauss(7.0, 1.5),    #N6 Tonic Inhibitor
         'axonl' : random.randint(5, 40),
         'refractory_period' : random.randint(5, 10) / 1000,
         'activation_threshold' : random.gauss(30.0, 2.5),
         'sens' : 0.5, 'I_tonic' : 1.8, 'myelinated' : True,
         'primary_afferent' : False, 'is_noci' : True, 'is_brainstem' : False},
        {'name' : 'BN', 'diam' : random.gauss(7.0, 1.5),    #N7 Brainstem
         'axonl' : random.randint(5, 40),
         'refractory_period' : random.randint(5, 10) / 1000,
         'activation_threshold' : random.gauss(30.0, 2.5),
         'sens' : 0.5, 'I_tonic' : 0, 'myelinated' : True,
         'primary_afferent' : False, 'is_noci' : True, 'is_brainstem' : True}]
