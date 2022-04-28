# mech-noci-model
A model consolidating 1D tissue strains and neural circuitry to predict nociceptive activity in the brainstem.

Works by simulating a stress-strain timeseries, using that time series to generate input currents in peripheral neurons, then using those inputs to drive a simplified spinal cord circuit.

## Files:
- "model.py" is the main script
- "neuro_dict.py" is a dictionary with the neuron properties for the Neurological Module. Neurons can be added and modified as needed and built using the constructor class in the model.py file

## Use Instructions:
- Put both .py files in the same directory
- Update/confirm the neuron properties in the "neuro_dict.py" file
- In the "model.py" file:
  - Change the variable "P" to indicate the number of iterations desired in Section 1: Constants
  - Use desired "Mechcanical" function to generate a a stress-time curve in MPa that spans from 0 to T seconds and has Q elements; this defaults to the "LinPPT" function in Section 2: Mechanical Section
  - Set variable "st" in Section 5: Model Simulation Space to euqal desired stress-time function from above
- Run the the "model.py" file

## Builds off of:
- Pressure-pain thresholds from Viggiani & Callaghan (2021)
- Mechanosensation model of Gerling et al., (2018)
- Neuron membrane potential model of Hodgkin and Huxley (1952)
- Neuron proerties of Gerstner et al., (2014) Hursh (1939), Kandel et al (2013)
- Connectivity from Braz et al., (2014), Guo and Hu (2014), and Cordero-Erausquin et al., (2016)

## References:
1. Braz, J.M., Solorzano, C., Wang, X., Basbaum, A.I., 2014. Transmitting Pain and Itch Messages: A Contemporary View of the Spinal Cord Circuits that Generate Gate Control. Neuron 82, 522–536. https://doi.org/10.1016/j.neuron.2014.01.018
2. Cordero-Erausquin, M., Inquimbert, P., Schlichter, R., Hugel, S., 2016. Neuronal networks and nociceptive processing in the dorsal horn of the spinal cord. Neuroscience 338, 230–247. https://doi.org/10.1016/j.neuroscience.2016.08.048
3. Gerling, G.J., Wan, L., Hoffman, B.U., Wang, Y., Lumpkin, E.A., 2018. Computation predicts rapidly adapting mechanotransduction currents cannot account for tactile encoding in Merkel cell-neurite complexes. PLoS Comput. Biol. 14, 1–21. https://doi.org/10.1371/journal.pcbi.1006264
4. Gerstner, W., Kistler, W.M., Naud, R., Paninski, L., 2014. Neuronal Dynamics: From single neurons to networks and models of cognition and beyond, 1st ed. Cambridge University Press, Cambridge, UK.
5. Guo, D., Hu, J., 2014. Spinal presynaptic inhibition in pain control. Neuroscience 283, 95–106. https://doi.org/10.1016/j.neuroscience.2014.09.032
6. Hodgkin, A.L., Huxley, A.F., 1952. A quantitative description of membrane current and its application to conduction and excitation in nerve. J. Physiol. 117, 500–544. https://doi.org/10.1113/jphysiol.1952.sp004764
7. Hursh, J.B., 1939. Conduction velocity and diameter of nerve fibers. Am. J. Physiol. 127, 131–139.
8. Kandel, E., Schwartz, J.H., Jessell, T.M., Siegelbaum, S.A., Hudspeth, A.J., 2013. Principles of Neural Science, 5th ed. McGraw-Hill Companies Inc., New York.
9. Viggiani, D., Callaghan, J.P., 2021. Interrelated hypoalgesia, creep, and muscle fatigue following a repetitive trunk flexion exposure. J. Electromyogr. Kinesiol. 57, 102531. https://doi.org/10.1016/j.jelekin.2021.102531
