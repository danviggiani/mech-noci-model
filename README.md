# mech-noci-model
A model consolidating 1D tissue strains and neural circuitry to predict nociceptive activity in the brainstem.

Works by simulating a stress-strain timeseries, using that time series to generate input currents in peripheral neurons, then using those inputs to drive a simplified spinal cord circuit.

Files:
- "model.py" is the main script
- "neuro_dict.py" is a dictionary with the neuron properties for the Neurological Module. Neurons can be added and modified as needed and built using the constructor class in the model.py file

Use Instructions:
- Put both .py files in the same directory
- Update/confirm the neuron properties in the "neuro_dict.py" file
- In the "model.py" file:
  - Change the variable "P" to indicate the number of iterations desired in Section 1: Constants
  - Use desired "Mechcanical" function to generate a a stress-time curve in MPa that spans from 0 to T seconds and has Q elements; this defaults to the "LinPPT" function in Section 2: Mechanical Section
  - Set variable "st" in Section 5: Model Simulation Space to euqal desired stress-time function from above
- Run the the "model.py" file

Builds off of:
- Indestructible Ligament Simplification to represent tissue stress/strains from Barrett and Callaghan (2018)
- Mechanosensation model of Gerling et al., (2018)
- Neuron membrane potential model of Hodgkin and Huxley (1952)
