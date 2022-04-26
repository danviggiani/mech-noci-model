# mech-noci-model
A model consolidating 1D tissue strains and neural circuitry to predict nociceptive activity in the brainstem.

Works by simulating a stress-strain timeseries, using that time series to generate input currents in peripheral neurons, then using those inputs to drive a simplified spinal cord circuit.

Builds off of:
- Indestructible Ligament Simplification to represent tissue stress/strains from Barrett and Callaghan (2018)
- Mechanosensation model of Gerling et al., (2018)
- Neuron membrane potential model of Hodgkin and Huxley (1952)
