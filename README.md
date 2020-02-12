# ensemble-flood-forecasting
Master's thesis

This repository contains the work I carried out for the master thesis in Atmospheric Sciences at the University of Innsbruck (ACINN), in collaboration with the Swiss Federal Research Institute WSL in Zurich, under the supervision of Prof. Dr. Mathias Rotach (ACINN) and Dr. Massimiliano Zappa (WSL).

To produce flood forecasts a meteorological model forcing an hydrological model are coupled and ran; in our case the meteorological model is the ensemble prediction system (EPS) COSMO-E, coupled with the runoff model PREVAH. This modeling chain has already been calibrated for many different catchments in the Alpine region, as for the Verzasca basin, located in southern Switzerland, which is considered for this study.

The main open questions we wanted to address are:
-

(1) To what extent does the enhanced resolution of the forcing meteorological model impact on the resulting flood forecast uncertainty?

(2) Is it possible to develop a "Peak-box" approach for detecting multiple flood peaks within the same runoff ensemble forecast, and does it actually outperform the former method of Zappa et al. (2013)? (more details in the [peakbox  repository](https://github.com/agiord/peakbox))

(3) What is the impact of a reduced number of meteorological ensemble members on the flood forecast results?
