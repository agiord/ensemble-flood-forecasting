# ensemble-flood-forecasting
Master's thesis

This repository contains the work I am currently carrying out for my ongoing master thesis in Atmospheric Sciences at the University of Innsbruck (ACINN), in collaboration with the Swiss Federal Research Institute WSL in Zurich, under the supervision of Prof. Dr. Mathias Rotach (ACINN) and Dr. Massimiliano Zappa (WSL).

To produce flood forecasts a meteorological model forcing an hydrological model are coupled and ran; in our case the meteorological model is the ensemble prediction system (EPS) COSMO-E, coupled with the runoff model PREVAH. This modeling chain has already been calibrated for many different catchments in the Alpine region, as the Verzasca basin, located in southern Switzerland, which is considered for this study.

The main goals of this study are:
-

- Investigate and estimate the impacts of different sources of uncertainties present in the modeling chain
- Extend the 'Peak-Box' approach (Zappa et al., 2013), used for interpreting flood forecasts, to multiple peaks events in runoff
- Apply a cluster algorithm to the precipitation ensemble forecast to extract a restricted number of forecasts (Representative Members) and compare the results between the entire set of forecasts and the clustered set
