# Co-Simulation-of-Dynamic-Stall
Data, SSNN models and Matlab codes used in the MA1 Project

Quick explenation of the files:

- Data is a folder containing the wind tunnel measurements.
- Models is a folder containing the SSNN models for the application of chapter 4.1, called "SSNN_model_Pieter", and chapter 4.2, called "CycleToCycleVariations". 
- UKF.mat and EKF.m are the files containing the algorithms used in chapter 4.1.
- SSNN.m is the function of the State-Space Neural Network model.
- Linearize.m is the function that linearizes the SSNN model's equation for the EKF algorithm.
- CycleToCyclevariability.m is a file showing the cycle variability of three concecutive cycles with the results of the KF applied to them.
- Application1_EKF.m and Application1_UKF.m are the EKF and UKF algorithms for the application discussed in chapter 4.2.
- Application2_EKF.m and Application2_UKF.m is a second considered application that was not implemented in the final report. An explanation is given in the "SSNN_CycleToCycleVariations.pdf" file, which can ve found in ../Models/CycleToCycleVariations folder.
