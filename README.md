# KAN-ROM: A Novel Approach for High-Dimensional Fluid Dynamics Prediction

This repository contains the code and documentation for the Kolmogorov–Arnold Network-based Model with Reduced Order Modelling (KAN-ROM). This model leverages machine learning techniques to predict high-dimensional fluid dynamics, specifically targeting the Shallow Water Model (SWM) and Compressible Navier-Stokes (CNS) datasets.

**Overview of this project**
The KAN-ROM framework integrates a Kolmogorov-Arnold Network (KAN) with a Convolutional Autoencoder (CAE) for dimensionality reduction, followed by a Multi-Layer KAN for time-series forecasting. This combination enhances the prediction accuracy of complex spatiotemporal dynamics while significantly reducing the computational cost compared to traditional methods.

### Key Features:
- **Dimensionality Reduction**: Uses CAE to compress high-dimensional data.
- **Time-Series Prediction**: Multi-layer KAN efficiently forecasts future time steps within the latent space.
- **High Accuracy**: Outperforms conventional machine learning models like MLP and LSTM on fluid dynamics datasets.

### Datasets
- **Shallow Water Model (SWM)**: Simulates large-scale atmospheric and oceanic fluid dynamics, which is available in code.
- **Compressible Navier-Stokes (CNS)**: Models high-speed fluid flows where changes in fluid density are crucial. Which is available on:[https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)



