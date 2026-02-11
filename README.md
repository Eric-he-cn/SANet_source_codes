# Source Codes for SANet

This repository provides the reference implementation for the GLOBECOM 2025 paper **"SANet: Sensing-Aided Beamforming for LEO Satellite-Ground Communications"**.

## 1. Paper Overview

Low Earth-Orbit (LEO) satellite-ground links suffer from severe Doppler shifts, which significantly degrades instantaneous CSI estimation and beamforming performance. The proposed **SANet** is an end-to-end sensing-aided deep learning framework that leverages an ISAC (Integrated Sensing and Communications) design to extract relative velocity information from radar echoes and uses it to adapt the beamforming network weights via a sensing-aided hypernetwork.

SANet consists of four concatenated modules trained end-to-end:

-   Downlink pilot training network
-   Uplink feedback quantization DNN
-   Beamforming RNN
-   Sensing Hypernetwork

Simulation results show that SANet achieves approximately 25% sum-rate improvement over the SOTA DNN-based baseline under identical settings, with ZF beamforming included as a classical baseline.


## 2. Repository Structure

    .
    ├── main.py            # Entry script
    ├── CE_test.py         # Channel estimation experiments
    ├── BF_test.py         # Beamforming experiments
    ├── helper.py          # Core NN blocks (DNN/Hypernet/RNN)
    ├── net_train.py       # Training loops for CE
    └── net_train2.py      # Training loops for BF




## 3. Environment & Dependencies

This project is implemented with **TensorFlow 1.x** (e.g., `tf.placeholder`, `tf.reset_default_graph`, `tf.Session`) and uses `tensorflow.contrib.layers.xavier_initializer`, so TensorFlow 2.x is not recommended unless using `tf.compat.v1` compatibility mode.

### Recommended setup

-   Python: 3.6
-   TensorFlow: 1.14
-   NumPy
-   SciPy

### Example installation

``` bash
pip install numpy scipy
pip install tensorflow==1.14.0
```


## 4. How to Run

### 4.1 Run ALL baseline
You can directly run:
``` bash
python main.py
```

### 4.2 Run CE experiment

You can directly run:

``` bash
python CE_test.py
```

### 4.3 Run Beamforming experiments

You can directly run:

``` bash
python BF_test.py
```


## 5. Outputs

During training/testing, the scripts periodically save metrics to MATLAB `.mat` files via `scipy.io.savemat`.


## 6. Reproducibility Notes

-   Random seeds are set in several training functions to improve reproducibility.
-   Channel generation follows the time-varying LEO satellite-ground model used in the paper, including Doppler/velocity generation for the hypernetwork input.



