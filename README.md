<p align="center">
  <h2 align="center">Neural Netowk based Safety Metric for Autonomous Vehicles</h2>
  <p align="center">
  </p>
</p>

## Introduction
Safety metrics are crucial for autonomous vehicles (AVs) as they provide early warnings to human drivers and aid in the development of AV technologies. Numerous safety metrics exist in the literature; however, their underlying assumptions are often overly simplistic, which impacts their precision and accuracy. To address this issue, I developed a neural network-based safety metric that uses collected trajectory data without assuming specific driving behaviors. In a case study involving a three-lane highway, the proposed neural network-based safety metric was compared with four existing safety metrics: Safety Metric based on the Assessment of Risk, Model Predictive Instantaneous Safety Metric, Pegasus Criticality Measure, and time-to-collision. The results indicate that the neural network-based safety metric achieves superior precision and accuracy. The detailed report is [here](docs/STATS507_report.pdf).

## Installation
- Clone the repository
- Create the [conda](https://docs.conda.io/en/latest/miniconda.html) environment by running 
    ```bash
    conda create -n nn_metric python=3.9
    conda activate nn_metric
    ```
- Install the dependencies. Run 
    ```bash
    pip install -r requirements.txt
    ```
- Download the dataset from the [link](https://drive.google.com/drive/u/1/folders/1_knBQmKAUFFyHr6pddmPqTIUha_VatJP) and extract it to the `dataset` folder. The folder structure should look like this:
    ```
    dataset
    |── checkpoints_neural_network_based_safety_metric_sota
    |   ├── best_ckpt.pt # the best checkpoint of the trained model
    |   ├── logfile.json # the log file of the training process
    ├── neuralmetric
    │   ├── testing # the testing data
    |   |   ├── crash
    |   |   |   ├── test_ep.fcd.json
    |   |   ├── safe
    |   |   |   ├── test_ep.fcd.npy
    |   ├── testing_results_nnmetric # the testing results of the neural network based safety metric
    |   |   ├── test_crash_data_NNMetric.npy
    |   |   ├── test_safe_data_NNMetric.npy
    |   ├── testing_results_others # the testing results of other four safety metrics
    |   |   ├── test_crash_data_gt.npy
    |   |   ├── test_crash_data_MPrISM.npy
    |   |   ├── test_crash_data_PCM.npy
    |   |   ├── test_crash_data_SMAR.npy
    |   |   ├── test_crash_data_TTC.npy
    |   |   ├── test_safe_data_gt.npy
    |   |   ├── test_safe_data_MPrISM.npy
    |   |   ├── test_safe_data_PCM.npy
    |   |   ├── test_safe_data_SMAR.npy
    |   |   ├── test_safe_data_TTC.npy
    │   ├── training # the training data
    │   │   ├── negative
    |   |   |   ├── state_test.npy
    |   |   |   ├── state_train.npy
    |   |   |   ├── value_test.npy
    |   |   |   ├── value_train.npy
    │   │   ├── positive
    |   |   |   ├── state_test.npy
    |   |   |   ├── state_train.npy
    |   |   |   ├── value_test.npy
    |   |   |   ├── value_train.npy
    ```

## Usage
### Training

Please refer to [neural_network_based_safety_metric.yaml](core/configs/neural_network_based_safety_metric.yaml) for configurations. For training the neural network based safety metric, run
```bash
python core/train.py --yaml_conf core/configs/neural_network_based_safety_metric.yaml
```
To visualize the training process, please refer to the [Jupyter Notebook](analysis/viz_training_results.ipynb).

## Testing
Please run the following command to test the trained model using the recoded test data.
```bash
python core/inference.py --yaml_conf core/configs/neural_network_based_safety_metric.yaml --checkpoint dataset/checkpoints_neural_network_based_safety_metric_sota/best_ckpt.pt --data_folder dataset/neuralmetric/testing
```
The results will be saved in the `dataset/neuralmetric/testing_results_nnmetric` folder. 

The testing results of other four safety metrics (i.e., SMAR, MPrISM, PCM, and TTC) are stored in the `dataset/neuralmetric/testing_results_others` folder. 

Please refer to the [Jupyter Notebook](analysis/pr_curve_comparison.ipynb) for visualization of the comparison results.

## License

This software is distributed under the [MIT License](LICENSE).

## Contact
Haojie Zhu - zhuhj@umich.edu