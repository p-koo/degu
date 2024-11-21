 repository


# DEGU: Distilling Ensembles of Genomic Uncertainty-aware models (DEGU)

This repository contains the implementation of **DEGU**, a framework for leveraging deep ensemble models to improve uncertainty estimation, predictive performance, and interpretability in deep learning. 

## Features
- **Ensemble Predictions**: Generate mean predictions and uncertainty estimates from ensembles of models.
- **Uncertainty Estimation**: Employ Gaussian uncertainty metrics to quantify model reliability.
- **Knowledge Distillation**: Transfer ensemble knowledge to a single student model for computational efficiency.
- **Dynamic Data Augmentation**: Improve training robustness through augmentations.
- **Flexible Ensemble Management**: Support for homogeneous and heterogeneous model architectures.
- **Comprehensive Metrics**: Evaluate model performance using metrics like MSE, Pearson, and Spearman correlations.

## Core Components
### 1. `degu.py`
This is the main script containing the core DEGU classes and functionalities:
- **`Distiller`**: Handles ensemble prediction, evaluation, and knowledge distillation.
- **`EnsemblerBase`**: Manages ensembles with a shared base architecture.
- **`EnsemblerMixed`**: Supports ensembles with different architectures.
- **`EnsemblerDynamic`**: Incorporates dynamic data augmentation during ensemble training.

### 2. `train.py`
Includes functions for training models with custom settings:
- **`train_standard_fun`**: Standard training with early stopping and learning rate scheduling.
- Example usage:
    ```python
    from train import train_standard_fun
    history = train_standard_fun(
        model, x_train, y_train, validation_data, 
        loss='mse', max_epochs=100, batch_size=100, 
        initial_lr=0.001, es_patience=10, lr_decay=0.1, lr_patience=5
    )
    ```

### 3. `metrics.py`
Implements utilities for model evaluation:
- **`eval_regression`**: Evaluate regression models using metrics such as MSE, Pearson, and Spearman.
- Example usage:
    ```python
    from metrics import eval_regression
    metrics = eval_regression(predictions, true_values)
    print("MSE, Pearson, Spearman:", metrics)
    ```

### 4. `utils.py`
Provides helper functions for uncertainty calculation:
- **`logvar`**: Calculate logarithmic variance along a specified axis.
- **`std`**: Compute standard deviation.
- Example usage:
    ```python
    from utils import logvar, std
    log_variance = logvar(predictions, axis=0)
    standard_deviation = std(predictions, axis=0)
    ```

## Installation
Ensure that you have Python 3.8+ and TensorFlow 2.x installed. Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/degu.git
cd degu
pip install -r requirements.txt
```

## Usage

### Scenario 1: Build Ensemble in Serial
Train an ensemble of models using the same architecture and distill knowledge into a student model.

```python
from degu import EnsemblerBase, Distiller, utils
from tensorflow import keras

# Define the base model
def DeepSTARR(input_shape=(249, 4), output_shape=2):
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(256, kernel_size=7, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(output_shape, activation='linear')(x)
    return keras.Model(inputs=inputs, outputs=outputs)

# Train ensemble models
ensembler = EnsemblerBase(base_model=DeepSTARR(), uncertainty_fun=utils.logvar)
ensemble_history = ensembler.train(
    x_train, y_train, num_ensemble=5, 
    train_fun=train_standard_fun, 
    validation_data=(x_valid, y_valid), 
    save_prefix='deepstarr'
)

# Distill to student model
student_model = DeepSTARR(output_shape=4)
distiller = Distiller(ensembler, uncertainty_fun=utils.logvar)
student_path, history = distiller.distill_student(
    x_train, y_train, model=student_model, 
    train_fun=train_standard_fun, 
    validation_data=(x_valid, y_valid), 
    save_prefix='deepstarr_distilled'
)
```

### Scenario 2: Build Ensemble in Parallel
Train models in parallel and combine their predictions into an ensemble.

```python
weight_paths = ['model_{}.h5'.format(i) for i in range(5)]
ensembler = EnsemblerBase(base_model=DeepSTARR(), weight_paths=weight_paths, uncertainty_fun=utils.logvar)

# Evaluate ensemble and distill to student model
ensemble_results, _ = ensembler.eval_ensemble(x_test, y_test, eval_function)
student_model = DeepSTARR(output_shape=4)
distiller = Distiller(ensembler, uncertainty_fun=utils.logvar)
distiller.distill_student(x_train, y_train, model=student_model, train_fun=train_standard_fun)
```

### Scenario 3: Train with Dynamic Augmentations
Use EvoAug to improve ensemble diversity and robustness.

```python
from degu import train, EnsemblerDynamic
augment_list = [
    augment.RandomDeletion(delete_min=0, delete_max=25),
    augment.RandomTranslocationBatch(shift_min=0, shift_max=25),
    augment.RandomNoise(noise_mean=0, noise_std=0.25),
    augment.RandomMutation(mutate_frac=0.05)
]
ensembler = EnsemblerDynamic(base_model=DeepSTARR(), uncertainty_fun=utils.logvar)
ensemble_history = ensembler.train(
    x_train, y_train, num_ensemble=5, 
    train_fun=train.train_dynamic_aug_fun, 
    validation_data=(x_valid, y_valid), 
    save_prefix='deepstarr_evoaug', 
    augment_list=augment_list
)


```

## Citation
If you use this framework in your research, please cite:

```plaintext
[Provide citation details from the attached document here]
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

