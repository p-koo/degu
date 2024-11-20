import tensorflow as tf
from tensorflow import keras 
import numpy as np
import gc 
from . import utils

#-----------------------------------------------------------------------------
# DEGU classes
#-----------------------------------------------------------------------------

class DEGU():
    """Deep Ensemble with Gaussian Uncertainty (DEGU).
    
    Manages deep learning model ensembles for:
    - Ensemble prediction and uncertainty estimation
    - Model evaluation and comparison
    - Knowledge distillation to student models
    
    Attributes:
        ensembler: Manager for ensemble models
        num_ensemble: Number of active models
        uncertainty_fun: Function for uncertainty calculation
    """
    
    def __init__(self, ensembler, uncertainty_fun=utils.logvar):
        """Initialize DEGU manager.
        
        Args:
            ensembler: Ensemble model manager
            uncertainty_fun: Uncertainty estimation function
        """
        self.ensembler = ensembler
        self.num_ensemble = ensembler.num_ensemble
        self.uncertainty_fun = uncertainty_fun

    def pred_ensemble(self, x, batch_size=512):
        """Get ensemble predictions with uncertainty.
        
        Args:
            x: Input data tensor
            batch_size: Prediction batch size
            
        Returns:
            tuple: (mean_predictions, uncertainty_estimates, individual_predictions)
        """
        # Use ensembler to get predictions and uncertainty
        ensemble_mean, ensemble_uncertainty, all_preds = self.ensembler.predict(x, batch_size)
        return ensemble_mean, ensemble_uncertainty, all_preds

    def eval_ensemble(self, x, y, eval_fun, batch_size=512):
        """Evaluate ensemble and individual model performance.
        
        Args:
            x: Input data
            y: Target data
            eval_fun: Evaluation function
            batch_size: Batch size
            
        Returns:
            tuple: (ensemble_metrics, individual_model_metrics)
        """
        # Get predictions from all models
        ensemble_mean, ensemble_uncertainty, all_preds = self.ensembler.predict(x, batch_size)

        # Evaluate each model separately
        standard_results = []
        for model_idx in range(self.num_ensemble):
            print('Model %d'%(model_idx + 1))
            standard_results.append(eval_fun(all_preds[model_idx], y))

        # Evaluate ensemble performance
        print('Ensemble')
        ensemble_results = eval_fun(ensemble_mean, y)

        return ensemble_results, standard_results

    def distill_student(self, x_train, y_train, model, train_fun, save_prefix, 
                       validation_data, batch_size=512):
        """Train student model using ensemble knowledge.
        
        Distills ensemble knowledge by training student on:
        - Ensemble mean predictions
        - Uncertainty estimates
        
        Args:
            x_train: Training features
            y_train: Training labels
            model: Student model to train
            train_fun: Training function
            save_prefix: Prefix for saved weights
            validation_data: Validation dataset
            batch_size: Training batch size
            
        Returns:
            tuple: (weights_save_path, training_history)
        """
        # Generate training targets from ensemble
        train_mean, train_unc,_ = self.pred_ensemble(x_train, batch_size=batch_size)
        y_train_ensemble = tf.concat([train_mean, train_unc], axis=1)

        # Generate validation targets
        x_valid, y_valid = validation_data
        valid_mean, valid_unc,_ = self.pred_ensemble(x_valid, batch_size=batch_size)
        y_valid_ensemble = tf.concat([y_valid, valid_unc], axis=1)
        validation_data = [x_valid, y_valid_ensemble]

        # Train and save student model
        history = train_fun(model, x_train, y_train_ensemble, validation_data)
        save_path = save_prefix+'.weights.h5'
        model.save_weights(save_path)
        
        return save_path, history

    def distill_student_dynamic(self, x_train, y_train, model, train_fun, save_prefix, 
                              validation_data, batch_size=512):
        """Train student model with dynamic distillation.
        
        Similar to distill_student but computes targets during training
        rather than pre-computing them.
        
        Args:
            Similar to distill_student method
            
        Returns:
            tuple: (weights_save_path, training_history)
        """
        history = train_fun(model, x_train, y_train, validation_data)
        save_path = save_prefix+'.weights.h5'
        model.save_weights(save_path)
        return save_path, history
    
    def eval_student(self, x, y, student_model, eval_fun, batch_size=512):
        """Evaluate trained student model performance.
        
        Compares student predictions against:
        - Ground truth labels
        - Ensemble uncertainty estimates
        
        Args:
            x: Input data
            y: Target data
            student_model: Trained student
            eval_fun: Evaluation function
            batch_size: Batch size
            
        Returns:
            tuple: (evaluation_metrics, student_predictions, ensemble_predictions)
        """
        # Get ensemble predictions
        test_mean, test_unc,_ = self.pred_ensemble(x, batch_size=512)

        # Create evaluation targets
        y_ensemble = np.concatenate([y, test_unc], axis=1)

        # Evaluate student performance
        pred = student_model.predict(x, batch_size=batch_size)
        results = eval_fun(pred, y_ensemble)
        y_ensemble = np.concatenate([test_mean, test_unc], axis=1)

        return results, pred, y_ensemble


#-----------------------------------------------------------------------------
# Ensemble helper classes 
#-----------------------------------------------------------------------------

class EnsemblerBase():
    """Base ensembler managing deep learning model ensembles.
    
    Core functionality for:
    - Training ensemble models with different initializations
    - Generating ensemble predictions with uncertainty
    - Managing model weights and persistence
    
    Attributes:
        base_model: Template model for ensemble
        weight_paths: Paths to saved model weights
        uncertainty_fun: Function for uncertainty estimation
        num_ensemble: Number of active models
    """
    
    def __init__(self, base_model, weight_paths=[], uncertainty_fun=utils.logvar):
        """Initialize ensemble manager.
        
        Args:
            base_model: Pre-compiled template model
            weight_paths: Paths to model weight files
            uncertainty_fun: Function to compute prediction uncertainty
        """
        self.base_model = base_model    
        self.weight_paths = weight_paths    
        self.uncertainty_fun = uncertainty_fun
        self.num_ensemble = len(weight_paths)       

    def predict(self, x, batch_size=512):
        """Generate ensemble predictions with uncertainty.
        
        Gets predictions from each model and calculates ensemble statistics.
        
        Args:
            x: Input data tensor
            batch_size: Batch size for prediction
            
        Returns:
            tuple: (mean_predictions, uncertainty_estimates, individual_predictions)
        """
        preds = []
        # Get predictions from each model
        for model_idx in range(self.num_ensemble):
            self.base_model.load_weights(self.weight_paths[model_idx])
            preds.append(self.base_model.predict(x, batch_size=batch_size, verbose=False))

        # Calculate ensemble statistics
        preds = tf.stack(preds)
        ensemble_mean = tf.reduce_mean(preds, axis=0)
        ensemble_uncertainty = self.uncertainty_fun(preds, axis=0)
        return ensemble_mean, ensemble_uncertainty, preds
  
    def train(self, x_train, y_train, num_ensemble, train_fun, validation_data, save_prefix, **kwargs):
        """Train multiple models with different initializations.
        
        Args:
            x_train: Training features
            y_train: Training labels
            num_ensemble: Number of models to train
            train_fun: Training function
            validation_data: Validation dataset
            save_prefix: Prefix for saved weights
            **kwargs: Additional training parameters
            
        Returns:
            list: Training histories for each model
        """
        ensemble_history = []
        # Train specified number of models
        for model_idx in range(num_ensemble):
            print('Training model %d'%(model_idx + 1))
            
            # Reset weights to new random values
            self._reinitialize_model_weights()
            
            # Train and save model
            history, save_path = self.train_another_model(
                x_train, y_train, 
                train_fun, 
                validation_data, 
                save_prefix+'_'+str(model_idx), 
                **kwargs
            )
            ensemble_history.append(history)
        return ensemble_history
    
    def train_another_model(self, x_train, y_train, train_fun, validation_data, save_prefix, **kwargs):
        """Train and add a single model to ensemble.
        
        Args:
            Similar to train() method
            
        Returns:
            tuple: (training_history, saved_weights_path)
        """
        # Train model
        history = train_fun(self.base_model, x_train, y_train, validation_data, **kwargs)
        
        # Save weights and update tracking
        save_path = save_prefix+'.weights.h5'
        self.base_model.save_weights(save_path)
        
        self.weight_paths.append(save_path)
        self.num_ensemble += 1
        return history, save_path
        
    def _reinitialize_model_weights(self):
        """Reset model weights to new random values.
        
        Uses layer initializers to randomize weights and biases.
        """
        # Reset each layer's weights using its initializer
        for layer in utils.get_model_layers(self.base_model):
            if hasattr(layer, 'kernel_initializer'):
                if hasattr(layer, 'kernel'):
                    kernel_initializer = layer.kernel_initializer
                    layer.kernel.assign(kernel_initializer(shape=layer.kernel.shape))
                if hasattr(layer, 'bias') and layer.bias is not None:
                    bias_initializer = layer.bias_initializer
                    layer.bias.assign(bias_initializer(shape=layer.bias.shape))



class EnsemblerMixed():
    """Mixed architecture model ensemble with uncertainty estimation.
    
    Manages an ensemble of models with different architectures, enabling flexible
    combination of diverse models for improved predictions and uncertainty estimates.
    
    Attributes:
        model_funs: List of constructor functions for creating models
        weight_paths: Paths to saved model weights
        uncertainty_fun: Function to estimate prediction uncertainty 
        num_ensemble: Number of models in ensemble
        ensemble: List of active model instances
    """
    
    def __init__(self, model_funs=[], weight_paths=[], uncertainty_fun=utils.logvar):
        """Initialize ensemble manager with model constructors and weights.
        
        Args:
            model_funs: List of functions that create model instances
            weight_paths: Paths to saved model weights
            uncertainty_fun: Function for computing prediction uncertainty
        """
        # Initialize core components for ensemble management
        self.uncertainty_fun = uncertainty_fun
        self.model_funs = model_funs    
        self.weight_paths = weight_paths    
        self.num_ensemble = len(weight_paths)
        self.ensemble = []
        
        # Build ensemble if weights are provided
        if len(self.weight_paths) > 0:
            self.build_ensemble()
        
    def build_ensemble(self, model_funs=None, weight_paths=None, **kwargs):
        """Construct ensemble from model functions and weights.
        
        Args:
            model_funs: Optional list to override stored model constructors 
            weight_paths: Optional list to override stored weight paths
            **kwargs: Additional arguments passed to model constructors
        """
        self.ensemble = []  # Reset ensemble list
        
        # Create and initialize each model in ensemble
        for model_idx in range(self.num_ensemble):
            # Use provided or stored model constructor
            if model_funs is None:
                model = self.model_funs[model_idx](**kwargs)
            else:
                model = model_funs[model_idx](**kwargs)
                
            # Compile model and load appropriate weights
            model.compile()
            if weight_paths is None:
                model.load_weights(self.weight_paths[model_idx])
            else:
                model.load_weights(weight_paths[model_idx])
            self.ensemble.append(model)
        
        # Update stored constructors/paths if new ones provided
        if model_funs is not None:
            self.model_funs = model_funs
        if weight_paths is not None:
            self.weight_paths = weight_paths
        
    def predict(self, x, batch_size=512, **kwargs):
        """Generate predictions with uncertainty from ensemble.
        
        Args:
            x: Input data for prediction
            batch_size: Batch size for prediction
            **kwargs: Additional prediction parameters

        Returns:
            tuple: (ensemble_mean, uncertainty, individual_predictions)
        """
        # Collect predictions from each model
        preds = []
        for model_idx in range(self.num_ensemble):
            preds.append(self.ensemble[model_idx].predict(x, batch_size=batch_size, verbose=False))
            
        # Calculate ensemble statistics
        preds = tf.stack(preds)
        ensemble_mean = tf.reduce_mean(preds, axis=0)
        ensemble_uncertainty = self.uncertainty_fun(preds, axis=0)
        return ensemble_mean, ensemble_uncertainty, preds
    
    def train(self, x_train, y_train, model_funs, train_fun, validation_data, save_prefix, **kwargs):
        """Train multiple models for ensemble.
        
        Args:
            x_train: Training features
            y_train: Training labels
            model_funs: List of model constructor functions
            train_fun: Single or list of training functions
            validation_data: Data for validation
            save_prefix: Prefix for saved weights
            **kwargs: Additional training parameters

        Returns:
            list: Training histories for each model
        """
        ensemble_history = []
        
        # Train each model with appropriate constructor and training function
        for model_idx in range(len(model_funs)):
            print('Training model %d'%(model_idx + 1))
    
            # Handle single training function or per-model functions
            if isinstance(train_fun, list):
                history, save_path = self.train_another_model(
                    x_train, y_train,
                    model_funs[model_idx],
                    train_fun[model_idx],
                    validation_data,
                    save_prefix+'_'+str(model_idx),
                    **kwargs
                )
            else:
                history, save_path = self.train_another_model(
                    x_train, y_train,
                    model_funs[model_idx],
                    train_fun,
                    validation_data,
                    save_prefix+'_'+str(model_idx),
                    **kwargs
                )
            ensemble_history.append(history)
        return ensemble_history
    
    def train_another_model(self, x_train, y_train, model_fun, train_fun, 
                          validation_data, save_prefix, **kwargs):
        """Train and add a single model to ensemble.
        
        Args:
            x_train: Training features
            y_train: Training labels
            model_fun: Constructor for new model
            train_fun: Training function
            validation_data: Validation dataset
            save_prefix: Prefix for weight file
            **kwargs: Additional training parameters

        Returns:
            tuple: (training_history, weights_save_path)
        """
        # Create and train new model instance
        base_model = model_fun(**kwargs)
        history = train_fun(base_model, x_train, y_train, validation_data, **kwargs)

        # Save trained model weights
        save_path = save_prefix+'.weights.h5'
        base_model.save_weights(save_path)
        
        # Update ensemble tracking data
        self.model_funs.append(model_fun)
        self.weight_paths.append(save_path)
        self.num_ensemble += 1
        self.ensemble.append(base_model)
        return history, save_path



class EnsemblerDynamic():
    """Dynamic ensemble manager with data augmentation capabilities.
    
    Manages an ensemble of models with shared architecture but different weights.
    Supports dynamic data augmentation during training.
    
    Attributes:
        base_model: Template model for ensemble
        weight_paths: Paths to saved model weights
        uncertainty_fun: Function for uncertainty estimation
        num_ensemble: Number of active models
        ensemble: List of model instances
    """
    
    def __init__(self, base_model, weight_paths=[], uncertainty_fun=utils.logvar):
        """Initialize dynamic ensemble manager.
        
        Args:
            base_model: Base model to clone for ensemble
            weight_paths: Paths to saved weights
            uncertainty_fun: Function for computing uncertainty
        """
        # Initialize core components for ensemble management
        self.uncertainty_fun = uncertainty_fun
        self.base_model = base_model    
        self.weight_paths = weight_paths    
        self.num_ensemble = len(weight_paths)
        self.ensemble = []
        
        # Build ensemble if weights are provided
        if len(self.weight_paths) > 0:
            self.build_ensemble()
        
    def build_ensemble(self, weight_paths=None, **kwargs):
        """Construct ensemble from saved weights.
        
        Args:
            weight_paths: Optional new weight paths
            **kwargs: Additional model parameters
        """
        self.ensemble = []  # Reset ensemble list
        
        # Compile and load weights for each model
        for model_idx in range(self.num_ensemble):
            model = self.base_model
            model.compile()
            if weight_paths is None:
                model.load_weights(self.weight_paths[model_idx])
            else:
                model.load_weights(weight_paths[model_idx])
            self.ensemble.append(model)
            
        # Update paths if new ones provided
        if weight_paths is not None:
            self.weight_paths = weight_paths

    def predict(self, x, batch_size=512, **kwargs):
        """Generate predictions with uncertainty from ensemble.
        
        Args:
            x: Input data
            batch_size: Batch size for prediction
            **kwargs: Additional prediction parameters

        Returns:
            tuple: (ensemble_mean, uncertainty, individual_predictions)
        """
        # Get predictions from all models
        preds = []
        for model_idx in range(self.num_ensemble):
            preds.append(self.ensemble[model_idx].predict(x, batch_size=batch_size, verbose=False))
            
        # Calculate ensemble statistics
        preds = tf.stack(preds)
        ensemble_mean = tf.reduce_mean(preds, axis=0)
        ensemble_uncertainty = self.uncertainty_fun(preds, axis=0)
        return ensemble_mean, ensemble_uncertainty, preds
    
    def train(self, x_train, y_train, num_ensemble, train_fun, validation_data, save_prefix, 
              augment_list=[], max_augs_per_seq=2, hard_aug=True, **kwargs):
        """Train multiple models with data augmentation.
        
        Args:
            x_train: Training features
            y_train: Training labels
            num_ensemble: Number of models to train
            train_fun: Training function
            validation_data: Validation dataset
            save_prefix: Prefix for saved weights
            augment_list: Data augmentation operations
            max_augs_per_seq: Maximum augmentations per sequence
            hard_aug: Use hard augmentation mode
            **kwargs: Additional training parameters

        Returns:
            list: Training histories
        """
        ensemble_history = []
        
        # Train specified number of models
        for model_idx in range(num_ensemble):
            print('Training model %d'%(model_idx + 1))
            
            # Train individual model with augmentation
            history, save_path = self.train_another_model(
                x_train, y_train, 
                train_fun,
                validation_data,
                save_prefix+'_'+str(model_idx),
                augment_list=augment_list,
                max_augs_per_seq=max_augs_per_seq,
                hard_aug=hard_aug,
                **kwargs
            )
            ensemble_history.append(history)
        return ensemble_history
       
    def train_another_model(self, x_train, y_train, train_fun, validation_data, save_prefix,
                          augment_list=[], max_augs_per_seq=2, hard_aug=True, **kwargs):
        """Train and add single model to ensemble.
        
        Args:
            Similar to train() method

        Returns:
            tuple: (training_history, weights_save_path)
        """
        # Reset weights for new model
        self._reinitialize_model_weights()
        
        # Create dynamic model with augmentation
        base_model = DynamicModel(
            self.base_model,
            augment_list=augment_list,
            max_augs_per_seq=max_augs_per_seq,
            hard_aug=hard_aug
        )

        # Train and save model
        history = train_fun(base_model, x_train, y_train, validation_data, **kwargs)
        save_path = save_prefix+'.weights.h5'
        base_model.save_weights(save_path)
        
        # Update ensemble tracking
        self.weight_paths.append(save_path)
        self.num_ensemble += 1
        self.ensemble.append(base_model)
        return history, save_path

    def _reinitialize_model_weights(self):
        """Reset model weights to new random values.
        
        Reinitializes weights and biases for each layer using defined initializers.
        """
        # Reset weights for each layer using initializers
        for layer in get_model_layers(self.base_model):
            if hasattr(layer, 'kernel_initializer'):
                if hasattr(layer, 'kernel'):
                    kernel_initializer = layer.kernel_initializer
                    layer.kernel.assign(kernel_initializer(shape=layer.kernel.shape))
                if hasattr(layer, 'bias') and layer.bias is not None:
                    bias_initializer = layer.bias_initializer
                    layer.bias.assign(bias_initializer(shape=layer.bias.shape))

#-----------------------------------------------------------------------------
# Dynamic model for training model with dynamic augs based on ensemble targets
#-----------------------------------------------------------------------------

class DynamicModel(keras.Model):
    """Dynamic Keras model with augmentation and ensemble capabilities.
    
    Extends keras.Model for advanced training capabilities:
    - Dynamic data augmentation during training/inference
    - Ensemble prediction integration
    - Uncertainty estimation
    - Flexible training/finetuning modes
    
    Attributes:
        model: Base model being wrapped
        ensembler: Optional ensemble manager
        augment_list: List of augmentation functions
        max_augs_per_seq: Maximum augmentations per sequence
        hard_aug: Force maximum augmentations if True
        finetune: Disable augmentation in finetune mode
        inference_aug: Enable inference-time augmentation
        uncertainty: Include uncertainty in predictions
    """
    
    def __init__(self, model, ensembler=None, augment_list=[], max_augs_per_seq=2, 
                 hard_aug=False, finetune=False, inference_aug=False, uncertainty=False, **kwargs):
        """Initialize dynamic model wrapper.
        
        Args:
            model: Base model to wrap
            ensembler: Optional ensemble manager for predictions
            augment_list: List of augmentation functions
            max_augs_per_seq: Maximum augmentations per sequence
            hard_aug: Always use max augmentations if True
            finetune: Start in finetune mode if True
            inference_aug: Use augmentation during inference
            uncertainty: Include uncertainty in predictions
            **kwargs: Additional arguments
        """
        super(DynamicModel, self).__init__()
        
        # Core components
        self.model = model
        self.ensembler = ensembler
        
        # Augmentation configuration
        self.augment_list = augment_list
        self.max_augs_per_seq = tf.math.minimum(max_augs_per_seq, len(augment_list))
        self.max_num_aug = len(augment_list)
        self.hard_aug = hard_aug
        self.inference_aug = inference_aug
        
        # Model behavior settings
        self.finetune = finetune
        self.uncertainty = uncertainty
        self.kwargs = kwargs
        
    def call(self, inputs, training=False):
        """Forward pass through model.
        
        Args:
            inputs: Input tensor
            training: Training mode flag
            
        Returns:
            Model outputs
        """
        return self.model(inputs, training=training)
    
    @tf.function
    def ensemble_predict(self, x):
        """Get ensemble predictions with uncertainty.
        
        Args:
            x: Input tensor
            
        Returns:
            tuple: (mean_prediction, uncertainty_estimate)
        """
        preds = []
        if isinstance(ensembler, EnsemblerMixed):
            # Mixed ensemble: different architectures
            for model_idx in range(self.ensembler.num_ensemble):
                preds.append(self.ensembler.ensemble[model_idx](x))
        else:
            # Standard ensemble: same architecture
            for model_idx in range(self.ensembler.num_ensemble):
                self.ensembler.base_model.load_weights(self.ensembler.weight_paths[model_idx])
                preds.append(self.ensembler.base_model(x))

        # Calculate ensemble statistics
        preds = tf.stack(preds)
        ensemble_mean = tf.reduce_mean(preds, axis=0)
        ensemble_uncertainty = self.ensembler.uncertainty_fun(preds, axis=0)
        return ensemble_mean, ensemble_uncertainty
                
    @tf.function  
    def train_step(self, data):
        """Custom training step with augmentation.
        
        Handles:
        - Data augmentation
        - Ensemble target generation
        - Gradient updates
        
        Args:
            data: Training batch
            
        Returns:
            dict: Updated metrics
        """
        # Unpack data with optional sample weights
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
            
        # Apply augmentation if not in finetune mode
        x = tf.cast(x, tf.float32)
        if not self.finetune:
            x = self._apply_augment(x)

        # Get ensemble targets if using ensembler
        if self.ensembler is not None:
            y_mean, y_uncertainty = self.ensemble_predict(x)
            if self.uncertainty:
                y = tf.concat([y_mean, y_uncertainty], axis=1)  
            else:
                y = y_mean
        else:
            y = tf.cast(y, tf.float32)
    
        # Gradient update
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, batch):
        """Validation/test step with optional augmentation.
        
        Args:
            batch: Validation/test data
            
        Returns:
            dict: Updated metrics
        """
        x, y = batch
        if self.inference_aug:
            x = self._apply_augment(x)
        else:
            x = tf.cast(x, tf.float32)
            
        # Update targets if using ensemble
        if self.ensembler is not None:
            if self.uncertainty:
                y_mean, y_uncertainty = self.ensemble_predict(x)
                y = tf.concat([y, y_uncertainty], axis=1)  
        else:
            y = tf.cast(y, tf.float32)

        y_pred = self(x, training=False)  
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def predict_step(self, batch):
        """Generate predictions with optional augmentation.
        
        Args:
            batch: Input data
            
        Returns:
            Model predictions
        """
        x = batch
        if self.inference_aug:
            x = self._apply_augment(x)
        return self(x)

    @tf.function
    def _apply_augment(self, x):
        """Apply random augmentations to input batch.
        
        Randomly selects and applies N augmentations, where N is:
        - Fixed at max_augs_per_seq if hard_aug=True 
        - Random between 1 and max_augs_per_seq if hard_aug=False
        
        Args:
            x: Input tensor
            
        Returns:
            Augmented tensor
        """
        # Choose number of augmentations
        if self.hard_aug:
            batch_num_aug = tf.constant(self.max_augs_per_seq, dtype=tf.int32)
        else:
            batch_num_aug = tf.random.uniform(shape=[], minval=1, maxval=self.max_augs_per_seq+1, dtype=tf.int32)

        # Select which augmentations to apply
        aug_indices = tf.sort(tf.random.shuffle(tf.range(self.max_num_aug))[:batch_num_aug])

        # Apply selected augmentations in sequence
        ind = 0
        for augment in self.augment_list:
            augment_condition = tf.reduce_any(tf.equal(tf.constant(ind), aug_indices))
            x = tf.cond(augment_condition, lambda: augment(x), lambda: x)
            ind += 1
        return x
    
    def finetune_mode(self, status=True, optimizer=None, lr=None):
        """Configure model for finetuning.
        
        Args:
            status: Enable/disable finetuning mode
            optimizer: Optional new optimizer
            lr: Optional new learning rate
        """
        self.finetune = status
        if optimizer is not None:
            self.optimizer = optimizer
        if lr is not None:
            self.optimizer.learning_rate = lr
    
    def save_weights(self, filepath):
        """Save model weights to file.
        
        Args:
            filepath: Path to save weights
        """
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath):
        """Load weights from file.
        
        Args:
            filepath: Path to weight file
        """
        self.model.load_weights(filepath)




