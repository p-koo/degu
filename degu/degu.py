import tensorflow as tf
from tensorflow import keras 
import numpy as np
import gc 

#-----------------------------------------------------------------------------
# DEGU classes
#-----------------------------------------------------------------------------

class DEGU():

    def __init__(self, ensembler, uncertainty_fun=logvar):
        self.ensembler = ensembler
        self.num_ensemble = ensembler.num_ensemble
        self.uncertainty_fun = uncertainty_fun

    def pred_ensemble(self, x, batch_size=512):
        ensemble_mean, ensemble_uncertainty, all_preds = self.ensembler.predict(x, batch_size)
        return ensemble_mean, ensemble_uncertainty, all_preds

    def eval_ensemble(self, x, y, eval_fun, batch_size=512):
        ensemble_mean, ensemble_uncertainty, all_preds = self.ensembler.predict(x, batch_size)

        # individual model performance
        standard_results = []
        for model_idx in range(self.num_ensemble):
            print('Model %d'%(model_idx + 1))
            standard_results.append(eval_fun(all_preds[model_idx], y))

        # ensemble performance
        print('Ensemble')
        ensemble_results = eval_fun(ensemble_mean, y)

        return ensemble_results, standard_results

    def distill_student(self, x_train, y_train, model, train_fun, save_prefix, 
                        validation_data, batch_size=512):

        # generate new training labels based on ensemble
        train_mean, train_unc,_ = self.pred_ensemble(x_train, batch_size=batch_size)
        y_train_ensemble = tf.concat([train_mean, train_unc], axis=1)

        # generate validation labels (original activity, ensemble uncertainty)
        x_valid, y_valid = validation_data
        valid_mean, valid_unc,_ = self.pred_ensemble(x_valid, batch_size=batch_size)
        y_valid_ensemble = tf.concat([y_valid, valid_unc], axis=1)
        validation_data = [x_valid, y_valid_ensemble]

        history = train_fun(model, x_train, y_train_ensemble, validation_data)
        
        # save model
        save_path = save_prefix+'.weights.h5'
        model.save_weights(save_path)
        
        return save_path, history

    
    def distill_student_dynamic(self, x_train, y_train, model, train_fun, save_prefix, 
                        validation_data, batch_size=512):

        history = train_fun(model, x_train, y_train, validation_data)
        
        # save model
        save_path = save_prefix+'.weights.h5'
        model.save_weights(save_path)
        
        return save_path, history
    
    def eval_student(self, x, y, student_model, eval_fun, batch_size=512):

        # generate ensemble distribution stats
        test_mean, test_unc,_ = self.pred_ensemble(x, batch_size=512)

        # construct labels (true y, uncertainty y)
        y_ensemble = np.concatenate([y, test_unc], axis=1)

        # test performance
        pred = student_model.predict(x, batch_size=batch_size)
        results = eval_fun(pred, y_ensemble)
        y_ensemble = np.concatenate([test_mean, test_unc], axis=1)

        return results, pred, y_ensemble


#-----------------------------------------------------------------------------
# Ensemble helper classes 
#-----------------------------------------------------------------------------


class EnsemblerBase():
    
    def __init__(self, base_model, weight_paths=[], uncertainty_fun=logvar):
        # base_model is compiled model 

        self.base_model = base_model    
        self.weight_paths = weight_paths    
        self.uncertainty_fun = uncertainty_fun
        self.num_ensemble = len(weight_paths)       

    def predict(self, x, batch_size=512):
        
        preds = []
        for model_idx in range(self.num_ensemble):

            # load weights
            self.base_model.load_weights(self.weight_paths[model_idx])
            
            # get predictions for model
            preds.append(self.base_model.predict(x, batch_size=batch_size, verbose=False))

        # calculate ensemble distribution
        preds = tf.stack(preds)
        ensemble_mean = tf.reduce_mean(preds, axis=0)
        ensemble_uncertainty = self.uncertainty_fun(preds, axis=0)
        return ensemble_mean, ensemble_uncertainty, preds

  
    def train(self, x_train, y_train, num_ensemble, train_fun, validation_data, save_prefix, **kwargs):
 
        ensemble_history = []
        for model_idx in range(num_ensemble):
            print('Training model %d'%(model_idx + 1))
           
            # reset initializations
            self._reinitialize_model_weights()
            
            # train model
            history, save_path = self.train_another_model(x_train, y_train, 
                                                          train_fun, 
                                                          validation_data, 
                                                          save_prefix+'_'+str(model_idx), 
                                                          **kwargs)
            # store results
            ensemble_history.append(history)
        return ensemble_history

    
    def train_another_model(self, x_train, y_train, train_fun, validation_data, save_prefix, **kwargs):

        # train model
        history = train_fun(self.base_model, x_train, y_train, validation_data, **kwargs)

        # save weights
        save_path = save_prefix+'.weights.h5'
        self.base_model.save_weights(save_path)
        
        self.weight_paths.append(save_path)
        self.num_ensemble += 1
        return history, save_path
        
    
    def _reinitialize_model_weights(self):
        # reinitialize weights and biases for each layer
        for layer in get_model_layers(self.base_model):
            if hasattr(layer, 'kernel_initializer'):
                if hasattr(layer, 'kernel'):
                    kernel_initializer = layer.kernel_initializer
                    layer.kernel.assign(kernel_initializer(shape=layer.kernel.shape))
                if hasattr(layer, 'bias') and layer.bias is not None:
                    bias_initializer = layer.bias_initializer
                    layer.bias.assign(bias_initializer(shape=layer.bias.shape))

                    


class EnsemblerMixed():
    
    def __init__(self, model_funs=[], weight_paths=[], uncertainty_fun=logvar):

        self.uncertainty_fun = uncertainty_fun
        self.model_funs = model_funs    
        self.weight_paths = weight_paths    
        self.num_ensemble = len(weight_paths)
        self.ensemble = []
        if len(self.weight_paths) > 0:
            self.build_ensemble()
        
    def build_ensemble(self, model_funs=None, weight_paths=None, **kwargs):
        self.ensemble = []
        for model_idx in range(self.num_ensemble):
            if model_funs is None:
                model = self.model_funs[model_idx](**kwargs)
            else:
                model = model_funs[model_idx](**kwargs)
            model.compile()

            if weight_paths is None:
                model.load_weights(self.weight_paths[model_idx])
            else:
                model.load_weights(weight_paths[model_idx])
            self.ensemble.append(model)
        
        if model_funs is not None:
            self.model_funs = model_funs
        if weight_paths is not None:
            self.weight_paths = weight_paths
        

    def predict(self, x, batch_size=512, **kwargs):
        preds = []
        for model_idx in range(self.num_ensemble):
            preds.append(self.ensemble[model_idx].predict(x, batch_size=batch_size, verbose=False))
        preds = tf.stack(preds)
        ensemble_mean = tf.reduce_mean(preds, axis=0)
        ensemble_uncertainty = self.uncertainty_fun(preds, axis=0)
        return ensemble_mean, ensemble_uncertainty, preds
    
    def train(self, x_train, y_train, model_funs, train_fun, validation_data, save_prefix, **kwargs):
 

        ensemble_history = []
        for model_idx in range(len(model_funs)):
            print('Training model %d'%(model_idx + 1))
    
            # train model
            if isinstance(train_fun, list):
                history, save_path = self.train_another_model(x_train, y_train, 
                                                              model_funs[model_idx], 
                                                              train_fun[model_idx], 
                                                              validation_data, 
                                                              save_prefix+'_'+str(model_idx), 
                                                              **kwargs)
            else:
                history, save_path = self.train_another_model(x_train, y_train, 
                                                              model_funs[model_idx], 
                                                              train_fun, 
                                                              validation_data, 
                                                              save_prefix+'_'+str(model_idx), 
                                                              **kwargs)
                
            # save weights
            ensemble_history.append(history)
        return ensemble_history
    
       
    def train_another_model(self, x_train, y_train, model_fun, train_fun, 
                            validation_data, save_prefix, **kwargs):

        # train model
        base_model = model_fun(**kwargs)
        history = train_fun(base_model, x_train, y_train, validation_data, **kwargs)

        # save weights
        save_path = save_prefix+'.weights.h5'
        base_model.save_weights(save_path)
        
        self.model_funs.append(model_fun)
        self.weight_paths.append(save_path)
        self.num_ensemble += 1
        self.ensemble.append(base_model)
        return history, save_path



class EnsemblerDynamic():
    
    def __init__(self, base_model, weight_paths=[], uncertainty_fun=logvar):

        self.uncertainty_fun = uncertainty_fun
        self.base_model = base_model    
        self.weight_paths = weight_paths    
        self.num_ensemble = len(weight_paths)
        self.ensemble = []
        if len(self.weight_paths) > 0:
            self.build_ensemble()
        
    def build_ensemble(self, weight_paths=None, **kwargs):
        self.ensemble = []
        for model_idx in range(self.num_ensemble):
            model = self.base_model
            model.compile()
            if weight_paths is None:
                model.load_weights(self.weight_paths[model_idx])
            else:
                model.load_weights(weight_paths[model_idx])
            self.ensemble.append(model)
        if weight_paths is not None:
            self.weight_paths = weight_paths
        

    def predict(self, x, batch_size=512, **kwargs):
        preds = []
        for model_idx in range(self.num_ensemble):
            preds.append(self.ensemble[model_idx].predict(x, batch_size=batch_size, verbose=False))
        preds = tf.stack(preds)
        ensemble_mean = tf.reduce_mean(preds, axis=0)
        ensemble_uncertainty = self.uncertainty_fun(preds, axis=0)
        return ensemble_mean, ensemble_uncertainty, preds
    
    
    def train(self, x_train, y_train, num_ensemble, train_fun, validation_data, save_prefix, 
              augment_list=[], max_augs_per_seq=2, hard_aug=True, **kwargs):
 
        ensemble_history = []
        for model_idx in range(num_ensemble):
            print('Training model %d'%(model_idx + 1))
    
            # train model
            history, save_path = self.train_another_model(x_train, y_train, 
                                                          train_fun, 
                                                          validation_data, 
                                                          save_prefix+'_'+str(model_idx), 
                                                          **kwargs)
                
            # save weights
            ensemble_history.append(history)
        return ensemble_history
    
       
    def train_another_model(self, x_train, y_train, train_fun, validation_data, save_prefix, 
                            augment_list=[], max_augs_per_seq=2, hard_aug=True, **kwargs):

        
        self._reinitialize_model_weights()
        
        
        base_model = DynamicModel(self.base_model, 
                                  augment_list=augment_list, 
                                  max_augs_per_seq=max_augs_per_seq, 
                                  hard_aug=hard_aug)

        # train model
        history = train_fun(base_model, x_train, y_train, validation_data, **kwargs)

        # save weights
        save_path = save_prefix+'.weights.h5'
        base_model.save_weights(save_path)
        
        self.weight_paths.append(save_path)
        self.num_ensemble += 1
        self.ensemble.append(base_model)
        return history, save_path

    
    def _reinitialize_model_weights(self):
        # reinitialize weights and biases for each layer
        for layer in get_model_layers(self.base_model):
            if hasattr(layer, 'kernel_initializer'):
                if hasattr(layer, 'kernel'):
                    kernel_initializer = layer.kernel_initializer
                    layer.kernel.assign(kernel_initializer(shape=layer.kernel.shape))
                if hasattr(layer, 'bias') and layer.bias is not None:
                    bias_initializer = layer.bias_initializer
                    layer.bias.assign(bias_initializer(shape=layer.bias.shape))


def get_model_layers(model):
    if hasattr(model, 'layers'):
        return model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers

    
    

#-----------------------------------------------------------------------------
# Dynamic model for training model with dynamic augs based on ensemble targets
#-----------------------------------------------------------------------------


class DynamicModel(keras.Model):
    def __init__(self, model, ensembler=None, augment_list=[], max_augs_per_seq=2, 
                 hard_aug=False, finetune=False, inference_aug=False, uncertainty=False, **kwargs):
        super(DynamicModel, self).__init__()
        self.ensembler = ensembler
        self.model = model
        self.augment_list = augment_list
        self.max_augs_per_seq = tf.math.minimum(max_augs_per_seq, len(augment_list))
        self.hard_aug = hard_aug
        self.inference_aug = inference_aug
        self.max_num_aug = len(augment_list)
        self.ensembler = ensembler
        self.finetune = finetune
        self.uncertainty = uncertainty
        self.kwargs = kwargs
        
    def call(self, inputs, training=False):
        y_hat = self.model(inputs, training=training)
        return y_hat
    
    @tf.function
    def ensemble_predict(self, x):
        
        preds = []
        if isinstance(ensembler, EnsemblerMixed):
            for model_idx in range(self.ensembler.num_ensemble):
                preds.append(self.ensembler.ensemble[model_idx](x))
        else:
            for model_idx in range(self.ensembler.num_ensemble):
                self.ensembler.base_model.load_weights(self.ensembler.weight_paths[model_idx])
                preds.append(self.ensembler.base_model(x))

        # calculate ensemble distribution
        preds = tf.stack(preds)
        ensemble_mean = tf.reduce_mean(preds, axis=0)
        ensemble_uncertainty = self.ensembler.uncertainty_fun(preds, axis=0)
        
        return ensemble_mean, ensemble_uncertainty
                
    @tf.function  
    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
            
        x = tf.cast(x, tf.float32)
        if not self.finetune: # if finetune, no augmentations
            x = self._apply_augment(x)

        if self.ensembler is not None:
            y_mean, y_uncertainty = self.ensemble_predict(x)
            if self.uncertainty:
                y = tf.concat([y_mean, y_uncertainty], axis=1)  
            else:
                y = y_mean
        else:
            y = tf.cast(y, tf.float32)
    
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, batch):
        x, y = batch
        if self.inference_aug:
            x = self._apply_augment(x)
        else:
            x = tf.cast(x, tf.float32)
            
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
        x = batch
        if self.inference_aug:
            x = self._apply_augment(x)
        return self(x)

    @tf.function
    def _apply_augment(self, x):
        """Apply augmentations to each sequence in batch, x."""
        # number of augmentations per sequence
        if self.hard_aug:
            batch_num_aug = tf.constant(self.max_augs_per_seq, dtype=tf.int32)
        else:
            batch_num_aug = tf.random.uniform(shape=[], minval=1, maxval=self.max_augs_per_seq+1, dtype=tf.int32)

        # randomly choose which subset of augmentations from augment_list
        aug_indices = tf.sort(tf.random.shuffle(tf.range(self.max_num_aug))[:batch_num_aug])

        # apply augmentation combination to sequences
        ind = 0
        for augment in self.augment_list:
            augment_condition = tf.reduce_any(tf.equal(tf.constant(ind), aug_indices))
            x = tf.cond(augment_condition, lambda: augment(x), lambda: x)
            ind += 1
        return x
    
    def finetune_mode(self, status=True, optimizer=None, lr=None):
        """Turn on finetune flag -- no augmentations during training."""

        self.finetune = status
        if optimizer is not None:
            self.optimizer = optimizer
        if lr is not None:
            self.optimizer.learning_rate = lr
    
    def save_weights(self, filepath):
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)

