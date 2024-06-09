import tensorflow as tf
from tensorflow import keras
from evoaug_tf import augment, evoaug

class RandomShuffle(augment.AugmentBase):
    def __init__(self, axis=1):
        self.axis = axis
    
    @tf.function
    def __call__(self, x):        
        x_new = tf.random.shuffle(x, axis=self.axis)
        return x_new



class AugModel(keras.Model):
    
    def __init__(self, model_func, model_ensemble, input_shape=None, augment_list=[], max_augs_per_seq=2, 
                 hard_aug=False, finetune=False, inference_aug=False, concat=False, **kwargs):
        super(AugModel, self).__init__()
        self.model = model_func
        self.augment_list = augment_list
        self.max_augs_per_seq = tf.math.minimum(max_augs_per_seq, len(augment_list))
        self.hard_aug = hard_aug
        self.inference_aug = inference_aug
        self.max_num_aug = len(augment_list)
        self.insert_max = evoaug.augment_max_len(augment_list)
        self.finetune = finetune
        self.kwargs = kwargs
        self.model_ensemble = model_ensemble
        
        if input_shape is not None:
            self.build_model(input_shape)

    def build_model(self, input_shape):
        # Add batch dimension to input shape2
        augmented_input_shape = [None] + list(input_shape)
        # Extend sequence lengths based on augment_list
        augmented_input_shape[1] += evoaug.augment_max_len(self.augment_list)

        self.model = self.model(augmented_input_shape[1:], **self.kwargs)

    @tf.function
    def call(self, inputs, training=False):
        y_hat = self.model(inputs, training=training)
        return y_hat
    
    @tf.function
    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        if self.finetune: # if finetune, no augmentations
            if self.insert_max:  # if insert_max is larger than 0, then pad each sequence with random DNA
                x = self._pad_end(x)
        else:
            x_mut = self._apply_augment(x)
            y_mut = []
            for model in self.model_ensemble:
                y_mut.append(model(x_mut, training=False))
            y_mut = tf.math.reduce_mean(tf.stack(y_mut, axis=0), axis=0)

        if concat:
            x = tf.concat([x, x_mut], axis=0)
            y = tf.concat([y, y_mut], axis=0)
        else:
            y = y_mut
            x = x_mut

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
            if self.insert_max:
                x = self._pad_end(x)

        y_pred = self(x, training=False)  
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def predict_step(self, batch):
        x = batch
        if self.inference_aug:
            x = self._apply_augment(x)
        else:
            if self.insert_max:
                x = self._pad_end(x)
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
        insert_status = True
        ind = 0
        for augment in self.augment_list:
            augment_condition = tf.reduce_any(tf.equal(tf.constant(ind), aug_indices))
            x = tf.cond(augment_condition, lambda: augment(x), lambda: x)
            if augment_condition and hasattr(augment, 'insert_max'):
                insert_status = False
            ind += 1
        if insert_status:
            if self.insert_max:
                x = self._pad_end(x)
        return x
    

    def _pad_end(self, x):
        """Add random DNA padding of length insert_max to the end of each sequence in batch."""

        N = tf.shape(x)[0]
        L = tf.shape(x)[1]
        A = tf.cast(tf.shape(x)[2], dtype = tf.float32)
        p = tf.ones((A,)) / A
        padding = tf.transpose(tf.gather(tf.eye(A), tf.random.categorical(tf.math.log([p] * self.insert_max), N)), perm=[1,0,2])

        half = int(self.insert_max/2)
        x_padded = tf.concat([padding[:,:half,:], x, padding[:,half:,:]], axis=1)
        return x_padded


    def finetune_mode(self, optimizer=None, lr=None):
        """Turn on finetune flag -- no augmentations during training."""

        self.finetune = True
        if optimizer is not None:
            self.optimizer = optimizer
        if lr is not None:
            self.optimizer.learning_rate = lr


    def save_weights(self, filepath):
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)



