import os, sys
import numpy as np
import gc
from six.moves import cPickle
import heteroscedastic
import evoaug_tf
from evoaug_tf import evoaug, augment
import evoaug_custom
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#--------------------------------------------------------------------------------
# set up params

task = sys.argv[1]
loss_name = sys.argv[2]

if loss_name == 'gaussian':
    loss = heteroscedastic.gaussian_nll_loss
elif loss_name == 'laplace':
    loss = heteroscedastic.laplace_nll_loss
elif loss_name == 'cauchy':
    loss = heteroscedastic.cauchy_nll_loss

num_trials = 5
save_prefix = 'distilled_aug_'+task+'_'+loss_name
save_prefix = 'deepstarr_'+task+'_'+loss_name

batch_size = 100
epochs = 100
results_path = '../results/deepstarr'

es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                            patience=10,
                                            verbose=1,
                                            mode='min',
                                            restore_best_weights=True)
# reduce learning rate callback
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.2,
                                                patience=3,
                                                min_lr=1e-7,
                                                mode='min',
                                                verbose=1)
#--------------------------------------------------------------------------------
# load dataset

filepath = '../data/deepstarr_data.h5'
x_train, y_train, x_valid, y_valid, x_test, y_test = utils.load_deepstarr(filepath)
#x_train, y_train = utils.downsample_trainset(x_train, y_train, downsample, seed=12345)
N, L, A = x_train.shape

if task == 'Dev':
    y_train = np.expand_dims(y_train[:,0], axis=1)
    y_valid = np.expand_dims(y_valid[:,0], axis=1)
    y_test = np.expand_dims(y_test[:,0], axis=1)
elif task == 'HK':
    y_train = np.expand_dims(y_train[:,1], axis=1)
    y_valid = np.expand_dims(y_valid[:,1], axis=1)
    y_test = np.expand_dims(y_test[:,1], axis=1)

N, L, A = x_valid.shape


##########################################################################################
# Load ensemble of models
##########################################################################################

ensemble_models = []
for trial in range(num_trials):
    savename = os.path.join(results_path, save_prefix_old + '_' + str(trial) + '.h5')
    model = DeepSTARR(input_shape=(L,A))
    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss=loss) 
    model.load_weights(weight_path)
    ensemble_models.append(model)


##########################################################################################
# Pre-train with perturbed data
##########################################################################################

model = evoaug_custom.AugModel(DeepSTARR, ensemble_models, input_shape=(L,A), augment_list=augment_list, 
                        max_augs_per_seq=2, hard_aug=True, concat=concat)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer, loss=loss)
history_aug = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                        validation_data=(x_valid, y_valid), callbacks=[es_callback, reduce_lr])

save_path = savename+"_aug.h5"
model.save_weights(save_path)

# performance metrics
pred = model.predict(x_test, batch_size=batch_size)
mse_aug, pearsonr_aug, spearmanr_aug = utils.summary_statistics(pred,  y_test, index=0)
mse_aug2, pearsonr_aug2, spearmanr_aug2 = utils.summary_statistics(pred,  y_test, index=1)

##########################################################################################
# Fine-tune on original unperturbed data
##########################################################################################

finetune_model = evoaug.RobustModel(DeepSTARR, input_shape=(L,A), augment_list=[], 
                                    max_augs_per_seq=2, hard_aug=True)
finetune_optimizer = keras.optimizers.Adam(learning_rate=0.001)
finetune_model.compile(finetune_optimizer, loss=loss)
finetune_model.load_weights(save_path)
finetune_model.finetune_mode()
history = finetune_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                             validation_data=(x_valid, y_valid), callbacks=[es_callback, reduce_lr])

finetune_path = savename+"_finetune.h5"
finetune_model.save_weights(finetune_path)

# evaluate model
pred = finetune_model.predict(x_test, batch_size=batch_size)
mse, pearsonr, spearmanr = utils.summary_statistics(pred,  y_test, index=0)
mse2, pearsonr2, spearmanr2 = utils.summary_statistics(pred,  y_test, index=1)
with open(savename + '.pickle', 'wb') as fout:
    cPickle.dump([mse, pearsonr, spearmanr], fout)
    cPickle.dump([mse2, pearsonr2, spearmanr2], fout)
    cPickle.dump([mse_aug, pearsonr_aug, spearmanr_aug], fout)
    cPickle.dump([mse_aug2, pearsonr_aug2, spearmanr_aug2], fout)
    cPickle.dump(pred fout)


##############################################################################
# Generate labels from ensemble of models
##########################################################################################

y_new = []
pred = []
for trial in range(num_trials):

    savename = os.path.join(results_path, save_prefix_old + '_' + str(trial) + '.h5')
    model = DeepSTARR(input_shape=(L,A))
    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss=loss) 
    model.load_weights(weight_path)

    # generate predictions
    y_new.append(model.predict(x_train, batch_size=batch_size))
    pred.append(model.predict(x_test, batch_size=batch_size))
y_train = np.expand_dims(np.mean(np.array(y_new), axis=0)[:,0], axis=1)
pred = np.mean(np.array(pred), axis=0)

# run for each set and enhancer type
pred = model.predict(x_test, batch_size=batch_size)
mse, pearsonr, spearmanr = utils.summary_statistics(pred,  y_test, index=0)
with open(save_prefix_old + '_ensemble.pickle', 'wb') as fout:
    cPickle.dump([mse, pearsonr, spearmanr], fout)

##########################################################################################
# Train distilled models based on ensemble labels
##########################################################################################

for trial in range(num_trials):
    keras.backend.clear_session()
    gc.collect()

    savename = os.path.join(results_path, save_prefix+'_'+str(trial))
    print(savename)

    inputs, outputs = DeepSTARR(input_shape=(L,A))
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss=loss)
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(x_valid, y_valid),
                        callbacks=[es_callback, reduce_lr])
    model.save_weights(savename+'.h5')

    # evaluate results
    pred = model.predict(x_test, batch_size=batch_size)
    mse, pcc, scc = summary_statistics(pred, y_test, task='mean', index=0)
    with open(savename+'.pickle', 'wb') as fout:
        cPickle.dump([mse, pcc, scc], fout)
        cPickle.dump(pred, fout)
        

