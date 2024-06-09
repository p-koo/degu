import os, sys
import numpy as np
import gc
from six.moves import cPickle
import heteroscedastic
import utils
from tensorflow import keras
from model_zoo import DeepSTARR

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
#x_train, y_train = utils.downsample_trainset(x_train, y_train, downsample=None, seed=12345)
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


#--------------------------------------------------------------------------------
# train model

for trial in range(num_trials):
    keras.backend.clear_session()
    gc.collect()

    savename = os.path.join(results_path, save_prefix+'_'+str(trial))
    print(savename)

    model = DeepSTARR(input_shape=(L,A))
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

