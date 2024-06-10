import tensorflow as tf
from tensorflow import keras

def DeepSTARR(input_shape):

  inputs = keras.layers.Input(shape=input_shape)
  x = keras.layers.Conv1D(256, kernel_size=7, padding='same')(inputs)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.MaxPooling1D(2)(x)

  x = keras.layers.Conv1D(60, kernel_size=3, padding='same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.MaxPooling1D(2)(x)

  x = keras.layers.Conv1D(60, kernel_size=5, padding='same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.MaxPooling1D(2)(x)

  x = keras.layers.Conv1D(120, kernel_size=3, padding='same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.MaxPooling1D(2)(x)

  x = keras.layers.Flatten()(x) 

  x = keras.layers.Dense(256)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.Dropout(0.4)(x)

  x = keras.layers.Dense(256)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.Dropout(0.4)(x)

  outputs = keras.layers.Dense(2, activation='linear')(x)

  return tf.keras.Model(inputs=inputs, outputs=outputs)


def DeepSTARR_unc(input_shape):

  inputs = keras.layers.Input(shape=input_shape)
  x = keras.layers.Conv1D(256, kernel_size=7, padding='same')(inputs)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.MaxPooling1D(2)(x)

  x = keras.layers.Conv1D(60, kernel_size=3, padding='same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.MaxPooling1D(2)(x)

  x = keras.layers.Conv1D(60, kernel_size=5, padding='same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.MaxPooling1D(2)(x)

  x = keras.layers.Conv1D(120, kernel_size=3, padding='same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.MaxPooling1D(2)(x)

  x = keras.layers.Flatten()(x) 

  x = keras.layers.Dense(256)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.Dropout(0.4)(x)

  x = keras.layers.Dense(256)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.Dropout(0.4)(x)

  outputs = keras.layers.Dense(4, activation='linear')(x)

  return tf.keras.Model(inputs=inputs, outputs=outputs)



def residualbindMPRA(input_shape):
    '''
    CNN for predicting lentiMPRA data
    if aleatoric=True, predict aleatoric uncertainty
    if epistemic=True, predict epistemic uncertainty 
    '''

    def residual_block(input_layer, filter_size, activation='relu', dilated=5):
        '''
        define residual block for CNN
        '''
        factor = []
        base = 2
        for i in range(dilated):
            factor.append(base**i)
        num_filters = input_layer.shape.as_list()[-1]

        nn = keras.Conv1D(filters=num_filters,
                                        kernel_size=filter_size,
                                        activation=None,
                                        use_bias=False,
                                        padding='same',
                                        dilation_rate=1, 
                                        )(input_layer)
        nn = keras.BatchNormalization()(nn)
        for f in factor:
            nn = keras.Activation('relu')(nn)
            nn = keras.Dropout(0.1)(nn)
            nn = keras.Conv1D(filters=num_filters,
                                            kernel_size=filter_size,
                                            activation=None,
                                            use_bias=False,
                                            padding='same',
                                            dilation_rate=f,
                                            )(nn)
            nn = keras.BatchNormalization()(nn)
        nn = keras.add([input_layer, nn])
        return keras.Activation(activation)(nn)

    inputs = keras.Input(shape=input_shape)
    x = keras.Conv1D(196, kernel_size=19, padding='same')(inputs)
    x = keras.BatchNormalization()(x)
    x = keras.Activation('silu')(x)
    x = keras.Dropout(0.2)(x)
    x = residual_block(x, 3, activation='silu', dilated=5)
    x = keras.Dropout(0.2)(x)
    x = keras.MaxPooling1D(5)(x) 

    x = keras.Conv1D(256, kernel_size=7, padding='same')(x)
    x = keras.BatchNormalization()(x)
    x = keras.Activation('silu')(x)
    x = keras.Dropout(0.2)(x)
    x = keras.MaxPooling1D(5)(x) 

    x = keras.Dense(256)(x)
    x = keras.BatchNormalization()(x)
    x = keras.Activation('silu')(x)
    x = keras.Dropout(0.5)(x)

    x = keras.GlobalAveragePooling1D()(x)
    x = keras.Flatten()(x)

    x = keras.Dense(256)(x)
    x = keras.BatchNormalization()(x)
    x = keras.Activation('silu')(x)
    x = keras.Dropout(0.5)(x)

    outputs = keras.Dense(2, activation='linear')(x)

    return keras.Model(inputs=inputs, outputs=outputs)




def residualbindMPRA_unc(input_shape):
    '''
    CNN for predicting lentiMPRA data
    if aleatoric=True, predict aleatoric uncertainty
    if epistemic=True, predict epistemic uncertainty 
    '''

    def residual_block(input_layer, filter_size, activation='relu', dilated=5):
        '''
        define residual block for CNN
        '''
        factor = []
        base = 2
        for i in range(dilated):
            factor.append(base**i)
        num_filters = input_layer.shape.as_list()[-1]

        nn = keras.Conv1D(filters=num_filters,
                                        kernel_size=filter_size,
                                        activation=None,
                                        use_bias=False,
                                        padding='same',
                                        dilation_rate=1, 
                                        )(input_layer)
        nn = keras.BatchNormalization()(nn)
        for f in factor:
            nn = keras.Activation('relu')(nn)
            nn = keras.Dropout(0.1)(nn)
            nn = keras.Conv1D(filters=num_filters,
                                            kernel_size=filter_size,
                                            activation=None,
                                            use_bias=False,
                                            padding='same',
                                            dilation_rate=f,
                                            )(nn)
            nn = keras.BatchNormalization()(nn)
        nn = keras.add([input_layer, nn])
        return keras.Activation(activation)(nn)

    inputs = keras.Input(shape=input_shape)
    x = keras.Conv1D(196, kernel_size=19, padding='same')(inputs)
    x = keras.BatchNormalization()(x)
    x = keras.Activation('silu')(x)
    x = keras.Dropout(0.2)(x)
    x = residual_block(x, 3, activation='silu', dilated=5)
    x = keras.Dropout(0.2)(x)
    x = keras.MaxPooling1D(5)(x) 

    x = keras.Conv1D(256, kernel_size=7, padding='same')(x)
    x = keras.BatchNormalization()(x)
    x = keras.Activation('silu')(x)
    x = keras.Dropout(0.2)(x)
    x = keras.MaxPooling1D(5)(x) 

    x = keras.Dense(256)(x)
    x = keras.BatchNormalization()(x)
    x = keras.Activation('silu')(x)
    x = keras.Dropout(0.5)(x)

    x = keras.GlobalAveragePooling1D()(x)
    x = keras.Flatten()(x)

    x = keras.Dense(256)(x)
    x = keras.BatchNormalization()(x)
    x = keras.Activation('silu')(x)
    x = keras.Dropout(0.5)(x)

    outputs = keras.Dense(4, activation='linear')(x)

    return keras.Model(inputs=inputs, outputs=outputs)

