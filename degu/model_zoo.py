from tensorflow import keras 

def DeepSTARR(input_shape=(249,4), output_shape=2):
    "DeepSTARR model from deAlmeida et al. Nat Genetics (2022)"
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

    outputs = keras.layers.Dense(output_shape, activation='linear')(x)

    return keras.Model(inputs=inputs, outputs=outputs)



def DeepSTARR_student(input_shape=(249,4), output_shape=4):
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

    outputs = keras.layers.Dense(output_shape, activation='linear')(x)

    return keras.Model(inputs=inputs, outputs=outputs)

