import tensorflow as tf
import keras
import numpy as np

def create_model(x_train, x_test, classes, trainable_encoder=False):

    model = keras.models.Sequential([
        keras.layers.InputLayer((128, 64)),

        keras.layers.Conv1D(16, 3, padding = "same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPool1D(pool_size = 2, strides = 2),

        keras.layers.Conv1D(32, 3, padding = "same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPool1D(pool_size = 2, strides = 2),

        keras.layers.Conv1D(64, 3, padding = "same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Flatten(),

        keras.layers.Dense(5, activation='softmax')

    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=[keras.metrics.sparse_categorical_accuracy])

    model.summary()

    return model

# model_1

# def residual_block(x, filters, conv_num = 3, activation = "relu"):
#     s = keras.layers.Conv1D(filters, 1, padding = "same")(x)
    
#     for i in range(conv_num - 1):
#         x = keras.layers.Conv1D(filters, 3, padding = "same")(x)
#         x = keras.layers.Activation(activation)(x)
    
#     x = keras.layers.Conv1D(filters, 3, padding = "same")(x)
#     x = keras.layers.Add()([x, s])
#     x = keras.layers.Activation(activation)(x)
    
#     return keras.layers.MaxPool1D(pool_size = 2, strides = 2)(x)


# def create_model(x_train, x_test, classes, trainable_encoder=False):
#     # input shape (128, 2584, 1)

#     input_shape = (128,8)
#     num_classes = 5

#     inputs = keras.layers.Input(shape = input_shape, name = "input")
    
#     x = residual_block(inputs, 16, 2)
#     x = residual_block(inputs, 32, 2)
#     x = residual_block(inputs, 64, 3)
#     x = residual_block(inputs, 128, 3)
#     x = residual_block(inputs, 128, 3)
#     x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
#     x = keras.layers.Flatten()(x)
#     x = keras.layers.Dense(256, activation="relu")(x)
#     x = keras.layers.Dense(128, activation="relu")(x)
    
#     outputs = keras.layers.Dense(num_classes, activation = "softmax", name = "output")(x)
    
#     model = keras.models.Model(inputs = inputs, outputs = outputs)

#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#                   loss=keras.losses.sparse_categorical_crossentropy,
#                   metrics=[keras.metrics.sparse_categorical_accuracy])

#     model.summary()

#     return model