from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D, AvgPool2D, GlobalMaxPool2D, GlobalAvgPool2D, BatchNormalization, add, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

image_size = 224
input_object = Input(shape = (image_size,image_size,3))

output = Conv2D(256, kernel_size=1, strides=1, padding="same")(input_object)
output = Activation("relu")(output)

output = Conv2D(256, kernel_size=1, strides=1, padding="same")(output)
output = Activation("relu")(output)

output = Conv2D(256, kernel_size=1, strides=1, padding="same")(output)
output = Activation("relu")(output)

output = Conv2D(256, kernel_size=1, strides=1, padding="same")(output)
output = Activation("relu")(output)

output = GlobalAvgPool2D()(output)

#output = Flatten()(output)

#output = Dense(1000)(output)
#output = Activation("relu")(output)
#output = MaxPool2D(pool_size=(3,3), strides=(2,2))(output)

model = Model(inputs=input_object, outputs=output)

optimizer = Adam(lr=1e-3, decay=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.summary()
