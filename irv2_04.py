#InceptionResNetV2_04 model:
import tensorflow as tf
from tensorflow.keras import models,layers
from tensorflow.keras.applications import InceptionResNetV2
import numpy as np
'''
cnn = models.sequential([
    layers.Flatten(input_shape=(840,640)),
    layers.Dense(1024,activation='relu'),
    
])
'''
inputLyr = layers.Input(shape=[270,480,3])
irv2Lyr = InceptionResNetV2(include_top=False)(inputLyr)
gavgpool2dLyr = layers.GlobalAveragePooling2D(input_shape=(7,13,1536))(irv2Lyr)
denseLyr = layers.Dense(1536,activation='relu')(gavgpool2dLyr)
splitterdenseLyr = tf.split(denseLyr,num_or_size_splits=2,axis=-1)

dense1Lyr = layers.Dense(1024,activation='relu')(splitterdenseLyr[0])
dense4Lyr = layers.Dense(1024,activation='relu')(splitterdenseLyr[1])

dense2Lyr = layers.Dense(256,activation='relu')(dense1Lyr)
dense5Lyr = layers.Dense(256,activation='relu')(dense4Lyr)

dense3Lyr = layers.Dense(64,activation='relu')(dense2Lyr)
dense6Lyr = layers.Dense(64,activation='relu')(dense5Lyr)

outputbrakingLyr = layers.Dense(16,activation='linear')(dense3Lyr)
outputsteeringLyr = layers.Dense(16,activation='linear')(dense6Lyr)