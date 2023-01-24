#InceptionResNetV2_04 model:
import tensorflow as tf
from tensorflow.keras import models,layers
from tensorflow.keras.applications import InceptionResNetV2,inception_resnet_v2
import numpy as np

#Input layer takes the image:
inputLyr = layers.Input(shape=[270,480,3])
#Convert input to the expected format for IRV2:
processed_irv2_input = inception_resnet_v2.preprocess_input(inputLyr)
irv2Lyr = InceptionResNetV2(include_top=False)(processed_irv2_input)
gavgpool2dLyr = layers.GlobalAveragePooling2D(input_shape=(7,13,1536))(irv2Lyr)
denseLyr = layers.Dense(1024,activation='relu')(gavgpool2dLyr)
#The output of the above layer is split between the two forward layers:
splitterdenseLyr = tf.split(denseLyr,num_or_size_splits=2,axis=-1)

#L1 Dense layers
dense1Lyr = layers.Dense(256,activation='relu')(splitterdenseLyr[0])
dense4Lyr = layers.Dense(256,activation='relu')(splitterdenseLyr[1])

#L2 Dense layers
dense2Lyr = layers.Dense(64,activation='relu')(dense1Lyr)
dense5Lyr = layers.Dense(64,activation='relu')(dense4Lyr)

#L3 Dense layers
dense3Lyr = layers.Dense(16,activation='relu')(dense2Lyr)
dense6Lyr = layers.Dense(16,activation='relu')(dense5Lyr)

#Two output layers: braking and steering
outputbrakingLyr = layers.Dense(1,activation='linear')(dense3Lyr)
outputsteeringLyr = layers.Dense(1,activation='linear')(dense6Lyr)

model = models.Model(inputLyr,[outputbrakingLyr,outputsteeringLyr])