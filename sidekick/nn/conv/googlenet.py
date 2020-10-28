from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, AveragePooling2D, Dropout, Dense, Flatten
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow import nn as tfn

class GoogLeNet:
    @staticmethod
    def conv_module(x, k, k_size, stride, chanDim,
                    padding='same', reg_value=0.0005, name=None):
        if name is not None:
            conv_name= name+'_conv'
            bn_name= name+'_bn'
            acti_name= name+'_acti'
        else:
            conv_name= None
            bn_name= None
            acti_name= None

        x= Conv2D(filters=k, kernel_size=k_size, strides=stride, padding=padding,
                  kernel_regularizer=l2(reg_value), name=conv_name)(x)
        x= Activation(tfn.relu, name=acti_name)(x)
        x= BatchNormalization(axis=chanDim, name=bn_name)(x)

        return x

    @staticmethod
    def inception_module(x, k1, k1_3, k3, k1_5, k5, k1_pool, chanDim,
                         level, reg_value= 0.0005):
        branch_1= GoogLeNet.conv_module(x, k=k1, k_size=1, stride=(1,1),
                                        chanDim=chanDim, reg_value=reg_value,
                                        name= level+'_branch1')

        branch_2= GoogLeNet.conv_module(x, k=k1_3, k_size=1, stride=(1,1),
                                        chanDim=chanDim, reg_value=reg_value,
                                        name=level+'_branch2_1')
        branch_2= GoogLeNet.conv_module(branch_2, k=k3, k_size=3, stride=(1,1),
                                        chanDim=chanDim, reg_value=reg_value,
                                        name=level+'_branch2_2')

        branch_3= GoogLeNet.conv_module(x, k=k1_5, k_size=1, stride=(1,1),
                                        chanDim=chanDim, reg_value=reg_value,
                                        name=level+'_branch3_1')
        branch_3= GoogLeNet.conv_module(branch_3, k=k5, k_size=5, stride=(1,1),
                                        chanDim=chanDim, reg_value=reg_value,
                                        name=level+'_branch3_2')

        branch_4= MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same',
                            name=level+'_branch4_1')(x)
        branch_4= GoogLeNet.conv_module(branch_4, k=k1_pool, k_size=1, stride=(1,1),
                                        chanDim=chanDim, reg_value=reg_value,
                                        name=level+'_branch4_2')

        x= concatenate([branch_1, branch_2, branch_3, branch_4], axis=chanDim,
                       name= level+'_final')

        return x

    @staticmethod
    def build(width, height, depth, classes, reg= 0.0005):
        inputShape= (height, width, depth)
        chanDim=-1

        if K.image_data_format()=='channels_first':
            inputShape= (depth, height, width)
            chanDim=1

        inputs= Input(shape=inputShape)
        x= GoogLeNet.conv_module(x=inputs, k=64, k_size=3, stride=(1,1),
                                 chanDim=chanDim, name='Layer1')
        x= MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same', name="Layer2_pool")(x)
        x= Dropout(0.25, name="Layer3_drop")(x)

        x= GoogLeNet.conv_module(x=x, k=32, k_size=1, stride=(1,1),
                                chanDim=chanDim, name="Layer4")
        x= GoogLeNet.conv_module(x=x, k=128, k_size=5, stride=(1,1),
                                 chanDim=chanDim, name="Layer5")
        x= MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same', name="Layer5_pool")(x)
        x= Dropout(0.25, name="Layer6_drop")(x)

        x= GoogLeNet.inception_module(x=x, k1=32, k1_3=64, k3=128, k1_5=16, k5=32, k1_pool=16,
                                      chanDim=chanDim, level="Layer7")
        x = GoogLeNet.inception_module(x=x, k1=64, k1_3=128, k3=256, k1_5=32, k5=64, k1_pool=32,
                                       chanDim=chanDim, level="Layer8")
        x = GoogLeNet.inception_module(x=x, k1=32, k1_3=64, k3=128, k1_5=16, k5=32, k1_pool=16,
                                       chanDim=chanDim, level="Layer9")
        x= MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same', name="Layer10_pool")(x)
        x = Dropout(0.25, name="Layer11_drop")(x)

        x= GoogLeNet.inception_module(x=x, k1=64, k1_3=128, k3=256, k1_5=32, k5=64, k1_pool=32,
                                      chanDim=chanDim, level="Layer11")
        x = GoogLeNet.inception_module(x=x, k1=128, k1_3=256, k3=512, k1_5=64, k5=128, k1_pool=64,
                                       chanDim=chanDim, level="Layer12")
        x = GoogLeNet.inception_module(x=x, k1=256, k1_3=512, k3=1024, k1_5=128, k5=256, k1_pool=128,
                                       chanDim=chanDim, level="Layer13")
        x = GoogLeNet.inception_module(x=x, k1=128, k1_3=256, k3=512, k1_5=64, k5=128, k1_pool=64,
                                       chanDim=chanDim, level="Layer14")
        x = GoogLeNet.inception_module(x=x, k1=64, k1_3=128, k3=256, k1_5=32, k5=64, k1_pool=32,
                                       chanDim=chanDim, level="Layer15")
        x= MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same', name="Layer16_pool")(x)
        x = Dropout(0.25, name="Layer17_drop")(x)

        x= AveragePooling2D(pool_size=(4,4), name="Layer18_avgpool")(x)
        x= Dropout(0.38, name="Layer19_drop")(x)

        x= Flatten(name="Layer20_flat")(x)
        x= Dense(classes, kernel_regularizer=l2(reg), name="Layer21_output")(x)
        x= Activation(tfn.softmax, name="Layer22_soft")(x)

        model= Model(inputs, x, name='GoogLeNet')

        return model

        