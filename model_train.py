from config import setup_configs as configs
from sidekick.prepro.process import Process
from sidekick.prepro.imgtoarrayprepro import ImgtoArrPrePro
from sidekick.prepro.meanprocess import MeanProcess
from sidekick.nn.conv.googlenet import GoogLeNet
from sidekick.io.hdf5datagen import Hdf5DataGen
from sidekick.callbs.manualcheckpoint import ManualCheckpoint
from sidekick.callbs.trainmonitor import TrainMonitor
import json
import argparse
import matplotlib
matplotlib.use('Agg')
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

# Command line arguments
ap= argparse.ArgumentParser()
ap.add_argument('-o','--output', type=str, required=True ,help="Path to output directory")
ap.add_argument('-m', '--model', help='Path to checkpointed model')
ap.add_argument('-e','--epoch', type=int, default=0, help="Starting epoch of training")
args= vars(ap.parse_args())

# Building and processing dataset
print('[NOTE]:- Building Dataset...\n')
pro= Process(64, 64)
i2a= ImgtoArrPrePro()
# Loading means from stored JSON file
data_means= json.loads(open(configs.IMG_MEAN).read())
# Using means to initialize the mean preprocessor to normalize dataset
meanpro= MeanProcess(data_means['R'], data_means['G'], data_means['B'])
# Using image augmentation to get better results
aug= ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
# Building the HDF5 generators
train_gen= Hdf5DataGen(configs.HDF5_TRAIN, 64, 200, aug=aug, preprocessors=[pro, meanpro, i2a])
val_gen= Hdf5DataGen(configs.HDF5_VAL, 64, 200, preprocessors=[pro, meanpro, i2a])

# Compiling or loading a checkpointed model
if args['model'] is None:
    print("[NOTE]:- Building model from scratch...")
    model= GoogLeNet.build(64, 64, 3, 200, reg=0.0003)
    opt= Adam(learning_rate= 0.001)
    model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=opt)
else:
    print("[NOTE]:- Building model {}\n".format(args['model']))
    model= load_model(args['model'])
    # Update learning rate to desired value
    # a variable used to restart training with the same lr as of the model if needed
    first_time_flag=0
    if first_time_flag==1:
        print("[NOTE]:- No lr change is requested {} Change first_time_flag.".format(K.get_value(model.optimizer.lr)))
    else:
        new_lr= 1e-6
        old_lr= K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, new_lr)
        print('[NOTE]:- Changing lr from {} to {}\n'.format(old_lr, new_lr))

# Setting up callbacks
callbacks= [ManualCheckpoint(args['output'], save_at=1, start_from=args['epoch']),
            TrainMonitor(figPath=configs.FIG_PATH, jsonPath=configs.JSON_PATH, startAt=args['epoch'])]

# Training model
print("[NOTE]:- Training model...\n")
model.fit_generator(train_gen.generator(),
                    steps_per_epoch=train_gen.data_length//64,
                    validation_data= val_gen.generator(),
                    validation_steps= val_gen.data_length//64,
                    epochs=65,
                    max_queue_size=10,
                    callbacks=callbacks,
                    initial_epoch=args['epoch'])

train_gen.close()
val_gen.close()

