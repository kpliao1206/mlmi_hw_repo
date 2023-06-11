import os
from model import *
from data import *
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tic = time.time()
BatchSize = 1

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(BatchSize,'Membrane','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit_generator(myGene,steps_per_epoch=2000,epochs=10,callbacks=[model_checkpoint])
model.fit(myGene,steps_per_epoch=2000,epochs=10,callbacks=[model_checkpoint])
toc = time.time()
print("Time=" + str((toc-tic)) + "sec")

#testGene = testGenerator("data/membrane/test")
#results = model.predict_generator(testGene,30,verbose=1)
#results = model.predict(testGene,30,verbose=1)
#saveResult("data/membrane/test",results)