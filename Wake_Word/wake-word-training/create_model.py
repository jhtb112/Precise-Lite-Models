from precise_trainer.train import PreciseTrainer
#from precise_trainer.model import ModelParams

model_name = "tars"
folder = f"/Users/james/Documents/Projects/Voice_Assistant/wake-word-training/ww_datasets/{model_name}"  # dataset here 
# print(folder)
model_path = f"/Users/james/Documents/Projects/Voice_Assistant/wake-word-training/trained_models/{model_name}.keras"  # save here
log_dir = f"logs/fit/{model_name}"  # for tensorboard

# train a model
trainer = PreciseTrainer(model_path, folder, epochs=2000, log_dir=log_dir)
model_file = trainer.train()
# Data: <TrainData wake_words=155 not_wake_words=89356 test_wake_words=39 test_not_wake_words=22339>
# Loading wake-word...
# Loading not-wake-word...
# Loading wake-word...
# Loading not-wake-word...
# Inputs shape: (81602, 29, 13)
# Outputs shape: (81602, 1)
# Test inputs shape: (20486, 29, 13)
# Test outputs shape: (20486, 1)
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  net (GRU)                   (None, 20)                2100      
#                                                                  
#  dense (Dense)               (None, 1)                 21        
#                                                                  
# =================================================================
# Total params: 2,121
# Trainable params: 2,121
# Non-trainable params: 0
# .....
# _________________________________________________________________
# Epoch 1280/1379
# 157/160 [============================>.] - ETA: 0s - loss: 0.0308 - accuracy: 0.9868
# ....
# Wrote to /home/miro/PycharmProjects/ovos-audio-classifiers/trained/hey_computer/model.tflite
trainer.test()

# === Counts ===
# False Positives: 2
# True Negatives: 20445
# False Negatives: 2
# True Positives: 37
# 
# === Summary ===
# 20482 out of 20486
# 99.98%
# 
# 0.01% false positives
# 5.13% false negatives
