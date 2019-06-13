from imageai.Prediction.Custom import ModelTraining
import os

execution_path = os.getcwd()

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.usePretrainedModel(os.path.join(execution_path, "gestures", "models", "model_ex-006_acc-0.998940.h5"))
model_trainer.setDataDirectory("gestures")
model_trainer.trainModel(num_objects=10, num_experiments=10, enhance_data=True, batch_size=8, show_network_summary=True, training_image_size=224)
print("Complete!")
