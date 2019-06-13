from imageai.Prediction.Custom import CustomImagePrediction
import os
import cv2

class Prediction:
    def __init__(self):
        self.modelPath = ""
        self.__modelType = ""
        self.__modelloaded = False
        self.__prediction_collection = []
        self.__jsonPath = ""
        
    def setModelPath(self,model_path):
        self.modelPath = model_path
    def setJsonPath(self,json_path):
        self.__jsonPath = json_path
        
    def setModelTypeAsResNet(self):
        self.__modelType = "resnet"
    def setModelTypeAsSqueezeNet(self):
        self.__modelType = "squeezenet"
    def setModelTypeAsDenseNet(self):
        self.__modelType = "densenet"
    def setModelTypeAsInceptionV3(self):
        self.__modelType = "inceptionv3"
    def setModelTypeAsVgg(self):
        self.__modelType = "vgg"
        
    def loadPrediction(self, prediction_speed = 'normal',num_objects=10):
        if self.__modelloaded == False:
            if self.__modelType == "":
                raise ValueError("You must set a valid model type before loading the model.")
                if self.__jsonPath == "":
                    raise ValueError("You must set a valid json path before loading the model.")
            elif self.__modelType == "resnet":
                prediction = CustomImagePrediction()
                prediction.setModelTypeAsResNet()
                
            elif self.__modelType == "squeezenet":
                prediction = CustomImagePrediction()
                prediction.setModelTypeAsSqueezeNet()
                
            elif self.__modelType == "densenet":
                prediction = CustomImagePrediction()
                prediction.setModelTypeAsDenseNet()
                
            elif self.__modelType == "inceptionv3":
                prediction = CustomImagePrediction()
                prediction.setModelTypeAsInceptionV3()

            elif self.__modelType == "vgg":
                prediction = CustomImagePrediction()
                prediction.setModelTypeAsVgg()
                
            prediction.setModelPath(self.modelPath)
            prediction.setJsonPath(self.__jsonPath)
            prediction.loadModel(prediction_speed, num_objects)
            self.__prediction_collection.append(prediction)
            self.__modelloaded = True
        else:
            raise ValueError("You must set a valid model type before loading the model.")
                
    def predict(self, image_input, result_count=5, input_type="file"):
        if self.__modelloaded == True:
            predictions, probabilities = self.__prediction_collection[0].predictImage(image_input, result_count, input_type)
            return predictions, probabilities
        else:
            raise ValueError("You must load a model.")
            
    

if __name__ == '__main__':
    execution_path = os.getcwd()
    
    prediction = Prediction()
    prediction.setJsonPath(os.path.join(execution_path, "gestures", "json", "model_class.json"))
    prediction.setModelPath(os.path.join(execution_path, "gestures", "models1", "model_ex-006_acc-0.998940.h5"))
    prediction.setModelTypeAsResNet()
    prediction.loadPrediction()

    #img = cv2.imread('test/dd.jpg')
    #img = img[...,::-1]
    predictions, probabilities = prediction.predict(img, result_count = 10, input_type='array')
    predictions, probabilities = prediction.predict(os.path.join(execution_path, "test/dd.jpg"), result_count = 10)
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction + " : " + eachProbability)
        
        
        
        
