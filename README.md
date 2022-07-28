## Problem Definition
The objective is to predict the location and type of vehicle found in the scene using the images provided. The images have annotations in xml format, and the task is to create an Object Detector and Classifier.Eg: Cars, Bus, Motorbikes etc. Select 3 or 4 classes from the provided dataset and build the training module. Each Image may or may not have the classes to identify. You can find description along with the dataset

## Objectives:
- Data Analysis
  - Visualize the object data distribution as graph or in any meaningful representation. 

- Deep Learning:
  - Create an algorithm to identify the objects and classify.

## Data:
> Dataset Link: https://www.kaggle.com/datasets/pratikbarua/vehicle-detection-dataset

## Procedure:
1. The data available from kaggle is have annotations in yaml format and have multiple classes. So I have filtered specifica classes : "background", "car", "motorbike", "bus" and some annotations were not named properly so those data have to be cleaned.

2. The data is in RGB format but the assignment is to detect on grayscale images. Therefore the images are converted to grayscale for training. 

3. The dataframe is created and checked the data distribution to confirm almost similar distribution.

4. Data is split into Train, Test and Validation datasets

5. For better evaluation, I've converted the data into coco json format so as to use the coco evaluation functionality.

6. After creation of coco json format files, verify the annotations by loading the json data in data-visualization.ipynb notebook 

7. The training notebook: ssd_train.ipynb is provided for training the SSD based model. I've used multiple references for this model implementation. 

8. Once trained, the predict.ipynb notebook can be used to generate the results.

## To-Do:
The training loss is not reducing, and the evaluation scores are not improved with epochs. Have to check the pipeline. The prediction is not proper as of now.
