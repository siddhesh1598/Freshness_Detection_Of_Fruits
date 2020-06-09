# Freshness_Detection_Of_Fruits

Detecting freshness of fruits like apples, oranges and bananas. The dataset is taken from [Fruits fresh and rotten for classification](https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification). The dataset containes ~2700 images of fresh and rotten fruits for classification. The code uses VGG16 model for transfer learning to classify the fruit images as fresh or rotten. 


## Technical Concepts
**VGG16:** The paper can be found [here](https://arxiv.org/pdf/1705.03004.pdf)


## Getting Started

Clone the project repository to your local machine, then follow up with the steps as required.

### Requirements

After cloning the repository, install the necessary requirements for the project.
```
pip install -r requirements.txt
```

### Training

The fruits.h5 file is pre-trained in the images from the [Fruits fresh and rotten for classification](https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification). If you wish to train the model from scratch on your own dataset, prepare your dataset in the following way:
1. Load the images in the "*images*" folder
2. Store the images into the subdirectory of their class-names. Example: if image fresh_apple_01.jpg belongs to the class "Fresh_Apple" then the directory of that image should be /images/Fresh_Apple/fresh_apple_01.jpg

You can then train the model by using the train.py file
```
python train_model.py --dataset dataset
```
![alt text](https://github.com/siddhesh1598/Freshness_Detection_Of_Fruits/blob/master/plot.png?raw=true)

The plot for Training and Validation Loss and Accuracy.

### Testing

To test the model, use the test_model.py file. 
```
python test_model.py
```

You can pass some optional parameters for the main.py file:
1. --face: path to face detector model directory <br>
          *default: face_detector*
2. --model: path to trained face mask detector model <br>
          *default: maskNet.model*
3. --confidence: minimum probability to filter weak detections <br>
          *default: 0.35*



## Authors

* **Siddhesh Shinde** - *Initial work* - [SiddheshShinde](https://github.com/siddhesh1598)


## Acknowledgments

* Dataset: [Medical-Mask-Dataset](https://www.kaggle.com/vtech6/medical-masks-dataset) <br>
Dataset by **Eden Social Welfare Foundation, Taiwan**. (Re-uploaded by [Mikolaj Witkowski](https://www.kaggle.com/vtech6))
