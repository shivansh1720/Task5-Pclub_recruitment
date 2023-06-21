# Feed Forward Nueral Network
## Task 5

The code is written in python and I have explained it in the following lines.

### Following need to be installed in your local machine to run the code:
 - numpy
 - pandas
 - argparse
 - os
 - cv2
 - pickle


### To install cv2- 
    pip install opencv-python    

### To install argparse
    pip install argparse      

##### Else my code covers every details of a ML model from activation layers to optimizers and to loss functions.
### Explanation of my code

#### Import
Fisrt I installed all the dependencies and then imported them 
#### Data path 
I defined the data path from my local machine to test the code.Because I couldn't figure out how to get input and output seperately from train.csv and test.csv as they were meant to be different files.

### Class Neural Network
- Initialized the parameters form the input
- Made weights and biases
- I have wriiten down the loss functions SQ and CLE
- 

### Training Model
- Trained the model
- set the loss function

### Calling main
- taking bash input from argparse and making the network using those parameters

### What you can't find in there?
- The model is not complete, I have altough mentioned the defaults of bash script but still it would not run
- The main problem I encountered was beacuse of the CIFAR-10 dataset, I was not able to load the dataset into my code however I tried many ways. Eve when I tried to upload it from local host, it returned error that is "It can't read a file of type none'
- The dataset on unziping gives in the format of images and hence is difficult to use.
- Setting it up consumed most of my time.
- I have wriiten the code comparing test_predictions and correct output but it is commented out as I could  not find the correct output file.
- One more point, I couldn't add flattening layer at end of my dataset.
