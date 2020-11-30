img1 = imread('file01.jpg');
imshow(img1)
deepnet = alexnet
pred1 = classify(deepnet,img1)

Task 1
Save layers
ly=deepnet.Layers



Task 2
Extract first layer
inlayer = ly(1)



Task 3
Extract input size
insz= inlayer.InputSize



Task 4
Extract last layer

outlayer = ly(end)
%ly(25) will also work


Task 5
Extract class names
categorynames = outlayer.Classes


Investigate Predictions
Instructions are in the task pane to the left. Complete and submit each task one at a time.

This code loads in an image and imports AlexNet.
img = imread('file01.jpg');
imshow(img)
net = alexnet;
categorynames = net.Layers(end).ClassNames;

Task 1
Classify an image
[pred,scores] = classify(net,img)



Task 2
Display scores

bar(scores)


Task 3
Threshold scores
highscores = scores > 0.01;



Task 4
Display thresholded scores
bar(scores(highscores))



Task 5
Add tick labels


xticklabels(categorynames(highscores))
xticks(1:length(scores(highscores)))
xticklabels(categorynames(highscores))
xtickangle(60)



Create a Datastore
Instructions are in the task pane to the left. Complete and submit each task one at a time.

This code displays the images in the current folder and imports AlexNet.
ls *.jpg
net = alexnet;

Task 1
Create datastore
imds = imageDatastore('file*.jpg')



Task 2
Extract file names
fname = imds.Files



Task 3
Read an image

img = readimage(imds,7);


Task 4
Classify images
preds = classify(net,imds)
max(scores,[],2)


Process Images for Classification
Instructions are in the task pane to the left. Complete and submit each task one at a time.

This code imports and displays the image from the file file01.jpg.
img = imread('file01.jpg');
imshow(img)

Task 1
View image size

sz = size(img)


Task 2
Load network and view input size

net = alexnet;
inlayer = net.Layers(1)
insz = inlayer.InputSize


Task 3
Resize image and display
img = imresize(img,[227 227]);
imshow(img)



Resize Images in a Datastore
Instructions are in the task pane to the left. Complete and submit each task one at a time.

This code displays the images in the current folder and imports AlexNet.
ls *.jpg
net = alexnet

Task 1
Create datastore

imds = imageDatastore('file*.jpg')


Task 2
Create augmentedImageDatastore
auds = augmentedImageDatastore([227 227],imds)




Task 3
Classify datastore

preds = classify(net,auds)


Preprocess Color Using a Datastore
Instructions are in the task pane to the left. Complete and submit each task one at a time.

This code displays the images in the current folder and imports AlexNet.
ls *.jpg
net = alexnet
This code creates an image datastore of these images.
imds = imageDatastore('file*.jpg')

Task 1
Display images in 
montage(imds)



Task 2
Create augmentedImageDatastore
auds = augmentedImageDatastore([227 227],imds,'ColorPreprocessing','gray2rgb')



Task 3
Classify datastore

preds = classify(net,auds)


Create a Datastore Using Subfolders
Instructions are in the task pane to the left. Complete and submit each task one at a time.

This code imports AlexNet
net = alexnet;

Task 1
Create datastore


flwrds = imageDatastore('Flowers','IncludeSubfolders',true)

Task 2
Classify images
preds = classify(net,flwrds)


Label Images in a Datastore
Instructions are in the task pane to the left. Complete and submit each task one at a time.

This code creates a datastore of 960 flower images.
load pathToImages
flwrds = imageDatastore(pathToImages,'IncludeSubfolders',true);
flowernames = flwrds.Labels

Task 1
Create datastore with labels

flwrds = imageDatastore(pathToImages,'IncludeSubfolders',true,'LabelSource','foldernames')

Task 2
Extract new labels
flowernames = flwrds.Labels


Split Data for Training and Testing
Instructions are in the task pane to the left. Complete and submit each task one at a time.

This code creates a datastore of 960 flower images.
load pathToImages
flwrds = imageDatastore(pathToImages,'IncludeSubfolders',true,'LabelSource','foldernames')

Task 1
Split datastore
[flwrTrain,flwrTest] = splitEachLabel(flwrds,0.6)



Task 2
Split datastore randomly


[flwrTrain,flwrTest] = splitEachLabel(flwrds,0.8,'randomized')

Task 3
Split datastore by number of images

[flwrTrain,flwrTest] = splitEachLabel(flwrds,50)



Modify Network Layers
Instructions are in the task pane to the left. Complete and submit each task one at a time.

This code imports AlexNet and extracts its layers.
anet = alexnet;
layers = anet.Layers

Task 1
Create new layer


fc = fullyConnectedLayer(12)

Task 2
Replace 23rd layer
layers(23) = fc



Task 3
Replace last layer
layers(end) = classificationLayer


Set Training Options
Instructions are in the task pane to the left. Complete and submit each task one at a time.

Task 1
Set default options

opts = trainingOptions('sgdm')


Task 2
Set initial learning rate
opts = trainingOptions('sgdm','InitialLearnRate',0.001)



Evaluate Performance
Instructions are in the task pane to the left. Complete and submit each task one at a time.

This code loads the training information of flowernet.
load pathToImages
load trainedFlowerNetwork flowernet info

Task 1
Plot training loss
plot(info.TrainingLoss)



This code creates a datastore of the flower images.
dsflowers = imageDatastore(pathToImages,'IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs,testImgs] = splitEachLabel(dsflowers,0.98);

Task 2
Classify images

flwrPreds = classify(flowernet,testImgs)


