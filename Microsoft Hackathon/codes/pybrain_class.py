# -*- coding: cp1252 -*-
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from sklearn.cross_validation import train_test_split
## Importing all modules.
 
CLASSES = 3 # One for each digit
data = pd.read_csv('BingHackathonTrainingData.csv')
X,y = data.summary, data.topic_id
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
 
# Declaring the structure of the training and testing dataset
trndata = ClassificationDataSet(784, 1, nb_classes=CLASSES)
tstdata = ClassificationDataSet(784, 1, nb_classes=CLASSES)
 
# Adding data to our training and testing dataset
training = get_label_data('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
testing = get_label_data('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
for i in range(len(testing['x'])):
    tstdata.addSample(ravel(testing['x'][i]), [testing['y'][i]])
for i in range(len(training['x'])):
    trndata.addSample(ravel(training['x'][i]), [training['y'][i]])
 
# For neural network classification, it is highly advisable to
# encode classes with one output neuron per class. Note that this
# operation duplicates the original targets and stores them in an (integer) field named ‘clas# s’.
trndata._convertToOneOfMany() # this is still needed to make the fnn feel comfy
tstdata._convertToOneOfMany()
 
fnn = buildNetwork(trndata.indim, 250, trndata.outdim, outclass=SoftmaxLayer)
 
trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1,
                          verbose=True, weightdecay=0.01,
                          learningrate=0.01 ,
                          lrdecay=1)
for i in range(30):
    trainer.trainEpochs(1) # Train the network for some epochs. Usually you would set something like 5 here, but for visualization purposes we do this one epoch at a time.
    trnresult = percentError(trainer.testOnClassData(),
                             trndata['class'])
    tstresult = percentError(trainer.testOnClassData(
                             dataset=tstdata), tstdata['class'])
 
    print("epoch: %4d" % trainer.totalepochs,
                 "  train error: %5.2f%%" % trnresult,
                 "  test error: %5.2f%%" % tstresult)
