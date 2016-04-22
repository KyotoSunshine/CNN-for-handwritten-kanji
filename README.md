# CNN-for-handwritten-kanji

#####Requirements:  
 - Python 2.7
 - Keras 1.0 with Tensorflow (I used version 0.7) or Theano (I used version 0.8)
 - scipy, numpy
 - hdf5 linux package, and h5py package for python

######Note:  
You can install hdf5 and h5py as follows:  
 - sudo apt-get install libhdf5-serial-dev  
 - sudo pip install h5py  


#####If you want to use a pretrained model to classify your own images:

Download the pretrained model weights file from here:  
https://drive.google.com/open?id=0B-B1607WQeSSRmEtRC1xQW5ZdlU  
and save it as "model_weights.h5" in some folder.

Download the model architecture file from here:  
https://drive.google.com/open?id=0B-B1607WQeSSdWR4M3dXNzN5dGs  
and save it as "model_architecture.json" in the same folder.

Place the file "classify.py" from this repository into the same folder as the above two files.  

Now you can run the pretrained model on your own images as follows:  
 - python classify.py path_to_image1.png path_to_image2.png ...  

You can use absolute or relative paths.

The output will be as folows:  

path_to_image1.png kanji_label_1  
path_to_image2.png kanji_label_2  
...


#####If you want to train the model:  
At first, run  
 - python create_dataset.py path_to_folder  

Here path_to_folder is the folder which contains the folders with images.
This will produce a file named "characters_dataset" which will contain the train/validation/test datasets and labels in numpy arrays.

Now you can train the model on this dataset. Place the "model.py" file in the folder where you have the newly created dataset, and run:  
 - python model.py 

This will produce two files: "model_architecture.json" and "model_weights.h5" which contain the model architecture and weights.  
Then, you can use these files with "classify.py" to make predictions.

----

The folder "logs" contains the training logs of the model. It shows that the validation accuracy (when using the 40000-10000 train-validation split) is about 97.4%.  

The model in the file "model.py" does not use any kind of dataset preprocessing or augmentation. On the other hand, in the folder "Dataset-Augmentation" there is a file "create_augmented_dataset.py" which operates on the file "characters_dataset" and creates another augmented dataset, called "characters_dataset_elastic". It uses elastic distortions (See http://arxiv.org/pdf/1003.0358.pdf for description of the method) to transform the training data and generate very natural looking characters. I had to write it from scratch, as I was unable to find efficient implementation of elastic transformations in any python library. You can see some of the transformed characters in the images located in the same folder.  The newly generated augmented dataset contains ten times as many examples as the original one. It is expected that one can achieve significantly better accuracy using this augmented dataset (see also http://arxiv.org/pdf/1003.0358.pdf).





