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


#####Performance
Below is the performance on validation set when using 40000-10000 training-validation split.  

- 97.4% with no preprocessing and no data augmentation, achieved by "model.py".
- 98.5% with elastic deformations as a dataset augmentation method (increasing dataset ten times), achieved by "model_for_augmented_dataset.py".
- 97.2% with no preprocessing and no data augmentation, achieved by "model30x30.py". This version rescales the input images to 30 by 30 pixels as opposed to 32 by 32 pixels in all the other models.



#####If you want to use a pretrained model to classify your own images:

######To use the model which was trained on the non-augmented original dataset (97.4% accuracy):  
Download the pretrained model weights file from here:  
https://drive.google.com/open?id=0B-B1607WQeSSRmEtRC1xQW5ZdlU  
and save it as "model_weights.h5" in some folder.

Download the model architecture file from here:  
https://drive.google.com/open?id=0B-B1607WQeSSdWR4M3dXNzN5dGs  
and save it as "model_architecture.json" in the same folder.


######To use the model which was trained on the augmented dataset (98.5% accuracy):  
Download the pretrained model weights file from here:  
https://drive.google.com/open?id=0B-B1607WQeSSaXdJUzlFRy01TkU  
and rename and save it as "model_weights.h5" in some folder.

Download the model architecture file from here:  
https://drive.google.com/open?id=0B-B1607WQeSSUUVKVkdacmFiOTQ  
and rename and save it as "model_architecture.json" in the same folder.


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

######For the case of the model on the non-augmented original dataset:

Place the "model.py" file in the folder where you have the newly created dataset, and run:  
 - python model.py 

This will produce two files: "model_architecture.json" and "model_weights.h5" which contain the model architecture and weights. 
Then, you can use these files with "classify.py" to make predictions.

######For the case of the model on the augmented dataset:

Place the file "create_augmented_dataset.py" in the same folder as the newly created dataset and run the following command:  
 - python create_augmented_dataset.py

Now place the "model_for_augmented_dataset.py" file in the folder where you have the newly created augmented dataset, and run: 
 - python model_for_augmented_dataset.py 

This will produce two files: "augmented_model_architecture.json" and "augmented_model_weights.h5" which contain the model architecture and weights.  

Then, you can use these files with "classify.py" to make predictions, but keep in mind that "classify.py" expects the model weights to be in the file "model_weights.h5" and model architecture to be in the file "model_architecture.json", so you need to rename the files to suit it.

----

The folder "logs" contains the training logs of the "model.py", "model30x30.py" and "model_for_augmented_dataset.py" models. 

The file named "create_augmented_dataset.py" operates on the file "characters_dataset" (which must be created first using the "create_dataset.py" script) and creates an augmented dataset, called "characters_dataset_elastic". It uses elastic distortions (See http://arxiv.org/pdf/1003.0358.pdf for description of the method) to transform the training data and generate very natural looking characters. The newly generated augmented dataset contains ten times as many examples as the original one. Using this dataset with the "model_for_augmented_dataset.py" model, I was able to achieve 98.5% accuracy on the validation set (see also http://arxiv.org/pdf/1003.0358.pdf).  

I had to write the elastic distortion function from scratch, as I was unable to find efficient implementation of elastic transformations in any python library. You can see some of the transformed character images in the folder "Dataset-Augmentation". 
