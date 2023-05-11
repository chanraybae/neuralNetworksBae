Note: Final code is located in "final.py" ("main.py" is for presentation showcase, "test.py" is ironically for the validation and training phase)
1) A description of the test database you collected or downloaded: What is the size of the database? What is different when compared to the training and validation subsets? 
Why you believe these differences are sufficient to test the generalization capabilities of your final programs?

The test database I downloaded was accessed via Roboflow (https://public.roboflow.com/object-detection/american-sign-language-letters), a public domain data set created by Data Scientist
David Lee, which contains 72 images for the testing folder, but 1728 images in total for the train, validation, and test stages. 
I divided the images into folders for the purposes of Tensorflow and its hierarchy with respect the architecture that my 
code followed, manually labeling them for the final accuracy rating. Compared to the training  (1512 images) and validation (144 images) subsets, this was much smaller number, as the test 
set's goal is to properly evaluate the performance of the trained model on previously unseen image data. However, data augmentation with batch_size = 32, epochs = 100 influenced even more
the training dataset to have more images for the model to learn off of. Therefore, the model could accurately learn with a large set of data, which is why training is so much larger than
the test and validation sequences, the latter of which is only to estimate the model skill and thus again does not require as much data as the training phase. These differences are sufficient,
as there are 26 letters corresponding to signs in the American (Sign) Language. Additionally, the accuracy of the model while training was over 95% and validation about 75%, which appears to
be sufficient, as ASL is a contextual language highly based on movement and finger location precision. Therefore, some unfavorable angles can be misidentified, but an idea can be gained, as
people or location names are spelled out to be intuitive in a real world context. 

2) A classification accuracy achieved on the test set. Use the same metrics as in deliverable #3

In the final model, the classification accuracy achieved on the test set equated to 75.41%, which as aforementioned is sufficient given the angle difficulty and the precise yet contextual 
nature of ASL. Although MediaPipe was used to frame the hands, blurry or unoptimal images can cause confusion; when learning ASL myself, it was often confusing with a human brain to discern
two signs from each other. 

3) Most of you should see worse results on the test set when compared to the results obtained on train/validation sets. You should not be worried about that, but please provide the 
reasons why your solution performs worse (with a few illustrations, such as pictures or videos, what went wrong). What improvements would you make to lower the observed error rates?

Yes, the classification accuracy on the test set was generally much worse than the train/validation sets. This could result from a multitude of reasons, with the primary being overfitting from
the training phase. It was able to train off of many images to contain a generalized understanding, but when blurry images such as this one (https://ibb.co/5xyzTHH) for the letter J are shown 
to the model to identify, it has a harder time identifying the sign. Such a phenomenon can occur due to the close nature of certain signs. Even with mediapipe, there is not the most accurate
identification, including to my own human eyes seeing the difference between the sign for H (https://media.istockphoto.com/id/1182201671/photo/finger-spelling-letter-h-in-asl-on-white-background-american-sign-language-concept.jpg?s=612x612&w=0&k=20&c=CIDY8_ZSabu0Q7mWkOK0G4lDaJ_KLPfpJXhSKFBxzP0=)
and the sign for U (https://www.signingsavvy.com/images/words/alphabet/2/u1.jpg), as the angle is the discerning factor. Other similar signs, such as M (https://www.signingsavvy.com/sign/M/0/fingerspell)
and N (https://www.signingsavvy.com/search/n), were more readily detected through the help of hand and digit detection by MediaPipe, which I used using Google's own reference pages for the 
integrated library. The thumb was discernible through the help of MP. To make improvements, I could upscale the images through MediaPipe to highlight the edges of fingers, which would especially help
in the case of blurry images blending fingers together. I also could have, and once considered, using a pre-trained databse, such as the MobileNetV2 imported via tensorflow.keras.applications, but I
decided that the trajectory of my project should be on optimizing a neural network to the best of my ability. 

Presentation Slides PPT link:
Created "main.py" to showcase the model in a presentation application in a real life, live camera feed setting
https://acrobat.adobe.com/link/review?uri=urn:aaid:scds:US:ea068b46-3b6e-36e4-9e1a-22fd49f90ede
