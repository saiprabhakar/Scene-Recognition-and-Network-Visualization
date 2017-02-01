# scene-recognition-and-visualization
============

This project involves developing one-shot learning methods for indoor sub-scene classification.
Some network visualization techniques will also be implemented.


One-shot learning
----------

Aim is to recognise which floor the robot is currently at.

Methods implemented:

  1. Siamese Network:

    Uses contrastive loss
    ~~~
    python trainSiamese.py
    ~~~

    ![Siamese net](siamese1.png )

  2. Modified Siamese network:

    Uses identification inaddition to contrastive loss
    ~~~
    python trainModifiedSiamese.py
    ~~~

    ![Modified Siamese net (training net)](modified_siamese1.png )

    During test time a single branch with a softmax final layer is used with the trained weights.

trainSiamese.py or trainModifiedSiamese.py
----
To change the fc8 layer size, train, test and to visualize read the comments in the code


Visualization
---------------

Aim is to visualize what parts of the image are important for the classification.

Methods considered:

  1. Occulsion heat map (siamese and modified siamese net)
  2. Class Saliency map (modified siamese net)


# to train modified siamese use trainModifiedSiamese.py -> modifiedSiamese/SiameseTrainer.py
# to visualize any network use visuModels.py -> modifiedSiamese/SiameseTrainer.py
# to analyse visualized files use visuModels.py -> modifiedSiamese/analyse_visu.py
