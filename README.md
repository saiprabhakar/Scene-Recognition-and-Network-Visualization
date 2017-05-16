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
  3. Excitation backprop (modified siamese net)

Visualization evaluation Metrics:
  
  1. ACG
  2. CCG
  
Scripts Used:
1. To train modified siamese use 'trainModifiedSiamese.py -> modifiedSiamese/SiameseTrainer.py'
2. To visualize any network use 'visuModels.py -> modifiedSiamese/SiameseTrainer.py'
3. To analyse visualized files (metrics) use 'visuModels.py -> modifiedSiamese/analyse_visu.py'
4. To find average metrics generate metrics from 'visuModels.py -> modifiedSiamese/analyse_visu.py' and then use 'analyse_files.py'
5. To generate images for the paper 'gen_img.py -> modifiedSiamese/gen_images.py'
6. To generate heatmaps for a specific setting use 'visuScene.py -> modifiedSiamese/SiameseTrainer.py'
7. To explain scene generate object detection from 'yolo900', generate scene visualization heatmap from 'visuModels.py -> modifiedSiamese/SiameseTrainer.py' or 'visuScene.py -> modifiedSiamese/SiameseTrainer.py', then use 'explainScene.py'


