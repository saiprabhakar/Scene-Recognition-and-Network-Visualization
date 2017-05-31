#/usr/bin/python
# --------------------------------------------------------
# floor_recog
# Written by Sai Prabhakar
# CMU-RI Masters
# --------------------------------------------------------


def create_train_test_split(fileName, ratio):
    #TODO
    #Load images and imagedict
    load_image_name(fileName, class_adju)
    #split into test and train
    #write to txt file
    #return filenames
    pass


def load_image_name(fileName, class_adju):
    '''
    Load file names
    '''
    with open(fileName) as f:
        lines = [line.rstrip('\n') for line in f]
    imlist = []
    imageDict = {}
    for i in lines:
        temp = i.split(' ')
        imageDict[temp[0]] = int(temp[1]) - class_adju
        imlist.append(temp[0])

    return imlist, imageDict


def get_data_prop(dataset):
    if dataset == "places":
        #fileName_test_visu = 'images_all.txt'
        class_size = 365
        class_adju = 0
        im_target_size = 227
        initial_image_size = (256, 256)  #rows, cols
        class_names = [''] * 6  #TODO get actual label from file
    else:
        #fileName_test_visu = 'imagelist_all.txt'
        class_size = 6
        class_adju = 2
        im_target_size = 227
        initial_image_size = (768, 1024)  #rows, cols
        class_names = ['3rd floor', '4rd floor', '5rd floor', '6rd floor',
                       '7rd floor', '8rd floor'
                       ]  # [''] * 6  #TODO get actual label from file
    return class_size, class_adju, im_target_size, initial_image_size, class_names
