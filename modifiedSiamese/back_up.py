#for j in range(7):
#    size = 17 + 2*j
#    print size
#    kernel = np.ones((size,size),np.float32)/(size*size)
#    dst = cv2.filter2D(img,-1,kernel)
#
#    blurred_blob = h2._get_image_blob_from_image(dst, self.meanarr, self.im_target_size)
#    blur_prob, all_prob = get_prediction_prob(blurred_blob, imageDict[im_name])
#    print blur_prob, all_prob
#
