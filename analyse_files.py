import pickle
import numpy as np
from ipdb import set_trace as debug
from os import listdir
from os.path import isfile, join, splitext


def validate(array, validity):
    array1 = array * np.repeat(validity, array.shape[1], 1)
    array1[array1 < 0] = 0
    return array1


def sum_analysis(rel_inc, rel_inc_fin, req_mask_percent, req_dilate_iter,
                 gd_imgs):
    rel_inc1 = validate(rel_inc, gd_imgs)
    rel_inc_fin1 = validate(rel_inc_fin, gd_imgs)
    req_mask_percent1 = validate(req_mask_percent, gd_imgs)
    req_dilate_iter1 = validate(req_dilate_iter, gd_imgs)

    avg_rel_inc = rel_inc1.sum(axis=0) / gd_imgs.sum(axis=0)
    avg_rel_inc_fin = rel_inc_fin1.sum(axis=0) / gd_imgs.sum(axis=0)
    avg_req_mask_percent = req_mask_percent1.sum(axis=0) / gd_imgs.sum(axis=0)
    avg_req_dilate_iter = req_dilate_iter1.sum(axis=0) / gd_imgs.sum(axis=0)

    return avg_rel_inc, avg_rel_inc_fin, avg_req_mask_percent, avg_req_dilate_iter

#analyse_dir = 'analysis/analysis_results_places/'
analyse_dir = 'analysis/analysis_results_floor/'
# load list of all pickel files
visu_file_s = [
    f for f in listdir(analyse_dir)
    if (isfile(join(analyse_dir, f)) and splitext(f)[1] == '.pickle')
]

for i in range(len(visu_file_s)):
    visu_file = analyse_dir + visu_file_s[i]
    with open(visu_file) as f:
        [data] = pickle.load(f)

    # TODO
    #
    for key, val in data.items():
        exec(key + '=val')

    diff = orig_prob_s - mod_prob_s
    gd_imgs = np.zeros(diff.shape)
    gd_imgs[diff > 0] = 1

    print visu_file, '-------- 00000000000000000000000000'
    o_avg_rel_inc, o_avg_rel_inc_fin, o_avg_req_mask_percent, o_avg_req_dilate_iter = sum_analysis(
        o_rel_inc, o_rel_inc_fin, o_req_mask_percent, o_req_dilate_iter,
        gd_imgs)
    print 'o_avg_rel_inc--------', o_avg_rel_inc.astype(int)
    print 'o_avg_rel_inc_fin----', o_avg_rel_inc_fin.astype(int)
    print 'o_avg_req_mask_percent', o_avg_req_mask_percent.astype(int)
    print 'o_avg_req_dilate_iter-', o_avg_req_dilate_iter.astype(int)
    print '-----'

    g_avg_rel_inc, g_avg_rel_inc_fin, g_avg_req_mask_percent, g_avg_req_dilate_iter = sum_analysis(
        g_rel_inc, g_rel_inc_fin, g_req_mask_percent, g_req_dilate_iter,
        gd_imgs)
    print 'g_avg_rel_inc--------', g_avg_rel_inc.astype(int)
    print 'g_avg_rel_inc_fin----', g_avg_rel_inc_fin.astype(int)
    print 'g_avg_req_mask_percent', g_avg_req_mask_percent.astype(int)
    print 'g_avg_req_dilate_iter-', g_avg_req_dilate_iter.astype(int)
    print '-----'

    com_og_avg_rel_inc, com_og_avg_rel_inc_fin, com_og_avg_req_mask_percent, com_og_avg_req_dilate_iter = sum_analysis(
        com_og_rel_inc, com_og_rel_inc_fin, com_og_req_mask_percent,
        com_og_req_dilate_iter, gd_imgs)
    print 'com_og_avg_rel_inc--------', com_og_avg_rel_inc.astype(int)
    print 'com_og_avg_rel_inc_fin----', com_og_avg_rel_inc_fin.astype(int)
    print 'com_og_avg_req_mask_percent', com_og_avg_req_mask_percent.astype(
        int)
    print 'com_og_avg_req_dilate_iter-', com_og_avg_req_dilate_iter.astype(int)
    print '-----'

    neg_og_avg_rel_inc, neg_og_avg_rel_inc_fin, neg_og_avg_req_mask_percent, neg_og_avg_req_dilate_iter = sum_analysis(
        neg_og_rel_inc, neg_og_rel_inc_fin, neg_og_req_mask_percent,
        neg_og_req_dilate_iter, gd_imgs)
    print 'neg_og_avg_rel_inc--------', neg_og_avg_rel_inc.astype(int)
    print 'neg_og_avg_rel_inc_fin----', neg_og_avg_rel_inc_fin.astype(int)
    print 'neg_og_avg_req_mask_percent', neg_og_avg_req_mask_percent.astype(
        int)
    print 'neg_og_avg_req_dilate_iter-', neg_og_avg_req_dilate_iter.astype(int)
    print '-----'

    neg_go_avg_rel_inc, neg_go_avg_rel_inc_fin, neg_go_avg_req_mask_percent, neg_go_avg_req_dilate_iter = sum_analysis(
        neg_go_rel_inc, neg_go_rel_inc_fin, neg_go_req_mask_percent,
        neg_go_req_dilate_iter, gd_imgs)
    print 'neg_go_avg_rel_inc--------', neg_go_avg_rel_inc.astype(int)
    print 'neg_go_avg_rel_inc_fin----', neg_go_avg_rel_inc_fin.astype(int)
    print 'neg_go_avg_req_mask_percent', neg_go_avg_req_mask_percent.astype(
        int)
    print 'neg_go_avg_req_dilate_iter-', neg_go_avg_req_dilate_iter.astype(int)
    print '-----'
    print '-----'
    print '-----'
    print '-----'

    print "done"
