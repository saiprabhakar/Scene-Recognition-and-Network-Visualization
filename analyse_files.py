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
#analyse_dir = 'analysis/analysis_results_floor/'

analyse_dir = 'analysis/ana_floor_now/'
#analyse_dir = 'analysis/ana_places_now/'

#analyse_dir = 'analysis/analysis_results_floor_seperate_final/'
#analyse_dir = 'analysis/analysis_results_places_seperate_final/'

seperate_scr = 0

# load list of all pickel files
visu_file_s = [
    f for f in listdir(analyse_dir)
    if (isfile(join(analyse_dir, f)) and splitext(f)[1] == '.pickle')
]
occ_config = [10, 50, 100]
grad_config = [0, 2, 5]
exci_config = [1]
visu_file_s_o = visu_file_s
visu_file_s = []
for i in range(len(visu_file_s_o)):
    if int(visu_file_s_o[i][9]) == seperate_scr:
        visu_file_s.append(visu_file_s_o[i])

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
    print 'seperate', seperate
    if seperate == 0:
        print 'best_dilate', best_dilate,
        print 'best_patch', best_patch,
        print 'outputBlobName', outputBlobName,
        print 'outputLayerName', outputLayerName,
        print 'size_patch_s', size_patch_s,
        print 'dilate_iteration_s', dilate_iteration_s,

    print visu_file
    if seperate_scr == 1:
        o_avg_rel_inc, o_avg_rel_inc_fin, o_avg_req_mask_percent, o_avg_req_dilate_iter = sum_analysis(
            o_rel_inc, o_rel_inc_fin, o_req_mask_percent, o_req_dilate_iter,
            gd_imgs)
        o_per_area_fin = o_avg_rel_inc_fin / o_avg_req_mask_percent
        o_best_id = o_per_area_fin.argmax()
        o_best_id1 = o_avg_rel_inc.argmax()
        print 'o_avg_rel_inc--------', o_avg_rel_inc.astype(int)
        print "occ c1", occ_config[o_best_id1]
        #print 'o_avg_rel_inc_fin----', o_avg_rel_inc_fin.astype(int)
        #print 'o_avg_req_dilate_iter-', o_avg_req_dilate_iter.astype(int)
        #print 'o_avg_req_mask_percent', o_avg_req_mask_percent.astype(int)
        print 'AIM per area', o_per_area_fin
        print "occ", occ_config[o_best_id]
        print '-----'

        g_avg_rel_inc, g_avg_rel_inc_fin, g_avg_req_mask_percent, g_avg_req_dilate_iter = sum_analysis(
            g_rel_inc, g_rel_inc_fin, g_req_mask_percent, g_req_dilate_iter,
            gd_imgs)
        g_per_area_fin = g_avg_rel_inc_fin / g_avg_req_mask_percent
        g_best_id = g_per_area_fin.argmax()
        g_best_id1 = g_per_area_fin.argmax()
        print 'g_avg_rel_inc--------', g_avg_rel_inc.astype(int)
        print "grad c1", grad_config[g_best_id1]
        #print 'g_avg_rel_inc_fin----', g_avg_rel_inc_fin.astype(int)
        #print 'g_avg_req_dilate_iter-', g_avg_req_dilate_iter.astype(int)
        #print 'g_avg_req_mask_percent', g_avg_req_mask_percent.astype(int)
        print 'AIM per area', g_per_area_fin
        print "grad", grad_config[g_best_id]
        print '-----'

        e_avg_rel_inc, e_avg_rel_inc_fin, e_avg_req_mask_percent, e_avg_req_dilate_iter = sum_analysis(
            e_rel_inc, e_rel_inc_fin, e_req_mask_percent, e_req_dilate_iter,
            gd_imgs)
        e_per_area_fin = e_avg_rel_inc_fin / e_avg_req_mask_percent
        print 'e_avg_rel_inc--------', e_avg_rel_inc.astype(int)
        #print 'e_avg_rel_inc_fin----', e_avg_rel_inc_fin.astype(int)
        #print 'e_avg_req_dilate_iter-', e_avg_req_dilate_iter.astype(int)
        #print 'e_avg_req_mask_percent', e_avg_req_mask_percent.astype(int)
        print 'AIM per area', e_per_area_fin
        print '-----'
    elif seperate_scr == 0:
        com_og_avg_rel_inc, com_og_avg_rel_inc_fin, com_og_avg_req_mask_percent, com_og_avg_req_dilate_iter = sum_analysis(
            com_og_rel_inc, com_og_rel_inc_fin, com_og_req_mask_percent,
            com_og_req_dilate_iter, gd_imgs)
        com_og_per_area_fin = com_og_avg_rel_inc_fin / com_og_avg_req_mask_percent
        #com_og_best_id = com_og_per_area_fin.argmax()
        #com_og_occ_id = com_og_best_id % len(occ_config)
        #com_og_grad_id = com_og_best_id / len(occ_config)
        print 'com_og_avg_rel_inc--------', com_og_avg_rel_inc.astype(int)
        #print 'com_og_avg_rel_inc_fin----', com_og_avg_rel_inc_fin.astype(int)
        #print 'com_og_avg_req_dilate_iter-', com_og_avg_req_dilate_iter.astype(int)
        #print 'com_og_avg_req_mask_percent', com_og_avg_req_mask_percent.astype(
        #    int)
        print 'AIM per area', com_og_per_area_fin

        print '-----'
        com_ge_avg_rel_inc, com_ge_avg_rel_inc_fin, com_ge_avg_req_mask_percent, com_ge_avg_req_dilate_iter = sum_analysis(
            com_ge_rel_inc, com_ge_rel_inc_fin, com_ge_req_mask_percent,
            com_ge_req_dilate_iter, gd_imgs)
        com_ge_per_area_fin = com_ge_avg_rel_inc_fin / com_ge_avg_req_mask_percent
        #com_ge_best_id = com_ge_per_area_fin.argmax()
        #com_ge_occ_id = com_ge_best_id % len(occ_config)
        #com_ge_grad_id = com_ge_best_id / len(occ_config)
        print 'com_ge_avg_rel_inc--------', com_ge_avg_rel_inc.astype(int)
        #print 'com_ge_avg_rel_inc_fin----', com_ge_avg_rel_inc_fin.astype(int)
        #print 'com_ge_avg_req_dilate_iter-', com_ge_avg_req_dilate_iter.astype(int)
        #print 'com_ge_avg_req_mask_percent', com_ge_avg_req_mask_percent.astype(
        #    int)
        print 'AIM per area', com_ge_per_area_fin

        print '-----'
        com_oe_avg_rel_inc, com_oe_avg_rel_inc_fin, com_oe_avg_req_mask_percent, com_oe_avg_req_dilate_iter = sum_analysis(
            com_oe_rel_inc, com_oe_rel_inc_fin, com_oe_req_mask_percent,
            com_oe_req_dilate_iter, gd_imgs)
        com_oe_per_area_fin = com_oe_avg_rel_inc_fin / com_oe_avg_req_mask_percent
        #com_oe_best_id = com_oe_per_area_fin.argmax()
        #com_oe_occ_id = com_oe_best_id % len(occ_config)
        #com_oe_grad_id = com_oe_best_id / len(occ_config)
        print 'com_oe_avg_rel_inc--------', com_oe_avg_rel_inc.astype(int)
        #print 'com_oe_avg_rel_inc_fin----', com_oe_avg_rel_inc_fin.astype(int)
        #print 'com_oe_avg_req_dilate_iter-', com_oe_avg_req_dilate_iter.astype(int)
        #print 'com_oe_avg_req_mask_percent', com_oe_avg_req_mask_percent.astype(
        #    int)
        print 'AIM per area', com_oe_per_area_fin

    print '-----'

    print '-----'
print "done"
