import pickle
import numpy as np
from ipdb import set_trace as debug
from os import listdir
from os.path import isfile, join, splitext
import matplotlib.pyplot as plt


def validate(array, validity, thres=0):
    #mask = np.ones(array1.shape)
    if validity.shape[1] == 1:
        mask = np.repeat(validity, array.shape[1], 1)
    else:
        mask = validity

    array1 = array * mask
    mask[array1 < thres] = 0
    array1[array1 < thres] = 0
    return array1, mask


def find_mask_aig(rel_inc_fin, req_dilate_iter, gd_imgs):

    rel_inc_fin0, rel_inc_fin0_mask = validate(rel_inc_fin, gd_imgs)
    rel_inc_fin0_mask[req_dilate_iter < 0] = 0
    rel_inc_fin0_mask[req_dilate_iter > 100] = 0
    return rel_inc_fin0_mask


def sum_analysis(rel_inc, rel_inc_fin, req_mask_percent, req_dilate_iter,
                 gd_imgs):
    rel_inc1, rel_inc1_mask = validate(rel_inc, gd_imgs)

    #find final mask by combining iter and rel inc final
    AIG_mask = find_mask_aig(rel_inc_fin, req_dilate_iter, gd_imgs)
    req_dilate_iter1, req_dilate_iter1_mask = validate(req_dilate_iter,
                                                       AIG_mask)
    rel_inc_fin1, rel_inc_fin1_mask = validate(rel_inc_fin, AIG_mask)
    req_mask_percent1, req_mask_percent1_mask = validate(req_mask_percent,
                                                         AIG_mask)

    avg_rel_inc = rel_inc1.sum(axis=0) / rel_inc1_mask.sum(axis=0)
    avg_rel_inc_fin = rel_inc_fin1.sum(axis=0) / rel_inc_fin1_mask.sum(axis=0)
    avg_req_mask_percent = req_mask_percent1.sum(
        axis=0) / req_mask_percent1_mask.sum(axis=0)
    avg_req_dilate_iter = req_dilate_iter1.sum(
        axis=0) / req_dilate_iter1_mask.sum(axis=0)

    per_area_fin = avg_rel_inc_fin / avg_req_mask_percent
    #print 'o_avg_rel_inc--------', avg_rel_inc.astype(int)
    #print 'AIM per area', per_area_fin

    #debug()

    return avg_rel_inc, avg_rel_inc_fin, avg_req_mask_percent, avg_req_dilate_iter

#analyse_dir = 'analysis/analysis_results_places/'
#analyse_dir = 'analysis/analysis_results_floor/'

#analyse_dir = 'analysis/ana_floor_now/'
#analyse_dir = 'analysis/ana_places_now1/'

#analyse_dir = 'analysis/analysis_results_floor/'
analyse_dir = 'analysis/analysis_results_places/'

seperate_scr = -1  #0

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
    if visu_file_s_o[i][9] == '-':
        spe_file_value = int(visu_file_s_o[i][9:11])
    else:
        spe_file_value = int(visu_file_s_o[i][9])

    if spe_file_value == seperate_scr:
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
    size_image = 0  #doesnt matter
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

        #TODO do negative ignoring

        print 'o_avg_rel_inc--------', o_avg_rel_inc.astype(int)
        print "occ c1", occ_config[o_best_id1]
        #print 'o_avg_rel_inc_fin----', o_avg_rel_inc_fin.astype(int)
        #print 'o_avg_req_dilate_iter-', o_avg_req_dilate_iter.astype(int)
        #print 'o_avg_req_mask_percent', o_avg_req_mask_percent.astype(int)
        print 'AIM per area', o_per_area_fin
        print "occ", occ_config[o_best_id]

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
    elif seperate_scr == -1:
        com_oge_avg_rel_inc, com_oge_avg_rel_inc_fin, com_oge_avg_req_mask_percent, com_oge_avg_req_dilate_iter = sum_analysis(
            com_oge_rel_inc, com_oge_rel_inc_fin, com_oge_req_mask_percent,
            com_oge_req_dilate_iter, gd_imgs)
        com_oge_per_area_fin = com_oge_avg_rel_inc_fin / com_oge_avg_req_mask_percent
        com_oge_best_id = com_oge_per_area_fin.argmax()
        com_oge_best_id1 = com_oge_avg_rel_inc.argmax()

        print 'com_oge_avg_rel_inc--------', com_oge_avg_rel_inc.astype(int)
        print 'AIM per area', com_oge_per_area_fin

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
