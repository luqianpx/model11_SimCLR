import numpy as np
import support_based as spb
import os

fo_li = os.listdir('./save')
all_res = []
for fo in fo_li:
    path = './save/' + fo + '/'
    pretrain_res = spb.read_res(path, 'prettain-res')
    fineturn_nopretrain_res = spb.read_res(path, 'fineturn-nopretrain-res')
    fineturn_pretrain_full_res = spb.read_res(path, 'fineturn-pretrain_full-res')
    fineturn_pretrain_only_classifier_res = spb.read_res(path, 'fineturn-pretrain_only_classifier-res')

    print('')
    fineturn_nopretrain_res = np.array([x[1] for x in fineturn_nopretrain_res[1]])
    fineturn_pretrain_full_res = np.array([x[1] for x in fineturn_pretrain_full_res[1]])
    fineturn_pretrain_only_classifier_res = np.array([x[1] for x in fineturn_pretrain_only_classifier_res[1]])

    me_fineturn_nopretrain_res = np.mean(fineturn_nopretrain_res[:, -50:], 0)
    me_fineturn_pretrain_full_res = np.mean(fineturn_pretrain_full_res[:, -50:], 0)
    me_fineturn_pretrain_only_classifier_res = np.mean(fineturn_pretrain_only_classifier_res[:, -50:], 0)

    res = np.stack([ me_fineturn_nopretrain_res, me_fineturn_pretrain_full_res, me_fineturn_pretrain_only_classifier_res], 0)

    all_res.append([fo, res, pretrain_res])