import numpy as np
import support_based as spb

path = 'E:/Project20_ECG_foundation_model/Model/model11_SimCLR_moredata/save/cnntransf_nomix_2_20_2500_20_DzwU/'
pretrain_res = spb.read_res(path, 'prettain-res')
fineturn_nopretrain_res = spb.read_res(path, 'fineturn-nopretrain-res')
fineturn_pretrain_full_res = spb.read_res(path, 'fineturn-pretrain_full-res')
fineturn_pretrain_only_classifier_res = spb.read_res(path, 'fineturn-pretrain_only_classifier-res')

fineturn_nopretrain_res = np.array([x[1] for x in fineturn_nopretrain_res[1]])
fineturn_pretrain_full_res = np.array([x[1] for x in fineturn_pretrain_full_res[1]])
fineturn_pretrain_only_classifier_res = np.array([x[1] for x in fineturn_pretrain_only_classifier_res[1]])