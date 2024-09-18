import argparse
import email
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA','ntuhrnet/xsub', 'ntuhrnet/xview', 'ntu120hrnet/xsub', 'ntu120hrnet/xset',"k400"},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1.1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)

    parser.add_argument('--ema',
                        default=False,
                        help='EMA',
                        type=bool)
    parser.add_argument('--s2',
                        default=False,
                        help='s2',
                        type=bool)
    arg = parser.parse_args()

    dataset = arg.dataset
    if 'UCLA' in arg.dataset:
        label = []
        with open('/root/work/CTR-GCN-main/data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif "k400" in arg.dataset:
        label = []
        with open("data/kinetics/val_label.pkl", 'rb') as f:
            sample_name, label = pickle.load(f)
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('/root/work/CTR-GCN-main/data/' + 'ntu120/' + 'NTU120_CSub.npz')
            if 'hrnet' in arg.dataset:
                npz_data = np.load('/root/work/CTR-GCN-main/data/' + 'HRNet/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('/root/work/CTR-GCN-main/data/' + 'ntu120/' + 'NTU120_CSet.npz')
            if 'hrnet' in arg.dataset:
                npz_data = np.load('/root/work/CTR-GCN-main/data/' + 'HRNet/' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('/root/work/CTR-GCN-main/data/' + 'ntu/' + 'NTU60_CS.npz')
            if 'hrnet' in arg.dataset:
                npz_data = np.load('/root/work/CTR-GCN-main/data/' + 'HRNet/' + 'NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
            
        elif 'xview' in arg.dataset:
            npz_data = np.load('/root/work/CTR-GCN-main/data/' + 'newntu/'+"NTU60_CV.npz")
            if 'hrnet' in arg.dataset:
                npz_data = np.load('/root/work/CTR-GCN-main/data/' + 'HRNet/' + 'NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError
    # plist = ["work_dir_used/ntu60/xsub/bonesearched0.5/epoch1_test_ema_score.pkl","work_dir_used/ntu60/xsub/bonesearched/epoch1_test_ema_score.pkl"
    #          ,"work_dir_used/ntu60/xsub/jointsearched0.5/epoch1_test_ema_score.pkl","work_dir_used/ntu60/xsub/jointsearched/epoch1_test_ema_score.pkl"
    # ]#,"work_dir/ntu120/xset/bone_12_100110/epoch1_test_score.pkl","work_dir/ntu120/xset/joint_12_100110/epoch1_test_score.pkl"]
    # plist = ["work_dir_used/ntu60/xview/bone0.5_100110/epoch1_test_ema_score.pkl","work_dir_used/ntu60/xview/bone_100110/epoch1_test_ema_score.pkl"
    #          ,"work_dir_used/ntu60/xview/joint0.5_100110/epoch1_test_ema_score.pkl","work_dir_used/ntu60/xview/joint_100110/epoch1_test_ema_score.pkl"
    # ]#,"work_dir/ntu120/xset/bone_12_100110/epoch1_test_score.pkl","work_dir/ntu120/xset/joint_12_100110/epoch1_test_score.pkl"]
    plist = ["work_dir/newk400/k400bone0.5_300e_seed_new/epoch1_test_score.pkl","work_dir/newk400/k400bone_300e_seed_new/epoch1_test_ema_score.pkl"
             ,"work_dir/newk400/k400joint0.5_300e_new/epoch1_test_score.pkl","work_dir/newk400/k400joint_300e_seed_new/epoch1_test_score.pkl"
    ]#,"work_dir_v1/ntu60/xsub252/TSGCNext_jointmodern_3/epoch1_test_score.pkl","work_dir_v1/ntu60/xsub432/TSGCNext_bonemodern_3/epoch1_test_score.pkl"
    #]
    rlist = []
    for pp in plist:
        with open(pp, 'rb') as ri:
            ri = list(pickle.load(ri).items())
            rlist.append(ri)

    
    right_num = total_num = right_num_5 = 0

    #if (arg.joint_motion_dir is not None and arg.bone_motion_dir is not None) or arg.s2:
    print("1")
    arg.alpha = [0, 0, 0, 1]
    print(arg.alpha )
    for i in tqdm(range(len(label))):
        r = 0
        l = label[i]
        for j,ri in enumerate(rlist):
            _, r1i = ri[i]
            r += r1i * arg.alpha[j]# + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
    right_num = total_num = right_num_5 = 0
    