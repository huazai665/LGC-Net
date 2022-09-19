import os

import numpy as np
import torch
from matplotlib import pyplot as plt

import src.dataset as ds
from src.lie_algebra import SO3
from src.utils import pload, pdump, yload, ydump, mkdir, bmv
from src.utils import bmtm, bmtv, bmmt

base_dir = os.path.dirname(os.path.realpath(__file__))
# change to your dir
data_dir = '/home/pcl/Documents/Yaohua/LGC-Net/data/EUROC/'
my_address_tum = '/home/pcl/Documents/Yaohua/LGC-Net/Tum_results/my'
DIG_address_tum = '/home/pcl/Documents/Yaohua/LGC-Net/Tum_results/DIG'
my_address_euroc = '/home/pcl/Documents/Yaohua/LGC-Net/Euroc_results/my'
DIG_address_euroc = '/home/pcl/Documents/Yaohua/LGC-Net/Euroc_results/DIG'


figsize = (20, 12)


def integrate_with_quaternions_superfast(N, raw_us, net_us, gt):
    dt = 0.005
    imu_qs = SO3.qnorm(SO3.qexp(raw_us[:, :3].cuda().double()*dt))
    net_qs = SO3.qnorm(SO3.qexp(net_us[:, :3].cuda().double()*dt))
    Rot0 = SO3.qnorm(gt['qs'][:2].cuda().double())
    imu_qs[0] = Rot0[0]
    net_qs[0] = Rot0[0]

    N = np.log2(imu_qs.shape[0])
    for i in range(int(N)):
        k = 2**i
        imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:-k], imu_qs[k:]))
        net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:-k], net_qs[k:]))

    if int(N) < N:
        k = 2**int(N)
        k2 = imu_qs[k:].shape[0]
        imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:k2], imu_qs[k:]))
        net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:k2], net_qs[k:]))

    imu_Rots = SO3.from_quaternion(imu_qs).float()
    net_Rots = SO3.from_quaternion(net_qs).float()
    return net_qs.cpu(), imu_Rots, net_Rots

def display_test(dataset, mode):
    roes = {
        'Rots': [],
        'yaws': [],
    }
    # self.to_open_vins(dataset)
    for i, seq in enumerate(dataset.sequences):
        print('\n', 'Results for sequence ' + seq )
        # get ground truth
        gt = dataset.load_gt(i)
        Rots = SO3.from_quaternion(gt['qs'].cuda())
        gt['Rots'] = Rots.cpu()
        gt['rpys'] = SO3.to_rpy(Rots).cpu()
        # get data and estimate
        my_net_us = pload(my_address_euroc, seq, 'results.p')['hat_xs']
        DIG_net_us = pload(DIG_address_euroc, seq, 'results.p')['hat_xs']

        raw_us, _ = dataset[i]
        N = my_net_us.shape[0]
        my_gyro_corrections = (raw_us[:, :3] - my_net_us[:N, :3])
        DIG_gyro_corrections = (raw_us[:, :3] - DIG_net_us[:N, :3])

        ts = torch.linspace(0, N*0.005, N)

        # convert()
        # s -> min
        l = 1 / 60
        ts *= l
        # rad -> deg
        l = 180 / np.pi
        my_gyro_corrections *= l
        DIG_gyro_corrections *=1
        gt['rpys'] *= l
        # plot_gyro()
        N1 = raw_us.shape[0]
        raw_us = raw_us[:, :3]
        net_us = my_net_us[:, :3]
        DIG_us = DIG_net_us[:, :3]

        net_qs, imu_Rots, net_Rots = integrate_with_quaternions_superfast(N1, raw_us, net_us, gt)
        DIG_qs, DIG_imu_Rots, DIG_Rots = integrate_with_quaternions_superfast(N1, raw_us, DIG_us, gt)
        imu_rpys = 180 / np.pi * SO3.to_rpy(imu_Rots).cpu()
        net_rpys = 180 / np.pi * SO3.to_rpy(net_Rots).cpu()

        DIG_imu_rpys = 180 / np.pi * SO3.to_rpy(DIG_imu_Rots).cpu()
        DIG_net_rpys = 180 / np.pi * SO3.to_rpy(DIG_Rots).cpu()
        # plot_orientation(imu_rpys, net_rpys, N)
        title = "Attitude estimation"
        gt_rpys = gt['rpys'][:N1]
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(20,14))

        # parameters = {'xtick.labelsize': 15,
        #               'ytick.labelsize': 15}
        # plt.rcParams.update(parameters)

        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        for i in range(3):
            axs[i].plot(ts, gt_rpys[:, i], color='black', label=r'Ground Truth')
            axs[i].plot(ts, imu_rpys[:, i], color='red', label=r'Raw IMU')
            axs[i].plot(ts, DIG_net_rpys[:, i], color='green', label=r'DIG')
            axs[i].plot(ts, net_rpys[:, i], color='blue', label=r'LGC-Net')
            axs[i].set_xlim(ts[0], ts[-1])

            axs[i].title.set_size(35)
            axs[i].xaxis.label.set_size(30)
            axs[i].yaxis.label.set_size(30)
            # plt.tick_params(labelsize=25)
            axs[i].tick_params(labelsize=30)
            # axs[i].xtick.label.set_size(50)

        # savefig(axs, fig, 'orientation')
        if isinstance(axs, np.ndarray):
            for i in range(len(axs)):
                axs[i].grid()
                axs[0].legend(ncol=4, loc='upper left', fontsize=30)
        else:
            axs.grid()
            axs.legend(ncol=4, loc='upper left', fontsize=30)
        fig.tight_layout()
        fig.savefig(os.path.join(my_address_euroc, seq, 'orientation' + '.png'))
        # plot_orientation_error(imu_Rots, net_Rots, N)
        gt_Rots = gt['Rots'][:N1].cuda()
        raw_err = 180 / np.pi * SO3.log(bmtm(imu_Rots, gt_Rots)).cpu()
        net_err = 180 / np.pi * SO3.log(bmtm(net_Rots, gt_Rots)).cpu()
        DIG_err = 180 / np.pi * SO3.log(bmtm(DIG_Rots, gt_Rots)).cpu()
        title = "$SO(3)$ orientation error"
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(20, 12))
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')
        for i in range(3):
            axs[i].plot(ts, raw_err[:, i], color='red', label=r'Raw IMU')
            axs[i].plot(ts, DIG_err[:, i], color='green', label=r'DIG')
            axs[i].plot(ts, net_err[:, i], color='blue', label=r'LGC-Net')
            axs[i].set_ylim(-10, 10)
            axs[i].set_xlim(ts[0], ts[-1])

            axs[i].title.set_size(35)
            axs[i].xaxis.label.set_size(30)
            axs[i].yaxis.label.set_size(30)
            axs[i].tick_params(labelsize=30)
        # savefig(axs, fig, 'orientation_error')
        if isinstance(axs, np.ndarray):
            for i in range(len(axs)):
                axs[i].grid()
                axs[i].legend(ncol=3, loc='upper left', fontsize=30)
        else:
            axs.grid()
            axs.legend(ncol=3, loc='upper left', fontsize=30)
        fig.tight_layout()
        fig.savefig(os.path.join(my_address_euroc, seq, 'orientation_error' + '.png'))
        # plot_gyro_correction()
        title = "Gyro correction" + " for sequence " + seq.replace("_", " ")
        ylabel = 'gyro correction (deg/s)'
        fig, axs = plt.subplots(figsize=(20, 12))
        axs.set(xlabel='$t$ (min)', ylabel=ylabel, title=title)
        plt.plot(ts, my_gyro_corrections, label=r'LGC-Net')
        axs.set_xlim(ts[0], ts[-1])

        axs.title.set_size(35)
        axs.xaxis.label.set_size(30)
        axs.yaxis.label.set_size(30)
        axs.tick_params(labelsize=30)
        # savefig(ax, fig, 'gyro_correction')
        if isinstance(axs, np.ndarray):
            for i in range(len(axs)):
                axs[i].grid()
                # axs[i].legend()
                axs[i].legend(fontsize=30)
        else:
            axs.grid()
            axs.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(my_address_euroc, seq, 'gyro_correction' + '.png'))
        plt.show()

def test(dataset_class, dataset_params, modes):
    """test a network once training is over"""

    # test on each type of sequence
    for mode in modes:
        dataset = dataset_class(**dataset_params, mode=mode)
        # self.loop_test(dataset, criterion)
        display_test(dataset, mode)

if __name__ == '__main__':
    # base_dir = os.path.dirname(os.path.realpath(__file__))
    # data_dir = '/home/pcl/Documents/Yaohua/denoise-imu-gyro/data/TUM/'
    dataset_params = {
        # where are raw data ?
        'data_dir': data_dir,
        # where record preloaded data ?
        'predata_dir': os.path.join(base_dir, 'data/EUROC'),
        # set train, val and test sequence
        'train_seqs': [
            'MH_01_easy',
            'MH_03_medium',
            'MH_05_difficult',
            'V1_02_medium',
            'V2_01_easy',
            'V2_03_difficult'
        ],
        'val_seqs': [
            'MH_01_easy',
            'MH_03_medium',
            'MH_05_difficult',
            'V1_02_medium',
            'V2_01_easy',
            'V2_03_difficult',
        ],
        'test_seqs': [
            'MH_02_easy',
            'MH_04_difficult',
            'V2_02_medium',
            'V1_03_difficult',
            'V1_01_easy',
        ],
        # size of trajectory during training
        'N': 32 * 500,  # should be integer * 'max_train_freq'
        'min_train_freq': 16,
        'max_train_freq': 32,
    }
    # define datasets
    dataset_class = ds.EUROCDataset
    # dataset_test = ds.TUMVIDataset(**dataset_params, mode='test')
    test(dataset_class, dataset_params, ['test'])
