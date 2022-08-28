import torch
from torch.utils.data import Dataset
import h5py

def fmult(x, S, P):
    '''
    if len(x.shape) == 3:
        x = x.permute([1, 2, 0]).contiguous()
    else:
        x = x.permute([0, 2, 3, 1]).contiguous()
        S = S.permute([0, 2, 3, 4, 1]).contiguous()
    '''

    x = torch.view_as_complex(x)
    S = torch.view_as_complex(S)

    # x, groundtruth, shape: batch, width, height; dtype: complex
    # S, sensitivity maps, shape: batch, coils, width, height; dtype: complex
    # P, sampling mask, shape: batch, width, height; dtype: float/bool

    # compute forward of fast MRI, y = PFSx

    # S
    x = x.unsqueeze(1)

    y = x * S

    # F
    y = torch.fft.fft2(y)

    # P
    P = P.unsqueeze(1)
    y = y * P

    y = torch.view_as_real(y)
    S = torch.view_as_real(S)

    return y, S

def ftran(y, S, P):
    y = torch.view_as_complex(y)
    S = torch.view_as_complex(S)

    # y, under-sampled measurements, shape: batch, coils, width, height; dtype: complex
    # S, sensitivity maps, shape: batch, coils, width, height; dtype: complex
    # P, sampling mask, shape: batch, width, height; dtype: float/bool

    # compute adjoint of fast MRI, x = S^H F^H P^H x
    # P^H
    y = y * P.unsqueeze(1)

    # F^H
    x = torch.fft.ifft2(y)

    # S^H
    x = x * torch.conj(S)
    x = x.sum(1)

    x = torch.view_as_real(x)

    return x

import numpy as np

class MoDLDataset(Dataset):
    #def __init__(self, path="C:\\Users\\pyiph\\Desktop\\CIG\\Supervised_DeCoLearn\\20220718_DU_DEQ\\DU_DEQ\\dataset\\dataset.hdf5"):
    def __init__(self, path='C:/Users/pyiph/Desktop/CIG Project/dataset.hdf5'):
        with h5py.File(path, 'r') as f:
            self.P = np.concatenate([f['trnMask'], f['tstMask']], 0)
            self.P = torch.from_numpy(self.P).to(torch.float32)
            self.S = np.concatenate([f['trnCsm'], f['tstCsm']], 0)
            self.S = np.stack([self.S.real, self.S.imag], 1)
            self.S = torch.from_numpy(self.S)
            self.x = np.concatenate([f['trnOrg'], f['tstOrg']], 0)
            self.x = np.stack([self.x.real, self.x.imag], 1)
            self.x = torch.from_numpy(self.x)

            self.x = self.x.permute([0, 2, 3, 1]).contiguous()
            self.S = self.S.permute([0, 2, 3, 4, 1]).contiguous()

            self.y, self.S = fmult(self.x, self.S, self.P)
            self.x_init = ftran(self.y, self.S, self.P)


    def __len__(self):

        return self.y.shape[0]

    def __getitem__(self, item):

        return self.x_init[item], self.P[item], self.S[item], self.y[item], self.x[item]


def load_synthetic_MoDL_dataset(
        root_folder: str = './dataset/',
        translation: tuple = [0, 0],
        rotate: float = 0,
        scale: int = 0,
        nonlinear_P: int = 1000,
        nonlinear_theta: int = 5,
        nonlinear_sigma: int = 10,
        mask_type: str = 'cartesian',
        mask_fold: int = 4,
        input_snr: int = 40,
        mul_coil: bool = True
):
    #######################################
    # Source H5 File
    #######################################

    dataset_folder = root_folder

    source_folder_name = 'source_MoDL'

    source_h5_path = dataset_folder + '%s.h5' % source_folder_name

    source_path = dataset_folder + '%s/' % source_folder_name
    check_and_mkdir(source_path)

    source_qc   = dataset_folder + '%s_qc/' % source_folder_name
    check_and_mkdir(source_qc)

    if not os.path.exists(source_h5_path):
        print("Not Found Source H5 File. Start Generating it ...")

        with h5py.File(source_h5_path, 'w') as source_h5:

            with h5py.File(dataset_folder + 'dataset.hdf5') as f:
                for k in f.keys():
                    print(f[k].shape, k, f[k].dtype)

                x = np.concatenate([f['trnOrg'], f['tstOrg']], 0)
                x = np.stack([x.real, x.imag], 1)
                source_h5.create_dataset(name='x', data=x)

                s = np.concatenate([f['trnCsm'], f['tstCsm']], 0)
                source_h5.create_dataset(name='s', data=s)

                to_tiff(np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2), path=source_qc + 'fixed_x.tiff', is_normalized=False)

    else:
        print("Found Source H5 File.")

    #############################
    # Alignment H5 File
    #############################
    alignment_file_name = "alignment_translation=[%s]_rotate=[%s]_scale=[%s]_nonlinear_P=[%d]_theta=[%d]_sigma=[%d]" % (
        str(translation), str(rotate), str(scale), nonlinear_P, nonlinear_theta, nonlinear_sigma
    )

    alignment_h5_path = source_path + "%s.h5" % alignment_file_name

    alignment_path = source_path + "%s/" % alignment_file_name
    check_and_mkdir(alignment_path)

    alignment_qc = source_path + "%s_qc/" % alignment_file_name
    check_and_mkdir(alignment_qc)

    if not os.path.exists(alignment_h5_path):
        print("Not Found Alignment H5 File. Start Generating it ...")

        with h5py.File(source_h5_path, 'r') as source_h5:
            x = torch.from_numpy(source_h5['x'][:])
            print(x.shape, x.dtype)

            x_real, x_imag = x[:, 0], x[:, 1]

            moved_x = []

            n_slice = x.shape[0]
            for index_slice in tqdm(range(n_slice)):
                affine_grid = generate_affine_grid(
                    imgSize=x_real[index_slice].unsqueeze(0).shape,
                    translation=[
                        2 * translation[0] * np.random.rand(1) - translation[0],
                        2 * translation[1] * np.random.rand(1) - translation[1],
                    ],
                    rotate=2 * rotate * np.random.rand(1) - rotate,
                    scale=2 * scale * np.random.rand(1) - scale + 1,
                )

                non_linear_grid = generate_nonlinear_grid(
                    imgSize=x_real[index_slice].unsqueeze(0).shape,
                    P=nonlinear_P,
                    theta=nonlinear_theta,
                    sigma=nonlinear_sigma,
                    mask=1
                )

                moved_x_real = torch.nn.functional.grid_sample(x_real[index_slice].unsqueeze(0).unsqueeze(0), affine_grid, mode='bilinear')
                moved_x_real = torch.nn.functional.grid_sample(moved_x_real, non_linear_grid, mode='bilinear').squeeze(0).squeeze(0)

                moved_x_imag = torch.nn.functional.grid_sample(x_imag[index_slice].unsqueeze(0).unsqueeze(0), affine_grid, mode='bilinear')
                moved_x_imag = torch.nn.functional.grid_sample(moved_x_imag, non_linear_grid, mode='bilinear').squeeze(0).squeeze(0)

                moved_x.append(torch.stack([moved_x_real, moved_x_imag], 0))

            moved_x = torch.stack(moved_x, 0)

            with h5py.File(alignment_h5_path, 'w') as alignment_h5:
                alignment_h5.create_dataset(name='moved_x', data=moved_x)

            to_tiff(torch.sqrt(moved_x[:, 0] ** 2 + moved_x[:, 1] ** 2), path=alignment_qc + 'moved_x.tiff',
                    is_normalized=False)

    else:
        print("Found Alignment H5 File.")

    #############################
    # MRI H5 File
    #############################
    mri_file_name = 'mri_mask_type=[%s]_mask_fold=[%s]_input_snr=[%.2d]' % (
        mask_type, mask_fold, input_snr)

    mri_h5_path = alignment_path + "%s.h5" % mri_file_name

    mri_path = alignment_path + "%s/" % mri_file_name
    check_and_mkdir(mri_path)

    mri_qc = alignment_path + "%s_qc/" % mri_file_name
    check_and_mkdir(mri_qc)



    if not os.path.exists(mri_h5_path):
        print("Not Found MRI H5 File. Start Generating it ...")

        with h5py.File(source_h5_path, 'r') as source_h5:
            with h5py.File(alignment_h5_path, 'r') as alignment_h5:
                fixed_x = torch.from_numpy(source_h5['x'][:])
                moved_x = torch.from_numpy(alignment_h5['moved_x'][:])
                sensitivity_map = torch.from_numpy(source_h5['s'][:])

                num_shape = fixed_x.shape[0]
                fixed_y, fixed_mask, fixed_y_tran, moved_y, moved_mask, moved_y_tran, moved_y_warped_truth, moved_y_tran_warped_truth = [], [], [], [], [], [], [], []
                for i_shape in tqdm(range(num_shape)):
                    fixed_y_cur, fixed_mask_cur, fixed_noise_cur = fmult(fixed_x[i_shape], type_=mask_type, fold=mask_fold, sensitivity_map = sensitivity_map[i_shape], mul_coil = mul_coil, noise_snr=input_snr)
                    moved_y_cur, moved_mask_cur, moved_noise_cur = fmult(moved_x[i_shape], type_=mask_type, fold=mask_fold, sensitivity_map = sensitivity_map[i_shape], mul_coil= mul_coil, noise_snr=input_snr)
                    moved_y_warped_truth_cur, moved_mask_warped_truth_cur, _ = fmult(fixed_x[i_shape], type_=mask_type, fold=mask_fold, sensitivity_map=sensitivity_map[i_shape], mul_coil = mul_coil, noise_snr=input_snr, pre_generate_noise=moved_noise_cur, pre_generated_mask=moved_mask_cur)

                    fixed_y_tran_cur = ftran(fixed_y_cur, sensitivity_map[i_shape], fixed_mask_cur, mul_coil=mul_coil)
                    moved_y_tran_cur = ftran(moved_y_cur, sensitivity_map[i_shape], moved_mask_cur, mul_coil=mul_coil)
                    moved_y_tran_warped_truth_cur = ftran(moved_y_warped_truth_cur, sensitivity_map[i_shape], moved_mask_warped_truth_cur, mul_coil=mul_coil)

                    fixed_y.append(fixed_y_cur)
                    fixed_mask.append(fixed_mask_cur)
                    fixed_y_tran.append(fixed_y_tran_cur)

                    moved_y.append(moved_y_cur)
                    moved_mask.append(moved_mask_cur)
                    moved_y_tran.append(moved_y_tran_cur)

                    moved_y_warped_truth.append(moved_y_warped_truth_cur)
                    moved_y_tran_warped_truth.append(moved_y_tran_warped_truth_cur)


                fixed_y, fixed_mask, fixed_y_tran, moved_y, moved_mask, moved_y_tran, moved_y_warped_truth, moved_y_tran_warped_truth = [
                    torch.stack(i, 0) for i in [fixed_y, fixed_mask, fixed_y_tran, moved_y, moved_mask, moved_y_tran, moved_y_warped_truth, moved_y_tran_warped_truth]]

                with h5py.File(mri_h5_path, 'w') as mri_h5:
                    mri_h5.create_dataset(name='fixed_y', data=fixed_y)
                    mri_h5.create_dataset(name='fixed_mask', data=fixed_mask)
                    mri_h5.create_dataset(name='fixed_y_tran', data=fixed_y_tran)
                    mri_h5.create_dataset(name='moved_y', data=moved_y)
                    mri_h5.create_dataset(name='moved_mask', data=moved_mask)
                    mri_h5.create_dataset(name='moved_y_tran', data=moved_y_tran)
                    mri_h5.create_dataset(name='moved_y_warped_truth', data=moved_y_warped_truth)
                    mri_h5.create_dataset(name='moved_y_tran_warped_truth', data=moved_y_tran_warped_truth)

                if mul_coil is True:
                    #to_tiff(torch.sqrt(fixed_y[..., 0] ** 2 + fixed_y[..., 1] ** 2), path=mri_qc + 'fixed_y.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(fixed_mask[..., 0] ** 2 + fixed_mask[..., 1] ** 2), path=mri_qc + 'fixed_mask.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(fixed_y_tran[..., 0] ** 2 + fixed_y_tran[..., 1] ** 2), path=mri_qc + 'fixed_y_tran.tiff', is_normalized=False)
                    #to_tiff(torch.sqrt(moved_y[..., 0] ** 2 + moved_y[..., 1] ** 2), path=mri_qc + 'moved_y.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(moved_mask[..., 0] ** 2 + moved_mask[..., 1] ** 2), path=mri_qc + 'moved_mask.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(moved_y_tran[..., 0] ** 2 + moved_y_tran[..., 1] ** 2), path=mri_qc + 'moved_y_tran.tiff', is_normalized=False)
                    #to_tiff(torch.sqrt(moved_y_warped_truth[..., 0] ** 2 + moved_y_warped_truth[..., 1] ** 2), path=mri_qc + 'moved_y_warped_truth.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(moved_y_tran_warped_truth[..., 0] ** 2 + moved_y_tran_warped_truth[..., 1] ** 2), path=mri_qc + 'moved_y_tran_warped_truth.tiff', is_normalized=False)

                else:
                    to_tiff(torch.sqrt(fixed_y[..., 0] ** 2 + fixed_y[..., 1] ** 2), path=mri_qc + 'fixed_y.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(fixed_mask[..., 0] ** 2 + fixed_mask[..., 1] ** 2), path=mri_qc + 'fixed_mask.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(fixed_y_tran[..., 0] ** 2 + fixed_y_tran[..., 1] ** 2), path=mri_qc + 'fixed_y_tran.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(moved_y[..., 0] ** 2 + moved_y[..., 1] ** 2), path=mri_qc + 'moved_y.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(moved_mask[..., 0] ** 2 + moved_mask[..., 1] ** 2), path=mri_qc + 'moved_mask.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(moved_y_tran[..., 0] ** 2 + moved_y_tran[..., 1] ** 2), path=mri_qc + 'moved_y_tran.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(moved_y_warped_truth[..., 0] ** 2 + moved_y_warped_truth[..., 1] ** 2), path=mri_qc + 'moved_y_warped_truth.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(moved_y_tran_warped_truth[..., 0] ** 2 + moved_y_tran_warped_truth[..., 1] ** 2), path=mri_qc + 'moved_y_tran_warped_truth.tiff', is_normalized=False)

    else:
        print("Found MRI H5 File.")

    #######################################
    # Read Images from H5 Files
    #######################################
    source_h5 = h5py.File(source_h5_path, 'r')
    alignment_h5 = h5py.File(alignment_h5_path, 'r')
    mri_h5 = h5py.File(mri_h5_path, 'r')


    ret_cur = {
        'fixed_x': source_h5['x'][:],
        'sensitivity_map': source_h5['s'][:],
        'moved_x': alignment_h5['moved_x'][:],

        'moved_y': mri_h5['moved_y'][:],
        'moved_mask': mri_h5['moved_mask'][:],
        'moved_y_tran': np.transpose(mri_h5['moved_y_tran'][:], [0, 3, 1, 2]),

        'fixed_y': mri_h5['fixed_y'][:],
        'fixed_mask': mri_h5['fixed_mask'][:],
        'fixed_y_tran': np.transpose(mri_h5['fixed_y_tran'][:], [0, 3, 1, 2]),

        'moved_y_warped_truth': mri_h5['moved_y_warped_truth'][:],
        'moved_y_tran_warped_truth': np.transpose(mri_h5['moved_y_tran_warped_truth'][:], [0, 3, 1, 2])
    }


    ret = ret_cur

    return ret