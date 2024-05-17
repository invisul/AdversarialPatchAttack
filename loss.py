import torch
from kornia.geometry.conversions import angle_axis_to_rotation_matrix, angle_axis_to_quaternion, QuaternionCoeffOrder
from kornia.color.hsv import HsvToRgb
from pytorch_msssim import SSIM, MS_SSIM


def test_model(model, criterion, img1, img2, intrinsic, scale_gt, motions_target, target_pose,
               window_size=None, device=None):
    if window_size is None:
        if device is None:
            motions, flow = model.test_batch(img1, img2, intrinsic, scale_gt)
            crit = criterion((motions, flow), scale_gt, motions_target, target_pose).detach().cpu()
            return (motions, flow), crit

        img1_device = img1.clone().detach().to(device)
        img2_device = img2.clone().detach().to(device)
        intrinsic_device = intrinsic.clone().detach().to(device)
        scale_gt_device = scale_gt.clone().detach().to(device)
        motions_target_device = motions_target.clone().detach().to(device)
        target_pose_device = target_pose.clone().detach().to(device)

        motions_device, flow_device = model.test_batch(img1_device, img2_device, intrinsic_device, scale_gt_device)
        crit_device = criterion((motions_device, flow_device), scale_gt_device,
                                motions_target_device, target_pose_device)
        motions = motions_device.clone().detach().cpu()
        flow = flow_device.clone().detach().cpu()
        crit = crit_device.clone().detach().cpu()

        del img1_device
        del img2_device
        del intrinsic_device
        del scale_gt_device
        del motions_target_device
        del target_pose_device
        del motions_device
        del flow_device
        del crit_device
        torch.cuda.empty_cache()

        return (motions, flow), crit

    data_ind = list(range(img1.shape[0] + 1))
    window_start_list = data_ind[0::window_size]
    window_end_list = data_ind[window_size::window_size]
    if window_end_list[-1] != data_ind[-1]:
        window_end_list.append(data_ind[-1])
    motions_window_list = []
    flow_window_list = []
    if device is None:
        for window_idx, window_end in enumerate(window_end_list):
            window_start = window_start_list[window_idx]

            img1_window = img1[window_start:window_end].clone().detach()
            img2_window = img2[window_start:window_end].clone().detach()
            intrinsic_window = intrinsic[window_start:window_end].clone().detach()
            scale_gt_window = scale_gt[window_start:window_end].clone().detach()

            motions_window, flow_window = model.test_batch(img1_window, img2_window, intrinsic_window, scale_gt_window)
            motions_window_list.append(motions_window)
            flow_window_list.append(flow_window)

            del img1_window
            del img2_window
            del intrinsic_window
            del scale_gt_window
            torch.cuda.empty_cache()

        motions = torch.cat(motions_window_list, dim=0)
        flow = torch.cat(flow_window_list, dim=0)
        crit = criterion((motions, flow), scale_gt, motions_target, target_pose).detach().cpu()
        del motions_window_list
        del flow_window_list
        torch.cuda.empty_cache()
        return (motions, flow), crit

    for window_idx, window_end in enumerate(window_end_list):
        window_start = window_start_list[window_idx]

        img1_window = img1[window_start:window_end].clone().detach().to(device)
        img2_window = img2[window_start:window_end].clone().detach().to(device)
        intrinsic_window = intrinsic[window_start:window_end].clone().detach().to(device)
        scale_gt_window = scale_gt[window_start:window_end].clone().detach().to(device)

        motions_window, flow_window = model.test_batch(img1_window, img2_window, intrinsic_window, scale_gt_window)
        motions_window_list.append(motions_window)
        flow_window_list.append(flow_window)

        del img1_window
        del img2_window
        del intrinsic_window
        del scale_gt_window
        torch.cuda.empty_cache()

    motions_device = torch.cat(motions_window_list, dim=0)
    flow_device = torch.cat(flow_window_list, dim=0)
    scale_gt_device = scale_gt.clone().detach().to(device)
    motions_target_device = motions_target.clone().detach().to(device)
    target_pose_device = target_pose.clone().detach().to(device)

    crit_device = criterion((motions_device, flow_device), scale_gt_device, motions_target_device, target_pose_device)
    motions = motions_device.clone().detach().cpu()
    flow = flow_device.clone().detach().cpu()
    crit = crit_device.clone().detach().cpu()

    del scale_gt_device
    del motions_target_device
    del target_pose_device
    del motions_device
    del flow_device
    del crit_device
    del motions_window_list
    del flow_window_list
    torch.cuda.empty_cache()

    return (motions, flow), crit


def rtvec_to_pose(rtvec):
    pose = torch.zeros(rtvec.shape[0], 4, 4, device=rtvec.device, dtype=rtvec.dtype)

    pose[:, 0:3, 0:3] = angle_axis_to_rotation_matrix(rtvec[:, 3:6])
    pose[:, 0:3, 3] = rtvec[:, 0:3]
    pose[:, 3, 3] = 1
    return pose


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        print("using mean MSE from the clean flow for regularization of flow in attacks")
        self.MSELoss = torch.nn.MSELoss(reduction='mean')

    def forward(self, x, y):
        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        diff = self.MSELoss(x, y)
        return diff


class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        print("using cosine similarity for regularization of flow in attacks")

    def forward(self, gt, pred, epsilon=1e-8):
        # _, _, h_pred, w_pred = pred.size()
        # 0) get the dimesions of the optical flow images: batch_size, num_of_channels, height, width
        bs, nc, h_gt, w_gt = gt.size()
        # u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
        # pred = torch.nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
        # u_pred = pred[:,0,:,:] * (w_gt/w_pred)
        # v_pred = pred[:,1,:,:] * (h_gt/h_pred)

        # 1) Compute the cosine similarity over the first two channels, which are usually unit vectors of the optical flow direction
        similarity = 1 - torch.cosine_similarity(gt[:, :2], pred)

        # 2) If there is a 3rd channel, it usually represents the magnitude of the optical flow vector.
        #    We will weigh the sum of similarities by the magnitude.
        if nc == 3:
            valid = gt[:, 2, :, :]
            similarity = similarity * valid
            avg_sim = similarity.sum() / (valid.sum() + epsilon)
        else:
            avg_sim = similarity.sum(dim=(1,2)) / ( h_gt * w_gt)

        return avg_sim

class InverseCosineSimilarityLoss(torch.nn.Module):
    def __init__(self):
        super(InverseCosineSimilarityLoss, self).__init__()
        print("using cosine similarity for regularization of flow in attacks")

    def forward(self, gt, pred, epsilon=1e-8):
        # _, _, h_pred, w_pred = pred.size()
        # 0) get the dimesions of the optical flow images: batch_size, num_of_channels, height, width
        bs, nc, h_gt, w_gt = gt.size()
        # u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
        # pred = torch.nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
        # u_pred = pred[:,0,:,:] * (w_gt/w_pred)
        # v_pred = pred[:,1,:,:] * (h_gt/h_pred)

        # 1) Compute the cosine similarity over the first two channels, which are usually unit vectors of the optical flow direction
        similarity = torch.cosine_similarity(gt[:, :2], pred)

        # 2) If there is a 3rd channel, it usually represents the magnitude of the optical flow vector.
        #    We will weigh the sum of similarities by the magnitude.
        if nc == 3:
            valid = gt[:, 2, :, :]
            similarity = similarity * valid
            avg_sim = similarity.sum() / (valid.sum() + epsilon)
        else:
            avg_sim = similarity.sum(dim=(1,2)) / ( h_gt * w_gt)

        return avg_sim


def optical_flow_to_hsv(optical_flow: torch.Tensor) -> torch.Tensor:
    n, c, h, w = optical_flow.size()
    hsv = torch.zeros((n, 3, h, w), device=optical_flow.device)
    hsv[:, 1, :, :] = 1.
    hsv[:, 0, :, :] = torch.atan2(optical_flow[:, 1, :, :], optical_flow[:, 0, :, :]) 
    hsv[:, 2, :, :] = torch.linalg.norm(optical_flow, dim=1)
    hsv[:, 2, :, :] -= hsv[:, 2, :, :].view(-1, h*w).min(axis=1)[0][:, None, None]
    hsv[:, 2, :, :] /= hsv[:, 2, :, :].view(-1, h*w).max(axis=1)[0][:, None, None]
    return hsv


class OpticalFlow2HSV(torch.nn.Module):
    def __init__(self):
        super(OpticalFlow2HSV, self).__init__()

    def forward(self, optical_flow: torch.Tensor) -> torch.Tensor:
        return optical_flow_to_hsv(optical_flow)


class SSIMLoss(torch.nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        print("using 1-SSIM for regularization of flow in attacks")
        self.SSIM = SSIM(data_range=1, size_average=False, channel=3)
        self.HSV2RGB = HsvToRgb()
        self.OF2HSV = OpticalFlow2HSV()

    def forward(self, gt, pred):
        gt_rgb = self.HSV2RGB(self.OF2HSV(gt))
        pred_rgb = self.HSV2RGB(self.OF2HSV(pred))
        return 1 - self.SSIM(gt_rgb, pred_rgb)


class MSSSIMLoss(torch.nn.Module):
    def __init__(self):
        super(MSSSIMLoss, self).__init__()
        print("using 1-SSIM for regularization of flow in attacks")
        self.MSSSIM = MS_SSIM(data_range=1, size_average=False, channel=3, win_size=7)
        self.HSV2RGB = HsvToRgb()
        self.OF2HSV = OpticalFlow2HSV()

    def forward(self, gt, pred):
        gt_rgb = self.HSV2RGB(self.OF2HSV(gt))
        pred_rgb = self.HSV2RGB(self.OF2HSV(pred))
        return 1 - self.MSSSIM(gt_rgb, pred_rgb)


class CombinedFlowCriterion(torch.nn.Module):
    def __init__(self, flow_crit_str: str):
        super(CombinedFlowCriterion, self).__init__()
        print(f"using multiple criteria for flow attacks: {flow_crit_str}")

        crits =flow_crit_str.split('+')
        self.crit_list = []
        for crit in crits:
            if crit == 'mse':
                self.crit_list.append(MSELoss())
            elif crit == 'cosine_similarity':
                self.crit_list.append(CosineSimilarityLoss())
            elif crit == 'ssim':
                self.crit_list.append(SSIMLoss())
            elif crit == 'msssim':
                self.crit_list.append(MSSSIMLoss())
            else:
                raise ValueError(f"Unknown criterion {crit}")

    def forward(self, gt, pred):
        sum = torch.tensor(0, dtype= torch.float64, device=gt.device)
        for crit in self.crit_list:
            sum += crit(gt, pred)
        return sum / len(self.crit_list)


class CalcCriterion:
    def __init__(self, criterion_class, values= None):
        if values is not None:
                self.criterion = criterion_class(values)
        else:
                self.criterion = criterion_class()

    def apply(self, output, target):
        return self.criterion(output, target)

    def __call__(self, output, target):
        return self.apply(output, target)


class VOCriterion:
    def __init__(self, t_crit='rms', rot_crit='none', flow_crit='none', target_t_crit='none',
                 t_factor=1.0, rot_factor=1.0, flow_factor=1.0, target_t_factor=1.0):

        self.criterion_str = 't_crit_' + str(t_crit) + '_factor_' + str(t_factor).replace('.', '_')

        self.criterion_str += '_rot_crit_' + str(rot_crit)
        if rot_crit != "none":
            self.criterion_str += '_factor_' + str(rot_factor).replace('.', '_')

        self.criterion_str += '_flow_crit_' + str(flow_crit)
        if flow_crit != "none":
            self.criterion_str += '_factor_' + str(flow_factor).replace('.', '_')

        self.criterion_str += '_target_t_crit_' + str(target_t_crit)
        if target_t_crit != "none":
            self.criterion_str += '_factor_' + str(target_t_factor).replace('.', '_')

        print("initializing loss with criteria:")
        print(self.criterion_str)
        self.t_factor = t_factor
        self.rot_factor = rot_factor
        self.flow_factor = flow_factor
        self.target_t_factor = target_t_factor

        if t_crit == 'partial_rms':
            self.calc_t_crit = self.calc_partial_poses_t
        elif t_crit == 'mean_partial_rms':
            self.calc_t_crit = self.calc_mean_partial_poses_t
        else:
            self.calc_t_crit = self.calc_cumul_poses_t

        if rot_crit == 'quat_product':
            self.calc_rot_crit = self.calc_rot_quat_product
        else:
            self.calc_rot_crit = self.calc_none

        if flow_crit == 'mse':
            self.calc_flow_crit = CalcCriterion(MSELoss)
        elif flow_crit == 'cosine_similarity':
            self.calc_flow_crit = CalcCriterion(CosineSimilarityLoss)
        elif flow_crit == 'ssim':
            self.calc_flow_crit = CalcCriterion(SSIMLoss)
        elif flow_crit == 'msssim':
            self.calc_flow_crit = CalcCriterion(MSSSIMLoss)
        elif '+' in flow_crit:
            self.calc_flow_crit = CalcCriterion(CombinedFlowCriterion,flow_crit)
        elif flow_crit == 'inverse_cosine':
            self.calc_flow_crit = CalcCriterion(InverseCosineSimilarityLoss)
        else:
            self.calc_flow_crit = self.calc_none

        self.calc_target_t_product = True

    def apply(self, model_output, scale, motions_gt, target_pose, flow_clean=None):
        motions, flow = model_output
        if target_pose is not None and self.calc_target_t_product:
            t_crit, target_t_crit = self.calc_t_crit(motions, motions_gt, target_pose)
        else:
            t_crit, target_t_crit = self.calc_t_crit(motions, motions_gt, None)
        rot_crit = torch.zeros(motions.shape[0] + 1,
                               device=motions.device, dtype=motions.dtype)
        flow_crit = torch.zeros(motions.shape[0] + 1,
                                device=motions.device, dtype=motions.dtype)
        rot_crit[1:] = scale * self.calc_rot_crit(motions, motions_gt)
        flow_crit[1:] = scale * self.calc_flow_crit(flow, flow_clean)
        return self.t_factor * t_crit + self.target_t_factor * target_t_crit + \
               self.rot_factor * rot_crit + self.flow_factor * flow_crit

    def __call__(self, model_output, scale, motions_gt, target_pose, flow_clean=None) -> torch.tensor:
        return self.apply(model_output, scale, motions_gt, target_pose, flow_clean)

    def calc_none(self, est, preprocessed_gt):
        return 0

    def calc_partial_poses_t(self, motions, motions_gt, target_pose):
        rel_poses = self.rtvec_to_pose(motions)
        rel_poses_gt = self.rtvec_to_pose(motions_gt)
        t_errors_tot = torch.zeros(rel_poses.shape[0] + 1,
                                   device=rel_poses.device, dtype=rel_poses.dtype)
        target_t_errors_tot = torch.zeros(rel_poses.shape[0] + 1,
                                   device=rel_poses.device, dtype=rel_poses.dtype)
        for traj_s_idx in range(rel_poses.shape[0]):
            partial_traj = rel_poses[traj_s_idx:]
            partial_traj_gt = rel_poses_gt[traj_s_idx:]
            cumul_poses = self.cumulative_poses(partial_traj)
            cumul_poses_gt = self.cumulative_poses(partial_traj_gt)
            t_error, target_t_error = self.translation_error(cumul_poses, cumul_poses_gt, target_pose)
            t_errors_tot[traj_s_idx:] += t_error
            target_t_errors_tot[traj_s_idx:] += target_t_error
        return t_errors_tot, target_t_errors_tot

    def calc_mean_partial_poses_t(self, motions, motions_gt, target_pose):
        rel_poses = self.rtvec_to_pose(motions)
        rel_poses_gt = self.rtvec_to_pose(motions_gt)
        t_errors_tot = torch.zeros(rel_poses.shape[0] + 1,
                                   device=rel_poses.device, dtype=rel_poses.dtype)
        target_t_errors_tot = torch.zeros(rel_poses.shape[0] + 1,
                                   device=rel_poses.device, dtype=rel_poses.dtype)
        for traj_s_idx in range(rel_poses.shape[0]):
            max_traj_size = rel_poses.shape[0] - traj_s_idx + 1
            partial_traj = rel_poses[traj_s_idx:]
            partial_traj_gt = rel_poses_gt[traj_s_idx:]
            cumul_poses = self.cumulative_poses(partial_traj)
            cumul_poses_gt = self.cumulative_poses(partial_traj_gt)
            t_error, target_t_error = self.translation_error(cumul_poses, cumul_poses_gt, target_pose)
            t_errors_tot[:max_traj_size] += t_error
            target_t_errors_tot[:max_traj_size] += target_t_error
        t_errors_traj_num = list(range(rel_poses.shape[0] + 1, 0, -1))
        t_errors_traj_num = torch.tensor(t_errors_traj_num, device=rel_poses.device, dtype=rel_poses.dtype)
        t_errors_mean = t_errors_tot / t_errors_traj_num
        target_t_errors_mean = target_t_errors_tot / t_errors_traj_num
        return t_errors_mean, target_t_errors_mean

    def calc_cumul_poses_t(self, motions, motions_gt, target_pose):
        rel_poses = self.rtvec_to_pose(motions)
        rel_poses_gt = self.rtvec_to_pose(motions_gt)
        cumul_poses = self.cumulative_poses(rel_poses)
        cumul_poses_gt = self.cumulative_poses(rel_poses_gt)
        t_error, target_t_error = self.translation_error(cumul_poses, cumul_poses_gt, target_pose)
        return t_error, target_t_error

    def calc_rot_quat_product(self, motions, motions_gt):
        traj_rot_quat = angle_axis_to_quaternion(motions[:, 3:], order=QuaternionCoeffOrder.WXYZ)
        traj_rot_quat_gt = angle_axis_to_quaternion(motions_gt[:, 3:], order=QuaternionCoeffOrder.WXYZ)
        r_errors = self.rotation_quat_product(traj_rot_quat, traj_rot_quat_gt)
        return r_errors

    def cumulative_poses(self, rel_poses):
        cumulative_poses = torch.zeros(rel_poses.shape[0] + 1, rel_poses.shape[1], rel_poses.shape[2],
                                       device=rel_poses.device, dtype=rel_poses.dtype)
        curr_cumulative_pose = torch.eye(4, device=rel_poses.device, dtype=rel_poses.dtype)
        cumulative_poses[0] = curr_cumulative_pose
        for pose_idx, rel_pose in enumerate(rel_poses):
            curr_cumulative_pose = curr_cumulative_pose.mm(rel_pose)
            cumulative_poses[pose_idx + 1] = curr_cumulative_pose
        return cumulative_poses

    def translation_error(self, cumul_poses, cumul_poses_gt, target):
        cumul_delta_t = cumul_poses[:, 0:3, 3] - cumul_poses_gt[:, 0:3, 3]
        t_error = torch.norm(cumul_delta_t, p=2, dim=1)
        t_target_error = 0
        if target is not None:
            target_gt_t = (cumul_poses_gt[:, 0:3, 3] - target)
            target_gt_t_hat = torch.nn.functional.normalize(target_gt_t, p=2, dim=1).unsqueeze(2)
            t_target_error = (cumul_delta_t.unsqueeze(1).bmm(target_gt_t_hat)).view(-1)
        return t_error, t_target_error

    def rotation_quat_product(self, rot_quat, rot_quat_gt):
        scalar_product = rot_quat.unsqueeze(1).bmm(rot_quat_gt.unsqueeze(2)).view(-1)
        r_errors = 1 - scalar_product
        return r_errors

    def rtvec_to_pose(self, rtvec):
        return rtvec_to_pose(rtvec)

