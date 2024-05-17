import numpy as np
import torch
from attacks.attack import Attack
import time
from tqdm import tqdm
import cv2


# based on APGD paper: https://arxiv.org/pdf/2003.01690.pdf
class OurAPGD(Attack):
    def __init__(
            self,
            model,
            criterion,
            test_criterion,
            data_shape,
            norm='Linf',
            n_iter=20,
            n_restarts=1,
            alpha=None,
            prev_update_ratio=0.75,   # alpha in paper (page 3 equation (2)), regulates the influence of the previous update on the current one
            rho=0.75,                 # rho in paper (page 3, bottom-left column), see condition 1
            rand_init=False,
            sample_window_size=None,
            sample_window_stride=None,
            pert_padding=(0, 0),
            init_pert_path=None,
            init_pert_transform=None):
        super(OurAPGD, self).__init__(model, criterion, test_criterion, norm, data_shape,
                                  sample_window_size, sample_window_stride,
                                  pert_padding)

        self.alpha = alpha
        self.prev_update_ratio = prev_update_ratio
        self.counter = 0    # computes in how many cases the update step has been successful for condition 1 in the paper (page 3, bottom-left column)
        self.rho = rho
        self.step_size_at_prev_cp = None  # step size at previous checkpoint (for condition 2)
        self.best_loss_at_prev_cp = None  # best loss at previous checkpoint (for condition 2)

        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.checkpoints = self.calculate_checkpoints(self.n_iter)

        self.rand_init = rand_init

        self.init_pert = None
        if init_pert_path is not None:
            self.init_pert = cv2.cvtColor(cv2.imread(init_pert_path), cv2.COLOR_BGR2RGB)
            if init_pert_transform is None:
                self.init_pert = torch.tensor(self.init_pert).unsqueeze(0)
            else:
                self.init_pert = init_pert_transform({'img': self.init_pert})['img'].unsqueeze(0)

    def calc_sample_grad_single(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                         scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
        pert = pert.detach()
        pert.requires_grad_()
        img1_adv, img2_adv, output_adv = self.perturb_model_single(pert, img1_I0, img2_I0,
                                                            intrinsic_I0,
                                                            img1_delta, img2_delta,
                                                            scale,
                                                            mask1, mask2,
                                                            perspective1,
                                                            perspective2,
                                                            device)
        loss = self.criterion(output_adv, scale.to(device), y.to(device), target_pose.to(device), clean_flow.to(device))
        loss_sum = loss.sum(dim=0)
        grad = torch.autograd.grad(loss_sum, [pert])[0].detach()

        del img1_adv
        del img2_adv
        del output_adv
        del loss
        del loss_sum
        torch.cuda.empty_cache()

        return grad

    def calc_sample_grad_split(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                         scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
        sample_data_ind = list(range(img1_I0.shape[0] + 1))
        window_start_list = sample_data_ind[0::self.sample_window_stride]
        window_end_list = sample_data_ind[self.sample_window_size::self.sample_window_stride]

        if window_end_list[-1] != sample_data_ind[-1]:
            window_end_list.append(sample_data_ind[-1])
        grad = torch.zeros_like(pert, requires_grad=False)
        grad_multiplicity = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)

        for window_idx, window_end in enumerate(window_end_list):
            window_start = window_start_list[window_idx]
            grad_multiplicity[window_start:window_end] += 1

            pert_window = pert[window_start:window_end].clone().detach()
            img1_I0_window = img1_I0[window_start:window_end].clone().detach()
            img2_I0_window = img2_I0[window_start:window_end].clone().detach()
            intrinsic_I0_window = intrinsic_I0[window_start:window_end].clone().detach()
            img1_delta_window = img1_delta[window_start:window_end].clone().detach()
            img2_delta_window = img2_delta[window_start:window_end].clone().detach()
            scale_window = scale[window_start:window_end].clone().detach()
            y_window = y[window_start:window_end].clone().detach()
            clean_flow_window = clean_flow[window_start:window_end].clone().detach()
            target_pose_window = target_pose.clone().detach()
            perspective1_window = perspective1[window_start:window_end].clone().detach()
            perspective2_window = perspective2[window_start:window_end].clone().detach()
            mask1_window = mask1[window_start:window_end].clone().detach()
            mask2_window = mask2[window_start:window_end].clone().detach()

            grad_window = self.calc_sample_grad_single(pert_window,
                                                     img1_I0_window,
                                                     img2_I0_window,
                                                     intrinsic_I0_window,
                                                     img1_delta_window,
                                                     img2_delta_window,
                                                     scale_window,
                                                     y_window,
                                                     clean_flow_window,
                                                     target_pose_window,
                                                     perspective1_window,
                                                     perspective2_window,
                                                     mask1_window,
                                                     mask2_window,
                                                     device=device)
            with torch.no_grad():
                grad[window_start:window_end] += grad_window

            del grad_window
            del pert_window
            del img1_I0_window
            del img2_I0_window
            del intrinsic_I0_window
            del scale_window
            del y_window
            del clean_flow_window
            del target_pose_window
            del perspective1_window
            del perspective2_window
            del mask1_window
            del mask2_window
            torch.cuda.empty_cache()
        grad_multiplicity_expand = grad_multiplicity.view(-1, 1, 1, 1).expand(grad.shape)
        grad = grad / grad_multiplicity_expand
        del grad_multiplicity
        del grad_multiplicity_expand
        torch.cuda.empty_cache()
        return grad.to(device)

    def perturb(self, data_loader, y_list, eps,
                                   targeted=False, device=None, eval_data_loader=None, eval_y_list=None):

        a_abs = np.abs(eps / self.n_iter) if self.alpha is None else np.abs(self.alpha)
        multiplier = -1 if targeted else 1
        print("computing APGD attack with parameters:")
        print("attack random restarts: " + str(self.n_restarts))
        print("attack epochs: " + str(self.n_iter))
        print("attack norm: " + str(self.norm))
        print("attack epsilon norm limitation: " + str(eps))
        print("attack step size: " + str(a_abs))

        data_shape, dtype, eval_data_loader, eval_y_list, clean_flow_list, \
        eval_clean_loss_list, traj_clean_loss_mean_list, clean_loss_sum, \
        best_pert, best_loss_list, best_loss_sum, all_loss, all_best_loss = \
            self.compute_clean_baseline(data_loader, y_list, eval_data_loader, eval_y_list, device=device)

        for rest in tqdm(range(self.n_restarts)):
            self.counter = 0
            print("restarting attack optimization, restart number: " + str(rest))
            opt_start_time = time.time()

            pert = torch.zeros_like(best_pert)

            if self.init_pert is not None:
                print(" perturbation initialized from provided image")
                pert = self.init_pert.to(best_pert)
            elif self.rand_init:
                print(" perturbation initialized randomly")
                pert = self.random_initialization(pert, eps)
            else:
                print(" perturbation initialized to zero")

            # Perform first iteration (see `Algorithm1` in paper, page 2, upper-right column)
            pert = self.project(pert, eps)
            eval_loss_tot0, eval_loss_list0 = self.attack_eval(pert, data_shape, eval_data_loader, eval_y_list,
                                                               device)
            prev_pert = pert.clone().detach()
            pert = self.gradient_ascent_step(pert, data_shape, data_loader, y_list, clean_flow_list,
                                               multiplier, a_abs, eps, device=device)
            eval_loss_tot1, eval_loss_list1 = self.attack_eval(pert, data_shape, eval_data_loader, eval_y_list,
                                                               device)
            if eval_loss_tot0 < eval_loss_tot1:
                best_loss_sum = eval_loss_tot1
                best_loss_list = eval_loss_list1
                best_pert = pert.clone().detach()
                self.counter += 1
            else:
                best_loss_sum = eval_loss_tot0
                best_loss_list = eval_loss_list0
                best_pert = prev_pert.clone().detach()

            for k in tqdm(range(1, self.n_iter)):
                print(" attack optimization epoch: " + str(k))
                iter_start_time = time.time()

                if k in self.checkpoints:
                    # 0) find checkpoint index in checkpoints array
                    checkpointIdx = np.where(np.array(self.checkpoints) == k)[0][0]

                    # 1) check condition 1
                    condition_1 = False
                    if self.counter < self.rho * (k - self.checkpoints[checkpointIdx-1]):
                        condition_1 = True

                    # 2) Reset the counter that computes in how many cases the update step has been successful for condition 1
                    self.counter = 0

                    # 3) check condition 2
                    condition_2 = False
                    if (a_abs == self.step_size_at_prev_cp) and (best_loss_sum == self.best_loss_at_prev_cp):
                        condition_2 = True

                    # 4) save the best loss at current checkpoint
                    self.best_loss_at_prev_cp = best_loss_sum

                    # 4) update step size if needed and save it
                    if condition_1 or condition_2:
                        a_abs /= 2
                    else:
                        a_abs *= 1.1
                    self.step_size_at_prev_cp = a_abs

                # Equation (2) in paper. See page 3.
                curr_pert = pert.clone().detach()
                z_pert = self.gradient_ascent_step(pert, data_shape, data_loader, y_list, clean_flow_list,
                                                   multiplier, a_abs, eps, device=device)
                pert = self.project(pert + self.prev_update_ratio * (z_pert - pert) +
                                    (1 - self.prev_update_ratio) * (pert - prev_pert), eps)
                prev_pert = curr_pert.clone().detach()

                step_runtime = time.time() - iter_start_time
                print(" optimization epoch finished, epoch runtime: " + str(step_runtime))

                print(" evaluating perturbation")
                eval_start_time = time.time()

                with torch.no_grad():
                    eval_loss_tot, eval_loss_list = self.attack_eval(pert, data_shape, eval_data_loader, eval_y_list,
                                                                     device)

                    if eval_loss_tot > best_loss_sum:
                        self.counter += 1
                        best_pert = pert.clone().detach()
                        best_loss_list = eval_loss_list
                        best_loss_sum = eval_loss_tot
                    all_loss.append(eval_loss_list)
                    all_best_loss.append(best_loss_list)
                    traj_loss_mean_list = np.mean(eval_loss_list, axis=0)
                    traj_best_loss_mean_list = np.mean(best_loss_list, axis=0)

                    eval_runtime = time.time() - eval_start_time
                    print(" evaluation finished, evaluation runtime: " + str(eval_runtime))
                    print(" current trajectories loss mean list:")
                    print(" " + str(traj_loss_mean_list))
                    print(" current trajectories best loss mean list:")
                    print(" " + str(traj_best_loss_mean_list))
                    print(" trajectories clean loss mean list:")
                    print(" " + str(traj_clean_loss_mean_list))
                    print(" current trajectories loss sum:")
                    print(" " + str(eval_loss_tot))
                    print(" current trajectories best loss sum:")
                    print(" " + str(best_loss_sum))
                    print(" trajectories clean loss sum:")
                    print(" " + str(clean_loss_sum))
                    del eval_loss_tot
                    del eval_loss_list
                    torch.cuda.empty_cache()

            opt_runtime = time.time() - opt_start_time
            print("optimization restart finished, optimization runtime: " + str(opt_runtime))
        return best_pert.detach(), eval_clean_loss_list, all_loss, all_best_loss

    # ########################
    # auxilary functions
    # ########################
    @staticmethod
    def calculate_checkpoints(N_iter: int) -> list:
        """
        Calculates the iteration checkpoints for testing if the step size needs to be halved.
        This can be found in the paper in section 3.1, page 3, upper-right column.
        """
        checkpoints = []
        p_j_minus_1 = 0
        p_j = 0.22
        w = int(np.ceil(p_j * N_iter))
        checkpoints.append(p_j_minus_1*N_iter)
        while w < N_iter:
            checkpoints.append(w)
            p = p_j + max(p_j - p_j_minus_1 - 0.03, 0.06)
            p_j_minus_1 = p_j
            p_j = p
            w = int(np.ceil(p_j * N_iter))

        return checkpoints
