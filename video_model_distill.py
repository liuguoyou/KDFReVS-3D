import os
import math
import glob
import torch
import torch.nn as nn
import torchvision
import networks
import utils
from renderer import Renderer
import warnings
import torch.nn.functional as F
import soft_renderer as sr
from arcface.arcface import ArcFace
import cv2

warnings.filterwarnings("ignore")
EPS = 1e-7


class Videosup3D_LSTM_GRU():
    def __init__(self, cfgs):
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.device = cfgs.get('device', 'cpu')
        self.image_size = cfgs.get('image_size', 64)
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        self.border_depth = cfgs.get('border_depth', (0.7 * self.max_depth + 0.3 * self.min_depth))
        self.xyz_rotation_range = cfgs.get('xyz_rotation_range', 60)
        self.xy_translation_range = cfgs.get('xy_translation_range', 0.1)
        self.z_translation_range = cfgs.get('z_translation_range', 0.1)
        self.lam_perc = cfgs.get('lam_perc', 1)
        self.lam_flip = cfgs.get('lam_flip', 0.5)
        self.lam_flip_start_epoch = cfgs.get('lam_flip_start_epoch', 0)
        self.lr = cfgs.get('lr', 1e-4)
        self.load_gt_depth = cfgs.get('load_gt_depth', False)
        self.renderer = Renderer(cfgs)
        self.LoR_K = cfgs.get("LoR_K", 64)
        self.beta1 = cfgs.get("beta1", 0.9)
        self.beta2 = cfgs.get("beta2", 0.1)
        self.LoR_K = cfgs.get("LoR_K", 64)

        print("use beta1 {}, beta2:{}".format(self.beta1, self.beta2))
        self.use_perceptual_loss = cfgs.get("perceptual_loss", True)
        self.use_geometry_loss = cfgs.get("geometry_loss", True)
        self.use_distill_loss = cfgs.get("distill_loss", True)
        self.use_identity_loss = cfgs.get("identity_loss", True)
        self.use_one_albedo = cfgs.get("use_one_albedo", True)
        self.distill_loss_type = cfgs.get("distill_loss_type", "KL")
        self.detach_distill_teacher_grad = cfgs.get("detach_distill_teacher_grad", False)
        self.use_neutral_loss_pixel = cfgs.get("use_neutral_loss_pixel", False)
        self.use_neutral_loss_percep = cfgs.get("use_neutral_loss_percep", False)
        self.neutral_loss_weight = cfgs.get("neutral_loss_weight", 0.4)

        self.forward_type = cfgs.get("forward_type", "rnn")
        self.fusion_type = cfgs.get("fusion_type", "mean")

        ## networks and optimizers
        self.net_rnn = networks.TemporalNet(self.LoR_K * 3, self.LoR_K * 3, 2)

        self.netGeometryModel = networks.FaceModelNet(self.LoR_K, cout=1)
        self.netExpressionModel = networks.FaceModelNet(self.LoR_K, cout=1)
        self.netAlbedoModel = networks.FaceModelNet(self.LoR_K, cout=3)

        self.netEncoding_teacher = networks.FaceEmbeddingNet(cin=3, out_dim=self.LoR_K * 3, nf=64)
        self.netEncoding_student = networks.FaceEmbeddingNet(cin=3, out_dim=self.LoR_K * 3, nf=64)

        self.netFCGeo = networks.DecompositionNet(self.LoR_K * 3, self.LoR_K)
        self.netFCExp = networks.DecompositionNet(self.LoR_K * 3, self.LoR_K)
        self.netFCAlb = networks.DecompositionNet(self.LoR_K * 3, self.LoR_K)

        # albedo
        self.netL = networks.Encoder(cin=3, cout=4, nf=32)  # light
        self.netV = networks.Encoder(cin=3, cout=6, nf=32)  # view
        self.netC = networks.ConfNet(cin=3, cout=2, nf=64, zdim=128)
        # self.network_names = list(set([k for k in vars(self) if 'net' in k]) - set(["netAlbedoModel", "netCoeffA"]))
        self.network_names = [k for k in vars(self) if 'net' in k]
        # self.fixed_network_names = ["netAlbedoModel", "netCoeffA"]
        self.make_optimizer = lambda model: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        self.arcface = ArcFace()
        ## other parameters
        self.PerceptualLoss = networks.PerceptualLoss(requires_grad=False)
        self.other_param_names = ['PerceptualLoss']

        ## depth rescaler: -1~1 -> min_deph~max_deph
        self.depth_rescaler = lambda d: (1 + d) / 2 * self.max_depth + (1 - d) / 2 * self.min_depth
        self.lap_loss = None

    def init_optimizers(self):
        self.optimizer_names = []
        for net_name in self.network_names:
            optimizer = self.make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace('net', 'optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]

    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names:
                getattr(self, k).load_state_dict(cp[k])

            '''
            if k and k in self.fixed_network_names:
                getattr(self, k).load_state_dict(cp[k])
            '''

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names:
                getattr(self, k).load_state_dict(cp[k])

    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        '''
        for net_name in self.fixed_network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        '''
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def to_device(self, device):
        self.device = device
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))
        '''
        for net_name in self.fixed_network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))
        '''
        if self.other_param_names:
            for param_name in self.other_param_names:
                setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()
        '''
        for net_name in self.fixed_network_names:
            getattr(self, net_name).eval()
        '''

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def photometric_loss(self, im1, im2, mask=None, conf_sigma=None):
        loss = (im1 - im2).abs()
        if conf_sigma is not None:
            loss = loss * 2 ** 0.5 / (conf_sigma + EPS) + (conf_sigma + EPS).log()
        if mask is not None:
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def KL_loss(self, p_logit, q_logit):
        p = F.softmax(p_logit, dim=-1)
        _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                             - F.log_softmax(q_logit, dim=-1)), 1)
        return torch.mean(_kl)

    def MSELoss(self, p, q):
        cri = nn.MSELoss()
        return cri(p, q)

    def L1Loss(self, p, q):
        cri = nn.L1Loss()
        return cri(p, q)

    def SmoothL1Loss(self, p, q):
        cri = nn.SmoothL1Loss()
        return cri(p, q)

    def CosineLoss(self, p, q):
        lbl = torch.ones(len(p)).to(p.device)
        cri = nn.CosineEmbeddingLoss()
        return cri(p, q, lbl)

    def cal_distill_loss(self, p, q, loss_type):
        if loss_type == "KL":
            return self.KL_loss(p, q)
        elif loss_type == "MSE":
            return self.MSELoss(p, q)
        elif loss_type == "L1":
            return self.L1Loss(p, q)
        elif loss_type == "SmoL1":
            return self.SmoothL1Loss(p, q)
        elif loss_type == "Cosine":
            return self.CosineLoss(p, q)
        else:
            assert False, "unexpected distill loss type!"

    def backward(self):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).zero_grad()
        self.loss_total.backward()
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).step()

    def process_depth(self, depth, b, h, w):
        depth = depth - depth.view(b, -1).mean(1).view(b, 1, 1)
        depth = depth.tanh()
        depth = self.depth_rescaler(depth)

        ## clamp border depth
        depth_border = torch.zeros(1, h, w - 4).to(self.input_im.device)
        depth_border = nn.functional.pad(depth_border, (2, 2), mode='constant', value=1)
        depth = depth * (1 - depth_border) + depth_border * self.border_depth
        return depth

    def get_diffuse_shading_from_depth(self, depth, canon_light, b):

        canon_normal = self.renderer.get_normal_from_depth(depth)
        canon_diffuse_shading = (canon_normal * canon_light.view(-1, 1, 1, 3)).sum(3).clamp(
            min=0).unsqueeze(1)
        return canon_diffuse_shading

    def compute_neutral(self, inputs, out_lights, expression_coeff, albedo_coeff, single_identity_coeff, out_poses):
        b, c, h, w = inputs.shape
        vn = b
        canon_light = out_lights.repeat(2, 1)  # Bx4
        # print(canon_light)
        canon_light_a = canon_light[:, :1] / 2 + 0.5  # ambience term
        canon_light_b = canon_light[:, 1:2] / 2 + 0.5  # diffuse term
        canon_light_dxy = canon_light[:, 2:]
        canon_light_d = torch.cat([canon_light_dxy, torch.ones(vn * 2, 1).to(inputs.device)], 1)
        canon_light_d = canon_light_d / ((canon_light_d ** 2).sum(1, keepdim=True)) ** 0.5  # diffuse light direction

        conf_sigma_l1, conf_sigma_percl = self.netC(inputs)

        # print(self.beta1, self.beta2)
        canon_depth_video = self.beta1 * self.netGeometryModel(
            single_identity_coeff)

        canon_depth_video = canon_depth_video.squeeze(1)

        canon_depth_video = canon_depth_video - canon_depth_video.view(vn, -1).mean(1).view(vn, 1, 1)
        canon_depth_video = canon_depth_video.tanh()
        canon_depth_video = self.depth_rescaler(canon_depth_video)

        ## clamp border depth
        depth_border = torch.zeros(1, h, w - 4).to(inputs.device)
        depth_border = nn.functional.pad(depth_border, (2, 2), mode='constant', value=1)
        canon_depth_video = canon_depth_video * (1 - depth_border) + depth_border * self.border_depth
        canon_depth_video = torch.cat([canon_depth_video, canon_depth_video.flip(2)], 0)

        canon_depth = canon_depth_video

        canon_albed_video = self.netAlbedoModel(albedo_coeff)
        canon_albed_video = torch.cat([canon_albed_video, canon_albed_video.flip(3)], dim=0)
        canon_albedo = canon_albed_video
        ## shading
        canon_normal = self.renderer.get_normal_from_depth(canon_depth_video)
        canon_diffuse_shading = (canon_normal * canon_light_d.view(-1, 1, 1, 3)).sum(3).clamp(min=0).unsqueeze(1)
        canon_shading = canon_light_a.view(-1, 1, 1, 1) + canon_light_b.view(-1, 1, 1, 1) * canon_diffuse_shading

        canon_im = (canon_albed_video / 2 + 0.5) * canon_shading * 2 - 1

        ## predict viewpoint transformation
        view = out_poses.repeat(2, 1)
        view = torch.cat([
            view[:, :3] * math.pi / 180 * self.xyz_rotation_range,
            view[:, 3:5] * self.xy_translation_range,
            view[:, 5:] * self.z_translation_range], 1)

        ## reconstruct input view
        self.renderer.set_transform_matrices(view)

        vertices, faces = self.renderer.get_mesh_reconstuction(canon_depth_video)

        recon_depth = self.renderer.warp_canon_depth(canon_depth_video)
        recon_normal = self.renderer.get_normal_from_depth(recon_depth)
        grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(recon_depth)
        recon_im = nn.functional.grid_sample(canon_im, grid_2d_from_canon, mode='bilinear')

        margin = (self.max_depth - self.min_depth) / 2
        recon_im_mask = (
                recon_depth < self.max_depth + margin).float()  # invalid border pixels have been clamped at max_depth+margin
        recon_im_mask_both = recon_im_mask[:vn] * recon_im_mask[vn:]  # both original and flip reconstruction
        recon_im_mask_both = recon_im_mask_both.repeat(2, 1, 1).unsqueeze(1).detach()
        recon_im = recon_im * recon_im_mask_both

        return recon_im, recon_im_mask_both, conf_sigma_l1, conf_sigma_percl, vertices, faces

    def common_compute(self, inputs, out_lights, expression_coeff, albedo_coeff, single_identity_coeff, out_poses):
        b, c, h, w = inputs.shape
        vn = b
        canon_light = out_lights.repeat(2, 1)  # Bx4
        # print(canon_light)
        canon_light_a = canon_light[:, :1] / 2 + 0.5  # ambience term
        canon_light_b = canon_light[:, 1:2] / 2 + 0.5  # diffuse term
        canon_light_dxy = canon_light[:, 2:]
        canon_light_d = torch.cat([canon_light_dxy, torch.ones(vn * 2, 1).to(inputs.device)], 1)
        canon_light_d = canon_light_d / ((canon_light_d ** 2).sum(1, keepdim=True)) ** 0.5  # diffuse light direction

        conf_sigma_l1, conf_sigma_percl = self.netC(inputs)

        # print(self.beta1, self.beta2)
        canon_depth_video = self.beta1 * self.netGeometryModel(
            single_identity_coeff) + self.beta2 * self.netExpressionModel(
            expression_coeff)

        canon_depth_video = canon_depth_video.squeeze(1)

        canon_depth_video = canon_depth_video - canon_depth_video.view(vn, -1).mean(1).view(vn, 1, 1)
        canon_depth_video = canon_depth_video.tanh()
        canon_depth_video = self.depth_rescaler(canon_depth_video)

        ## clamp border depth
        depth_border = torch.zeros(1, h, w - 4).to(inputs.device)
        depth_border = nn.functional.pad(depth_border, (2, 2), mode='constant', value=1)
        canon_depth_video = canon_depth_video * (1 - depth_border) + depth_border * self.border_depth
        canon_depth_video = torch.cat([canon_depth_video, canon_depth_video.flip(2)], 0)

        canon_depth = canon_depth_video

        canon_albed_video = self.netAlbedoModel(albedo_coeff)
        canon_albed_video = torch.cat([canon_albed_video, canon_albed_video.flip(3)], dim=0)
        canon_albedo = canon_albed_video
        ## shading
        canon_normal = self.renderer.get_normal_from_depth(canon_depth_video)
        canon_diffuse_shading = (canon_normal * canon_light_d.view(-1, 1, 1, 3)).sum(3).clamp(min=0).unsqueeze(1)
        canon_shading = canon_light_a.view(-1, 1, 1, 1) + canon_light_b.view(-1, 1, 1, 1) * canon_diffuse_shading

        canon_im = (canon_albed_video / 2 + 0.5) * canon_shading * 2 - 1

        ## predict viewpoint transformation
        view = out_poses.repeat(2, 1)
        view = torch.cat([
            view[:, :3] * math.pi / 180 * self.xyz_rotation_range,
            view[:, 3:5] * self.xy_translation_range,
            view[:, 5:] * self.z_translation_range], 1)

        ## reconstruct input view
        self.renderer.set_transform_matrices(view)

        vertices, faces = self.renderer.get_mesh_reconstuction(canon_depth_video)

        recon_depth = self.renderer.warp_canon_depth(canon_depth_video)
        recon_normal = self.renderer.get_normal_from_depth(recon_depth)
        grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(recon_depth)
        recon_im = nn.functional.grid_sample(canon_im, grid_2d_from_canon, mode='bilinear')

        margin = (self.max_depth - self.min_depth) / 2
        recon_im_mask = (
                recon_depth < self.max_depth + margin).float()  # invalid border pixels have been clamped at max_depth+margin
        recon_im_mask_both = recon_im_mask[:vn] * recon_im_mask[vn:]  # both original and flip reconstruction
        recon_im_mask_both = recon_im_mask_both.repeat(2, 1, 1).unsqueeze(1).detach()
        recon_im = recon_im * recon_im_mask_both

        return recon_im, recon_im_mask_both, conf_sigma_l1, conf_sigma_percl, vertices, faces

    def feature_fusion(self, features, fusion_type="mean"):
        # features [batch, seq, dim]
        # return [batch, dim]
        if fusion_type == 'mean':
            return features.mean(dim=1)
        else:
            raise ValueError("Not Implement yet for {}".format(fusion_type))

    def forward_train(self, input):
        self.input_im = input.to(self.device) * 2. - 1.
        b, num_views, c, h, w = input.shape

        rich_codes = []
        poses, lights = [], []
        for i in range(b):
            rich_codes.append(self.netEncoding_teacher(self.input_im[i]).unsqueeze(0))
            poses.append(self.netV(self.input_im[i]).unsqueeze(0))
            lights.append(self.netL(self.input_im[i]).unsqueeze(0))

        rich_codes = torch.cat(rich_codes)  # [batch, seq, dim]
        out_poses = torch.cat(poses)  # [batch, seq, dim]
        out_lights = torch.cat(lights)  # [batch, seq, dim]

        if self.forward_type == 'rnn':
            out_rich_codes, _ = self.net_rnn(rich_codes.transpose(0, 1))  # [seq, batch, dim]
        elif self.forward_type == 'fusion':
            out_rich_codes = self.feature_fusion(rich_codes).unsqueeze(0).repeat((num_views, 1, 1))
        else:
            raise ValueError("forward_type only support for rnn and fusion")

        final_codes = out_rich_codes[-1, :, :].repeat((num_views, 1, 1)).transpose(0, 1)
        out_rich_codes = out_rich_codes.transpose(0, 1)

        out_geo_coeffs, out_exp_coeffs, out_alb_coeffs = [], [], []
        for i in range(b):
            out_geo_coeffs.append(self.netFCGeo(final_codes[i]).unsqueeze(0))
            if self.use_one_albedo:
                out_alb_coeffs.append(self.netFCAlb(final_codes[i]).unsqueeze(0))
            else:
                out_alb_coeffs.append(self.netFCAlb(out_rich_codes[i]).unsqueeze(0))
            out_exp_coeffs.append(self.netFCExp(out_rich_codes[i]).unsqueeze(0))

        out_geo_coeffs = torch.cat(out_geo_coeffs)
        out_alb_coeffs = torch.cat(out_alb_coeffs)
        out_exp_coeffs = torch.cat(out_exp_coeffs)

        lam_flip = 1 if self.trainer.current_epoch < self.lam_flip_start_epoch else self.lam_flip
        ## teacher

        use_neutral_loss = self.use_neutral_loss_pixel or self.use_neutral_loss_percep
        for i in range(b):
            vn = num_views
            recon_im, recon_im_mask_both, conf_sigma_l1, conf_sigma_percl, vertices, faces = self.common_compute(
                self.input_im[i], out_lights[i], out_exp_coeffs[i], out_alb_coeffs[i], out_geo_coeffs[i], out_poses[i])

            if use_neutral_loss:
                recon_im_neutral, _, _, _, _, _ = self.compute_neutral(self.input_im[i], out_lights[i], out_exp_coeffs[i],
                                                                   out_alb_coeffs[i], out_geo_coeffs[i], out_poses[i])

            ## loss function
            loss_l1_im = self.photometric_loss(recon_im[:vn], self.input_im[i], mask=recon_im_mask_both[:vn],
                                               conf_sigma=conf_sigma_l1[:, :1])
            loss_l1_im_flip = self.photometric_loss(recon_im[vn:], self.input_im[i],
                                                    mask=recon_im_mask_both[vn:],
                                                    conf_sigma=conf_sigma_l1[:, 1:])
            loss_perc_im = self.PerceptualLoss(recon_im[:vn], self.input_im[i], mask=recon_im_mask_both[:vn],
                                               conf_sigma=conf_sigma_percl[:, :1])
            loss_perc_im_flip = self.PerceptualLoss(recon_im[vn:], self.input_im[i],
                                                    mask=recon_im_mask_both[vn:],
                                                    conf_sigma=conf_sigma_percl[:, 1:])
            neutral_loss = 0
            if use_neutral_loss:
                loss_l1_im_neutral = self.photometric_loss(recon_im_neutral[:vn], self.input_im[i], mask=recon_im_mask_both[:vn],
                                                   conf_sigma=conf_sigma_l1[:, :1])
                loss_l1_im_flip_neutral = self.photometric_loss(recon_im_neutral[vn:], self.input_im[i],
                                                        mask=recon_im_mask_both[vn:],
                                                        conf_sigma=conf_sigma_l1[:, 1:])
                loss_perc_im_neutral = self.PerceptualLoss(recon_im_neutral[:vn], self.input_im[i], mask=recon_im_mask_both[:vn],
                                                   conf_sigma=conf_sigma_percl[:, :1])
                loss_perc_im_flip_neutral = self.PerceptualLoss(recon_im_neutral[vn:], self.input_im[i],
                                                        mask=recon_im_mask_both[vn:],
                                                        conf_sigma=conf_sigma_percl[:, 1:])

                if self.use_neutral_loss_pixel:
                    neutral_loss += loss_l1_im_neutral + lam_flip * loss_l1_im_flip_neutral
                if self.use_neutral_loss_percep:
                    neutral_loss += loss_perc_im_neutral + lam_flip * loss_perc_im_flip_neutral

            if self.lap_loss is None:
                self.lap_loss = sr.LaplacianLoss(vertices.detach().cpu()[0], faces.detach().cpu()[0]).cuda()

            if i == 0:
                self.teacher_loss = loss_l1_im + lam_flip * loss_l1_im_flip
            else:
                self.teacher_loss += loss_l1_im + lam_flip * loss_l1_im_flip

            if self.use_perceptual_loss:
                self.teacher_loss += self.lam_perc * (loss_perc_im + lam_flip * loss_perc_im_flip)
            if self.use_geometry_loss:
                laploss = torch.sum(self.lap_loss(vertices))
                self.teacher_loss += laploss
            if self.use_identity_loss:
                self.teacher_loss += self.arcface.arc_loss(self.input_im[i], recon_im[:vn],
                                                           torch.ones(vn).float().to(self.input_im.device))
            if use_neutral_loss:
                self.teacher_loss += self.neutral_loss_weight * neutral_loss

        self.teacher_loss = self.teacher_loss / b

        ## student
        idxs = torch.randint(0, num_views, size=(b,))
        student_inputs, student_light, student_pose = [], [], []
        for i in range(b):
            student_inputs.append(self.input_im[i, idxs[i]])
            student_light.append(out_lights[i, idxs[i]])
            student_pose.append(out_poses[i, idxs[i]])

        student_light = torch.stack(student_light)
        student_pose = torch.stack(student_pose)
        student_inputs = torch.stack(student_inputs)
        student_coding = self.netEncoding_student(student_inputs)

        student_alb = self.netFCAlb(student_coding)
        student_geo = self.netFCGeo(student_coding)
        student_exp = self.netFCExp(student_coding)

        teacher_geo = out_geo_coeffs[:, 0, :]
        teacher_exp = []
        if self.use_one_albedo:
            teacher_alb = out_alb_coeffs[:, 0, :]
        else:
            teacher_alb = []
            for i in range(b):
                teacher_alb.append(out_alb_coeffs[i, idxs[i]])
            teacher_alb = torch.stack(teacher_alb)

        for i in range(b):
            teacher_exp.append(out_exp_coeffs[i, idxs[i]])
        teacher_exp = torch.stack(teacher_exp)

        recon_im, recon_im_mask_both, conf_sigma_l1, conf_sigma_percl, vertices, faces = self.common_compute(
            student_inputs, student_light, student_exp, student_alb, student_geo, student_pose)

        loss_l1_im = self.photometric_loss(recon_im[:b], student_inputs, mask=recon_im_mask_both[:b],
                                           conf_sigma=conf_sigma_l1[:, :1])
        loss_l1_im_flip = self.photometric_loss(recon_im[b:], student_inputs,
                                                mask=recon_im_mask_both[b:],
                                                conf_sigma=conf_sigma_l1[:, 1:])
        loss_perc_im = self.PerceptualLoss(recon_im[:b], student_inputs, mask=recon_im_mask_both[:b],
                                           conf_sigma=conf_sigma_percl[:, :1])
        loss_perc_im_flip = self.PerceptualLoss(recon_im[b:], student_inputs,
                                                mask=recon_im_mask_both[b:],
                                                conf_sigma=conf_sigma_percl[:, 1:])
        laploss = torch.sum(self.lap_loss(vertices))
        self.student_loss = loss_l1_im + lam_flip * loss_l1_im_flip
        if self.use_perceptual_loss:
            self.student_loss += self.lam_perc * (loss_perc_im + lam_flip * loss_perc_im_flip)
        if self.use_geometry_loss:
            self.student_loss += laploss
        if self.use_identity_loss:
            self.student_loss += self.arcface.arc_loss(student_inputs, recon_im[:b],
                                                        torch.ones(b).float().to(self.input_im.device))

        if self.detach_distill_teacher_grad:
            teacher_alb = teacher_alb.detach()
            teacher_geo = teacher_geo.detach()
            teacher_exp = teacher_exp.detach()

        self.distill_loss = self.cal_distill_loss(teacher_alb, student_alb,
                                                  self.distill_loss_type) + self.cal_distill_loss(teacher_geo,
                                                                                                  student_geo,
                                                                                                  self.distill_loss_type) + self.cal_distill_loss(
            teacher_exp, student_exp, self.distill_loss_type)

        if self.use_distill_loss:
            self.loss_total = self.teacher_loss + self.student_loss + self.distill_loss
        else:
            self.loss_total = self.teacher_loss + self.student_loss

        print("teacher:{:.6f}, student:{:.6f}, distill:{:.6f}".format(self.teacher_loss, self.student_loss,
                                                                                       self.distill_loss,
                                                                                       ))
        metrics = {'loss': self.loss_total}

        return metrics

    def forward_valid(self, input):
        if self.load_gt_depth:
            input, depth_gt = input
        self.input_im = input.to(self.device) * 2. - 1.
        b, c, h, w = self.input_im.shape

        codes = self.netEncoding_student(self.input_im)

        identity_coeff = self.netFCGeo(codes)
        expression_coeff = self.netFCExp(codes)
        albedo_coeff = self.netFCAlb(codes)

        self.canon_middle_depth = self.netGeometryModel(identity_coeff)
        self.canon_delta_depth = self.netExpressionModel(expression_coeff)

        self.canon_depth_raw = self.beta1 * self.canon_middle_depth + self.beta2 * self.canon_delta_depth  # BxHxW
        self.canon_depth_raw = self.canon_depth_raw.squeeze(1)
        self.canon_depth = self.canon_depth_raw - self.canon_depth_raw.view(b, -1).mean(1).view(b, 1, 1)
        self.canon_depth = self.canon_depth.tanh()
        self.canon_depth = self.depth_rescaler(self.canon_depth)

        ## clamp border depth
        depth_border = torch.zeros(1, h, w - 4).to(self.input_im.device)
        depth_border = nn.functional.pad(depth_border, (2, 2), mode='constant', value=1)
        self.canon_depth = self.canon_depth * (1 - depth_border) + depth_border * self.border_depth
        self.canon_depth = torch.cat([self.canon_depth, self.canon_depth.flip(2)], 0)  # flip

        self.canon_middle_depth = self.process_depth(self.canon_middle_depth.squeeze(1), b, h, w)
        self.canon_middle_depth = torch.cat([self.canon_middle_depth, self.canon_middle_depth.flip(2)], 0)
        self.canon_delta_depth = self.process_depth(self.canon_delta_depth.squeeze(1), b, h, w)
        self.canon_delta_depth = torch.cat([self.canon_delta_depth, self.canon_delta_depth.flip(2)], 0)

        # vertices, faces = self.renderer.get_mesh_reconstuction(self.canon)
        ## predict canonical albedo
        # subAlbedo = torch.Tensor.repeat(self.SubAlbedo.unsqueeze(0), (b, 1, 1, 1, 1))
        # self.canon_albedo = torch.sum(albedo_coeff.unsqueeze(2).unsqueeze(3).unsqueeze(4) * subAlbedo, dim=1) # Bx3xHxW
        self.canon_albedo = self.netAlbedoModel(albedo_coeff)  # Bx3xHxW
        self.canon_albedo = torch.cat([self.canon_albedo, self.canon_albedo.flip(3)], 0)  # flip

        ## predict confidence map
        self.conf_sigma_l1, self.conf_sigma_percl = self.netC(self.input_im)  # Bx2xHxW

        ## predict lighting
        canon_light = self.netL(self.input_im).repeat(2, 1)  # Bx4
        self.canon_light_a = canon_light[:, :1] / 2 + 0.5  # ambience term
        self.canon_light_b = canon_light[:, 1:2] / 2 + 0.5  # diffuse term
        canon_light_dxy = canon_light[:, 2:]
        self.canon_light_d = torch.cat([canon_light_dxy, torch.ones(b * 2, 1).to(self.input_im.device)], 1)
        self.canon_light_d = self.canon_light_d / (
            (self.canon_light_d ** 2).sum(1, keepdim=True)) ** 0.5  # diffuse light direction

        ## shading
        self.canon_normal = self.renderer.get_normal_from_depth(self.canon_depth)
        self.canon_diffuse_shading = (self.canon_normal * self.canon_light_d.view(-1, 1, 1, 3)).sum(3).clamp(
            min=0).unsqueeze(1)
        self.canon_mid_diff_shading = self.get_diffuse_shading_from_depth(self.canon_middle_depth, self.canon_light_d,
                                                                          b)
        self.canon_delta_diff_shading = self.get_diffuse_shading_from_depth(self.canon_delta_depth, self.canon_light_d,
                                                                            b)

        canon_shading = self.canon_light_a.view(-1, 1, 1, 1) + self.canon_light_b.view(-1, 1, 1,
                                                                                       1) * self.canon_diffuse_shading

        self.canon_im = (self.canon_albedo / 2 + 0.5) * canon_shading * 2 - 1

        ## predict viewpoint transformation
        self.view = self.netV(self.input_im).repeat(2, 1)
        self.view = torch.cat([
            self.view[:, :3] * math.pi / 180 * self.xyz_rotation_range,
            self.view[:, 3:5] * self.xy_translation_range,
            self.view[:, 5:] * self.z_translation_range], 1)

        ## reconstruct input view
        self.renderer.set_transform_matrices(self.view)
        self.recon_depth = self.renderer.warp_canon_depth(self.canon_depth)
        self.recon_normal = self.renderer.get_normal_from_depth(self.recon_depth)
        grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(self.recon_depth)
        self.recon_im = nn.functional.grid_sample(self.canon_im, grid_2d_from_canon, mode='bilinear')

        margin = (self.max_depth - self.min_depth) / 2
        recon_im_mask = (
                self.recon_depth < self.max_depth + margin).float()  # invalid border pixels have been clamped at max_depth+margin
        recon_im_mask_both = recon_im_mask[:b] * recon_im_mask[b:]  # both original and flip reconstruction
        recon_im_mask_both = recon_im_mask_both.repeat(2, 1, 1).unsqueeze(1).detach()
        self.recon_im = self.recon_im * recon_im_mask_both

        ## render symmetry axis
        canon_sym_axis = torch.zeros(h, w).to(self.input_im.device)
        canon_sym_axis[:, w // 2 - 1:w // 2 + 1] = 1
        self.recon_sym_axis = nn.functional.grid_sample(canon_sym_axis.repeat(b * 2, 1, 1, 1), grid_2d_from_canon,
                                                        mode='bilinear')
        self.recon_sym_axis = self.recon_sym_axis * recon_im_mask_both
        green = torch.FloatTensor([-1, 1, -1]).to(self.input_im.device).view(1, 3, 1, 1)
        self.input_im_symline = (0.5 * self.recon_sym_axis) * green + (
                1 - 0.5 * self.recon_sym_axis) * self.input_im.repeat(2, 1, 1, 1)

        ## loss function
        self.loss_l1_im = self.photometric_loss(self.recon_im[:b], self.input_im, mask=recon_im_mask_both[:b],
                                                conf_sigma=self.conf_sigma_l1[:, :1])
        self.loss_l1_im_flip = self.photometric_loss(self.recon_im[b:], self.input_im, mask=recon_im_mask_both[b:],
                                                     conf_sigma=self.conf_sigma_l1[:, 1:])
        self.loss_perc_im = self.PerceptualLoss(self.recon_im[:b], self.input_im, mask=recon_im_mask_both[:b],
                                                conf_sigma=self.conf_sigma_percl[:, :1])
        self.loss_perc_im_flip = self.PerceptualLoss(self.recon_im[b:], self.input_im, mask=recon_im_mask_both[b:],
                                                     conf_sigma=self.conf_sigma_percl[:, 1:])

        lam_flip = 1
        self.loss_total = self.loss_l1_im + lam_flip * self.loss_l1_im_flip + self.lam_perc * (
                self.loss_perc_im + lam_flip * self.loss_perc_im_flip)

        metrics = {'loss': self.loss_total}
        if self.load_gt_depth:
            self.depth_gt = depth_gt[:, 0, :, :].to(self.input_im.device)
            self.depth_gt = (1 - self.depth_gt) * 2 - 1
            self.depth_gt = self.depth_rescaler(self.depth_gt)
            self.normal_gt = self.renderer.get_normal_from_depth(self.depth_gt)

            # mask out background
            mask_gt = (self.depth_gt < (self.depth_gt.max() - 0.11)).float()
            mask_gt = (nn.functional.avg_pool2d(mask_gt.unsqueeze(1), 3, stride=1, padding=1).squeeze(
                1) > 0.99).float()  # erode by 1 pixel
            mask_pred = (nn.functional.avg_pool2d(recon_im_mask[:b].unsqueeze(1), 3, stride=1, padding=1).squeeze(
                1) > 0.99).float()  # erode by 1 pixel
            mask = mask_gt * mask_pred
            self.acc_mae_masked = ((self.recon_depth[:b] - self.depth_gt[:b]).abs() * mask).view(b, -1).sum(
                1) / mask.view(b, -1).sum(1)
            self.acc_mse_masked = (((self.recon_depth[:b] - self.depth_gt[:b]) ** 2) * mask).view(b, -1).sum(
                1) / mask.view(b, -1).sum(1)
            self.sie_map_masked = utils.compute_sc_inv_err(self.recon_depth[:b].log(), self.depth_gt[:b].log(),
                                                           mask=mask)
            self.acc_sie_masked = (self.sie_map_masked.view(b, -1).sum(1) / mask.view(b, -1).sum(1)) ** 0.5
            self.norm_err_map_masked = utils.compute_angular_distance(self.recon_normal[:b], self.normal_gt[:b],
                                                                      mask=mask)
            self.acc_normal_masked = self.norm_err_map_masked.view(b, -1).sum(1) / mask.view(b, -1).sum(1)

            metrics['SIE_masked'] = self.acc_sie_masked.mean()
            metrics['NorErr_masked'] = self.acc_normal_masked.mean()
        return metrics

    def forward(self, input, valid=False):

        """Feedforward once.
            train : input: [batch_size, num_views, 3, 64, 64]
            valid : input: [batch_size, 1, 3, 64, 64]
        """
        if not valid:
            return self.forward_train(input)
        else:
            return self.forward_valid(input)

    def visualize(self, logger, total_iter, max_bs=25):
        '''
        b, num_views, c, h, w = self.input_im.shape
        b0 = min(max_bs, b)

        ## render rotations

        with torch.no_grad():
            v0 = torch.FloatTensor([-0.1 * math.pi / 180 * 60, 0, 0, 0, 0, 0]).to(self.input_im.device).repeat(b0, 1)
            canon_im_rotate = self.renderer.render_yaw(self.canon_im[:b0], self.canon_depth[:b0], v_before=v0,
                                                       maxr=90).detach().cpu() / 2. + 0.5  # (B,T,C,H,W)
            canon_normal_rotate = self.renderer.render_yaw(self.canon_normal[:b0].permute(0, 3, 1, 2),
                                                           self.canon_depth[:b0], v_before=v0,
                                                           maxr=90).detach().cpu() / 2. + 0.5  # (B,T,C,H,W)

        input_im = self.input_im[:b0].detach().cpu().numpy() / 2 + 0.5
        input_im_symline = self.input_im_symline[:b0].detach().cpu() / 2. + 0.5
        canon_albedo = self.canon_albedo[:b0].detach().cpu() / 2. + 0.5
        canon_im = self.canon_im[:b0].detach().cpu() / 2. + 0.5
        recon_im = self.recon_im[:b0].detach().cpu() / 2. + 0.5
        recon_im_flip = self.recon_im[b:b + b0].detach().cpu() / 2. + 0.5
        canon_depth_raw_hist = self.canon_depth_raw.detach().unsqueeze(1).cpu()
        canon_depth_raw = self.canon_depth_raw[:b0].detach().unsqueeze(1).cpu() / 2. + 0.5
        canon_depth = ((self.canon_depth[:b0] - self.min_depth) / (
                    self.max_depth - self.min_depth)).detach().cpu().unsqueeze(1)
        recon_depth = ((self.recon_depth[:b0] - self.min_depth) / (
                    self.max_depth - self.min_depth)).detach().cpu().unsqueeze(1)
        canon_diffuse_shading = self.canon_diffuse_shading[:b0].detach().cpu()
        canon_normal = self.canon_normal.permute(0, 3, 1, 2)[:b0].detach().cpu() / 2 + 0.5
        recon_normal = self.recon_normal.permute(0, 3, 1, 2)[:b0].detach().cpu() / 2 + 0.5
        conf_map_l1 = 1 / (1 + self.conf_sigma_l1[:b0, :1].detach().cpu() + EPS)
        conf_map_l1_flip = 1 / (1 + self.conf_sigma_l1[:b0, 1:].detach().cpu() + EPS)
        conf_map_percl = 1 / (1 + self.conf_sigma_percl[:b0, :1].detach().cpu() + EPS)
        conf_map_percl_flip = 1 / (1 + self.conf_sigma_percl[:b0, 1:].detach().cpu() + EPS)

        canon_im_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b0 ** 0.5))) for img in
                                torch.unbind(canon_im_rotate, 1)]  # [(C,H,W)]*T
        canon_im_rotate_grid = torch.stack(canon_im_rotate_grid, 0).unsqueeze(0)  # (1,T,C,H,W)
        canon_normal_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b0 ** 0.5))) for img in
                                    torch.unbind(canon_normal_rotate, 1)]  # [(C,H,W)]*T
        canon_normal_rotate_grid = torch.stack(canon_normal_rotate_grid, 0).unsqueeze(0)  # (1,T,C,H,W)

        ## write summary
        logger.add_scalar('Loss/loss_total', self.loss_total, total_iter)
        logger.add_scalar('Loss/loss_l1_im', self.loss_l1_im, total_iter)
        logger.add_scalar('Loss/loss_l1_im_flip', self.loss_l1_im_flip, total_iter)
        logger.add_scalar('Loss/loss_perc_im', self.loss_perc_im, total_iter)
        logger.add_scalar('Loss/loss_perc_im_flip', self.loss_perc_im_flip, total_iter)

        logger.add_histogram('Depth/canon_depth_raw_hist', canon_depth_raw_hist, total_iter)
        vlist = ['view_rx', 'view_ry', 'view_rz', 'view_tx', 'view_ty', 'view_tz']
        for i in range(self.view.shape[1]):
            logger.add_histogram('View/' + vlist[i], self.view[:, i], total_iter)
        logger.add_histogram('Light/canon_light_a', self.canon_light_a, total_iter)
        logger.add_histogram('Light/canon_light_b', self.canon_light_b, total_iter)
        llist = ['canon_light_dx', 'canon_light_dy', 'canon_light_dz']
        for i in range(self.canon_light_d.shape[1]):
            logger.add_histogram('Light/' + llist[i], self.canon_light_d[:, i], total_iter)

        def log_grid_image(label, im, nrow=int(math.ceil(b0 ** 0.5)), iter=total_iter):
            im_grid = torchvision.utils.make_grid(im, nrow=nrow)
            logger.add_image(label, im_grid, iter)

        log_grid_image('Image/input_image_symline', input_im_symline)
        log_grid_image('Image/canonical_albedo', canon_albedo)
        log_grid_image('Image/canonical_image', canon_im)
        log_grid_image('Image/recon_image', recon_im)
        log_grid_image('Image/recon_image_flip', recon_im_flip)
        log_grid_image('Image/recon_side', canon_im_rotate[:, 0, :, :, :])

        log_grid_image('Depth/canonical_depth_raw', canon_depth_raw)
        log_grid_image('Depth/canonical_depth', canon_depth)
        log_grid_image('Depth/recon_depth', recon_depth)
        log_grid_image('Depth/canonical_diffuse_shading', canon_diffuse_shading)
        log_grid_image('Depth/canonical_normal', canon_normal)
        log_grid_image('Depth/recon_normal', recon_normal)

        logger.add_histogram('Image/canonical_albedo_hist', canon_albedo, total_iter)
        logger.add_histogram('Image/canonical_diffuse_shading_hist', canon_diffuse_shading, total_iter)

        log_grid_image('Conf/conf_map_l1', conf_map_l1)
        logger.add_histogram('Conf/conf_sigma_l1_hist', self.conf_sigma_l1[:, :1], total_iter)
        log_grid_image('Conf/conf_map_l1_flip', conf_map_l1_flip)
        logger.add_histogram('Conf/conf_sigma_l1_flip_hist', self.conf_sigma_l1[:, 1:], total_iter)
        log_grid_image('Conf/conf_map_percl', conf_map_percl)
        logger.add_histogram('Conf/conf_sigma_percl_hist', self.conf_sigma_percl[:, :1], total_iter)
        log_grid_image('Conf/conf_map_percl_flip', conf_map_percl_flip)
        logger.add_histogram('Conf/conf_sigma_percl_flip_hist', self.conf_sigma_percl[:, 1:], total_iter)

        logger.add_video('Image_rotate/recon_rotate', canon_im_rotate_grid, total_iter, fps=4)
        logger.add_video('Image_rotate/canon_normal_rotate', canon_normal_rotate_grid, total_iter, fps=4)

        # visualize images and accuracy if gt is loaded
        if self.load_gt_depth:
            depth_gt = ((self.depth_gt[:b0] - self.min_depth) / (
                        self.max_depth - self.min_depth)).detach().cpu().unsqueeze(1)
            normal_gt = self.normal_gt.permute(0, 3, 1, 2)[:b0].detach().cpu() / 2 + 0.5
            sie_map_masked = self.sie_map_masked[:b0].detach().unsqueeze(1).cpu() * 1000
            norm_err_map_masked = self.norm_err_map_masked[:b0].detach().unsqueeze(1).cpu() / 100

            logger.add_scalar('Acc_masked/MAE_masked', self.acc_mae_masked.mean(), total_iter)
            logger.add_scalar('Acc_masked/MSE_masked', self.acc_mse_masked.mean(), total_iter)
            logger.add_scalar('Acc_masked/SIE_masked', self.acc_sie_masked.mean(), total_iter)
            logger.add_scalar('Acc_masked/NorErr_masked', self.acc_normal_masked.mean(), total_iter)

            log_grid_image('Depth_gt/depth_gt', depth_gt)
            log_grid_image('Depth_gt/normal_gt', normal_gt)
            log_grid_image('Depth_gt/sie_map_masked', sie_map_masked)
            log_grid_image('Depth_gt/norm_err_map_masked', norm_err_map_masked)
        '''
        return None

    def save_results(self, save_dir):
        b, c, h, w = self.input_im.shape

        with torch.no_grad():
            v0 = torch.FloatTensor([-0.1 * math.pi / 180 * 60, 0, 0, 0, 0, 0]).to(self.input_im.device).repeat(b, 1)
            canon_im_rotate = self.renderer.render_yaw(self.canon_im[:b], self.canon_depth[:b], v_before=v0, maxr=90,
                                                       nsample=15)  # (B,T,C,H,W)
            canon_im_rotate = canon_im_rotate.clamp(-1, 1).detach().cpu() / 2 + 0.5
            canon_normal_rotate = self.renderer.render_yaw(self.canon_normal[:b].permute(0, 3, 1, 2),
                                                           self.canon_depth[:b], v_before=v0, maxr=90,
                                                           nsample=15)  # (B,T,C,H,W)
            canon_normal_rotate = canon_normal_rotate.clamp(-1, 1).detach().cpu() / 2 + 0.5

        input_im = self.input_im[:b].detach().cpu().numpy() / 2 + 0.5
        input_im_symline = self.input_im_symline.detach().cpu().numpy() / 2. + 0.5
        canon_albedo = self.canon_albedo[:b].detach().cpu().numpy() / 2 + 0.5
        canon_im = self.canon_im[:b].clamp(-1, 1).detach().cpu().numpy() / 2 + 0.5
        recon_im = self.recon_im[:b].clamp(-1, 1).detach().cpu().numpy() / 2 + 0.5
        recon_im_flip = self.recon_im[b:].clamp(-1, 1).detach().cpu().numpy() / 2 + 0.5
        canon_depth = ((self.canon_depth[:b] - self.min_depth) / (self.max_depth - self.min_depth)).clamp(0,
                                                                                                          1).detach().cpu().unsqueeze(
            1).numpy()
        recon_depth = ((self.recon_depth[:b] - self.min_depth) / (self.max_depth - self.min_depth)).clamp(0,
                                                                                                          1).detach().cpu().unsqueeze(
            1).numpy()
        canon_diffuse_shading = self.canon_diffuse_shading[:b].detach().cpu().numpy()
        canon_normal = self.canon_normal[:b].permute(0, 3, 1, 2).detach().cpu().numpy() / 2 + 0.5
        recon_normal = self.recon_normal[:b].permute(0, 3, 1, 2).detach().cpu().numpy() / 2 + 0.5
        conf_map_l1 = 1 / (1 + self.conf_sigma_l1[:b, :1].detach().cpu().numpy() + EPS)
        conf_map_l1_flip = 1 / (1 + self.conf_sigma_l1[:b, 1:].detach().cpu().numpy() + EPS)
        conf_map_percl = 1 / (1 + self.conf_sigma_percl[:b, :1].detach().cpu().numpy() + EPS)
        conf_map_percl_flip = 1 / (1 + self.conf_sigma_percl[:b, 1:].detach().cpu().numpy() + EPS)
        canon_light = torch.cat([self.canon_light_a, self.canon_light_b, self.canon_light_d], 1)[
                      :b].detach().cpu().numpy()
        view = self.view[:b].detach().cpu().numpy()

        canon_im_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b ** 0.5))) for img in
                                torch.unbind(canon_im_rotate, 1)]  # [(C,H,W)]*T
        canon_im_rotate_grid = torch.stack(canon_im_rotate_grid, 0).unsqueeze(0).numpy()  # (1,T,C,H,W)
        canon_normal_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b ** 0.5))) for img in
                                    torch.unbind(canon_normal_rotate, 1)]  # [(C,H,W)]*T
        canon_normal_rotate_grid = torch.stack(canon_normal_rotate_grid, 0).unsqueeze(0).numpy()  # (1,T,C,H,W)

        canon_mid_diffuse_shading = self.canon_mid_diff_shading[:b].detach().cpu().numpy()
        canon_delta_diffuse_shading = self.canon_delta_diff_shading[:b].detach().cpu().numpy()

        sep_folder = True
        utils.save_images(save_dir, input_im, suffix='input_image', sep_folder=sep_folder)
        utils.save_images(save_dir, input_im_symline, suffix='input_image_symline', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_albedo, suffix='canonical_albedo', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_im, suffix='canonical_image', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_im, suffix='recon_image', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_im_flip, suffix='recon_image_flip', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_depth, suffix='canonical_depth', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_depth, suffix='recon_depth', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_diffuse_shading, suffix='canonical_diffuse_shading', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_normal, suffix='canonical_normal', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_normal, suffix='recon_normal', sep_folder=sep_folder)
        utils.save_images(save_dir, conf_map_l1, suffix='conf_map_l1', sep_folder=sep_folder)
        utils.save_images(save_dir, conf_map_l1_flip, suffix='conf_map_l1_flip', sep_folder=sep_folder)
        utils.save_images(save_dir, conf_map_percl, suffix='conf_map_percl', sep_folder=sep_folder)
        utils.save_images(save_dir, conf_map_percl_flip, suffix='conf_map_percl_flip', sep_folder=sep_folder)
        utils.save_txt(save_dir, canon_light, suffix='canonical_light', sep_folder=sep_folder)
        utils.save_txt(save_dir, view, suffix='viewpoint', sep_folder=sep_folder)

        utils.save_videos(save_dir, canon_im_rotate_grid, suffix='image_video', sep_folder=sep_folder, cycle=True)
        utils.save_videos(save_dir, canon_normal_rotate_grid, suffix='normal_video', sep_folder=sep_folder, cycle=True)
        utils.save_images(save_dir, canon_mid_diffuse_shading, suffix='canonical_mid_diffuse_shading',
                          sep_folder=sep_folder)
        utils.save_images(save_dir, canon_delta_diffuse_shading, suffix='canonical_delta_diffuse_shading',
                          sep_folder=sep_folder)

        # save scores if gt is loaded
        if self.load_gt_depth:
            depth_gt = ((self.depth_gt[:b] - self.min_depth) / (self.max_depth - self.min_depth)).clamp(0,
                                                                                                        1).detach().cpu().unsqueeze(
                1).numpy()
            normal_gt = self.normal_gt[:b].permute(0, 3, 1, 2).detach().cpu().numpy() / 2 + 0.5
            utils.save_images(save_dir, depth_gt, suffix='depth_gt', sep_folder=sep_folder)
            utils.save_images(save_dir, normal_gt, suffix='normal_gt', sep_folder=sep_folder)

            all_scores = torch.stack([
                self.acc_mae_masked.detach().cpu(),
                self.acc_mse_masked.detach().cpu(),
                self.acc_sie_masked.detach().cpu(),
                self.acc_normal_masked.detach().cpu()], 1)
            if not hasattr(self, 'all_scores'):
                self.all_scores = torch.FloatTensor()
            self.all_scores = torch.cat([self.all_scores, all_scores], 0)

    def save_scores(self, path):
        # save scores if gt is loaded
        if self.load_gt_depth:
            header = 'MAE_masked, \
                      MSE_masked, \
                      SIE_masked, \
                      NorErr_masked'
            mean = self.all_scores.mean(0)
            std = self.all_scores.std(0)
            header = header + '\nMean: ' + ',\t'.join(['%.8f' % x for x in mean])
            header = header + '\nStd: ' + ',\t'.join(['%.8f' % x for x in std])
            utils.save_scores(path, self.all_scores, header=header)
