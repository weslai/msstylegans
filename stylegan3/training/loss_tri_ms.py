# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        # logits = self.D(img, c, update_emas=update_emas)
        outpus_d = self.D(img, c, update_emas=update_emas)
        return outpus_d

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, lambda_):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c)
                # gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                gen_outputs_d = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                gen_outputs_d = gen_outputs_d.float()
                if real_img.shape[1] == 3: ## Retinal images
                    print("it is retinal")
                    gen_img_pred = gen_outputs_d[:, 0]
                    gen_cmap_pred = gen_outputs_d[:, [1, 3]]
                    gen_discease_pred = gen_outputs_d[:, [2] + list(range(4, gen_outputs_d.shape[1]-3))]
                    assert gen_discease_pred.shape[1] == 5
                    gen_source_pred = gen_outputs_d[:, -3:]
                elif gen_outputs_d.shape[1] == 12: ## MRI
                    print("it is mri")
                    gen_img_pred = gen_outputs_d[:, 0]
                    gen_cmap_pred = gen_outputs_d[:, 1:6] ## cmap prediction
                    # gen_cdr_pred = gen_outputs_d[:, 4:7] ## cdr prediction
                    gen_apoe_pred = gen_outputs_d[:, 6:-3] ## apoe prediction
                    gen_source_pred = gen_outputs_d[:, -3:] ## source prediction
                else:
                    raise ValueError('gen_outputs_d.shape[1] is not 11 or 12')
                training_stats.report('Loss/scores/fake', gen_img_pred)
                training_stats.report('Loss/signs/fake', gen_img_pred.sign())
                bce_loss = torch.nn.functional.binary_cross_entropy(
                    torch.sigmoid(gen_img_pred), torch.ones_like(gen_img_pred, requires_grad=True).to(self.device))
                if real_img.shape[1] == 3: ## Retinal images
                    gen_c = gen_c.float()
                    mse_loss = torch.nn.functional.mse_loss(gen_cmap_pred, gen_c[:, [0, 2]])
                    gen_discease_c = gen_c[:, [1] + list(range(3, gen_c.shape[1]-3))]
                    bce_discease_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        gen_discease_pred, gen_discease_c
                    )
                    source = torch.where(gen_c[:, -3:] == 1)[1]
                    ce_source_loss = torch.nn.functional.cross_entropy(
                        gen_source_pred, source.long())
                    training_stats.report('Loss/scores/fake_labels', mse_loss)
                    training_stats.report('Loss/scores/fake_discease (bce loss)', bce_discease_loss)
                    training_stats.report('Loss/scores/fake_source', ce_source_loss)
                    loss_Gmain = bce_loss + (mse_loss + bce_discease_loss + ce_source_loss) * lambda_
                elif gen_outputs_d.shape[1] == 12: ## mri
                    gen_c = gen_c.float()
                    mse_loss = torch.nn.functional.mse_loss(gen_cmap_pred, gen_c[:,0:5])
                    # cdr = torch.where(gen_c[:, 3:6] == 1)[1]
                    # ce_cdr_loss = torch.nn.functional.cross_entropy(
                    #     gen_cdr_pred, cdr.long())
                    apoe = torch.where(gen_c[:, 5:8] == 1)[1]
                    ce_apoe_loss = torch.nn.functional.cross_entropy(
                        gen_apoe_pred, apoe.long())
                    source = torch.where(gen_c[:, -3:] == 1)[1]
                    ce_source_loss = torch.nn.functional.cross_entropy(
                        gen_source_pred, source.long())
                    training_stats.report('Loss/scores/fake_labels', mse_loss)
                    # training_stats.report('Loss/scores/fake_cdr (ce loss)', ce_cdr_loss)
                    training_stats.report('Loss/scores/fake_apoe (ce loss)', ce_apoe_loss)
                    training_stats.report('Loss/scores/fake_source', ce_source_loss)
                    loss_Gmain = bce_loss + (mse_loss + ce_apoe_loss + ce_source_loss) * lambda_
                else:
                    loss_Gmain = bce_loss
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
                # gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                gen_outputs_d = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                gen_outputs_d = gen_outputs_d.float()
                if real_img.shape[1] == 3: ## Retinal images
                    gen_img_pred = gen_outputs_d[:, 0]
                    gen_cmap_pred = gen_outputs_d[:, [1, 3]]
                    gen_discease_pred = gen_outputs_d[:, [2] + list(range(4, gen_outputs_d.shape[1]-3))]
                    assert gen_discease_pred.shape[1] == 5
                    gen_source_pred = gen_outputs_d[:, -3:]
                elif gen_outputs_d.shape[1] == 12: ## MRI
                    gen_img_pred = gen_outputs_d[:, 0]
                    gen_cmap_pred = gen_outputs_d[:, 1:6] ## cmap prediction
                    # gen_cdr_pred = gen_outputs_d[:, 4:7] ## cdr prediction
                    gen_apoe_pred = gen_outputs_d[:, 6:-3] ## apoe prediction
                    gen_source_pred = gen_outputs_d[:, -3:] ## source prediction
                else:
                    raise ValueError('gen_outputs_d.shape[1] is not 11 or 12')
                training_stats.report('Loss/scores/fake', gen_img_pred)
                training_stats.report('Loss/signs/fake', gen_img_pred.sign())
                # training_stats.report('Loss/scores/fake', gen_logits)
                # training_stats.report('Loss/signs/fake', gen_logits.sign())
                # loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                bce_loss = torch.nn.functional.binary_cross_entropy(
                    torch.sigmoid(gen_img_pred), torch.zeros_like(gen_img_pred, requires_grad=True).to(self.device))
                
                loss_Dgen = bce_loss
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                # real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                real_outputs_d = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                real_outputs_d = real_outputs_d.float()
                if real_img.shape[1] == 3: ## Retinal images
                    real_img_pred = real_outputs_d[:, 0]
                    real_cmap_pred = real_outputs_d[:, [1, 3]]
                    real_discease_pred = real_outputs_d[:, [2] + list(range(4, real_outputs_d.shape[1]-3))]
                    assert real_discease_pred.shape[1] == 5
                    real_source_pred = real_outputs_d[:, -3:]
                elif real_outputs_d.shape[1] == 12: ## MRI
                    real_img_pred = real_outputs_d[:, 0]
                    real_cmap_pred = real_outputs_d[:, 1:6] ## cmap prediction
                    # real_cdr_pred = real_outputs_d[:, 4:7] ## cdr prediction
                    real_apoe_pred = real_outputs_d[:, 6:-3] ## apoe prediction
                    real_source_pred = real_outputs_d[:, -3:] ## source prediction
                else:
                    raise ValueError('real_outputs_d.shape[1] is not 11 or 12')
                training_stats.report('Loss/scores/real', real_img_pred)
                training_stats.report('Loss/signs/real', real_img_pred.sign())
                # training_stats.report('Loss/scores/real', real_logits)
                # training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    bce_loss = torch.nn.functional.binary_cross_entropy(
                        torch.sigmoid(real_img_pred), torch.ones_like(real_img_pred, requires_grad=True).to(self.device))
                    if real_img.shape[1] == 3: ## Retinal images
                        real_c = real_c.float()
                        mse_loss = torch.nn.functional.mse_loss(real_cmap_pred, real_c[:, [0, 2]])
                        real_discease_c = real_c[:, [1] + list(range(3, real_c.shape[1]-3))]
                        bce_discease_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            real_discease_pred, real_discease_c
                        )
                        source = torch.where(real_c[:, -3:] == 1)[1]
                        ce_source_loss = torch.nn.functional.cross_entropy(
                            real_source_pred, source.long())
                        loss_Dreal = bce_loss + (mse_loss + bce_discease_loss + ce_source_loss) * lambda_
                    elif real_outputs_d.shape[1] == 12: ## mri
                        real_c = real_c.float()
                        mse_loss = torch.nn.functional.mse_loss(real_cmap_pred, real_c[:,0:5])
                        # cdr = torch.where(real_c[:, 3:6] == 1)[1]
                        # ce_cdr_loss = torch.nn.functional.cross_entropy(
                        #     real_cdr_pred, cdr.long())
                        apoe = torch.where(real_c[:, 5:-3] == 1)[1]
                        ce_apoe_loss = torch.nn.functional.cross_entropy(
                            real_apoe_pred, apoe.long())
                        source = torch.where(real_c[:, -3:] == 1)[1]
                        ce_source_loss = torch.nn.functional.cross_entropy(
                            real_source_pred, source.long())
                        loss_Dreal = bce_loss + (mse_loss + ce_apoe_loss + ce_source_loss) * lambda_
                    else:
                        loss_Dreal = bce_loss
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_img_pred.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        # r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
