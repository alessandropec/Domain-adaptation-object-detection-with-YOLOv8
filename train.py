from ultralytics.models import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER, RANK, TQDM, colorstr, emojis, clean_url
import ultralytics.utils.callbacks.tensorboard as tb_module
import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import distributed as dist
import warnings
from torch.nn.utils import spectral_norm
from torchvision.ops import roi_align, box_convert
from functools import partial
import random
import os

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first



def seed_everything(seed=9527):
    # Fix random seed to make the entire training process reproducible
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class GradientReversalFunction(torch.autograd.Function):
    # Gradient reversal function, used to reverse the gradient during backpropagation
    @staticmethod
    def forward(ctx, x, alpha):
        """
        Forward pass, simply returns the input x.
        Args:
            ctx: Context object to save information for backward pass
            x: Input tensor
            alpha: Coefficient for gradient reversal, controlling the intensity of gradient reversal
        Returns:
            Tensor of the same shape as input x
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass, multiplies the incoming gradient by -alpha to reverse it.
        Args:
            ctx: Context object with saved information
            grad_output: Gradient passed back to this point
        Returns:
            Gradient multiplied by -alpha, and None for alpha since it is considered constant
        """
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, alpha=1.):
        """
        Initializes the gradient reversal layer.
        Args:
            alpha: Coefficient for gradient reversal, default is 1.0
        """
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        """
        Forward pass logic of the module, achieved through using GradientReversalFunction.
        Args:
            x: Input tensor
        Returns:
            Tensor processed by gradient reversal function
        """
        return GradientReversalFunction.apply(x, self.alpha)


class SpectralLinear(nn.Linear):
    # Linear layer using spectral normalization
    def __init__(self, *args, **kwargs):
        """
        Initializes a linear layer with spectral normalization on weights.
        """
        super().__init__(*args, **kwargs)
        spectral_norm(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class SpectralConv2d(nn.Conv2d):
    # 2D convolutional layer using spectral normalization
    def __init__(self, *args, **kwargs):
        """
        Initializes a 2D convolutional layer with spectral normalization on weights.
        """
        super().__init__(*args, **kwargs)
        spectral_norm(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class DiscriminatorHead(nn.Module):
    def __init__(self, dim_in, dim_h, dim_o=1):
        """
        Initializes the head of the discriminator with multiple layers.
        Args:
            dim_in: Dimension of input features
            dim_h: Dimension of hidden layer
            dim_o: Dimension of output layer, default is 1
        """
        super().__init__()
        # Flatten the input features and pass them through a linear transformation
        self.to_flat = nn.Sequential(
            SpectralConv2d(dim_in, dim_h // 2, kernel_size=1),
            nn.Flatten(),
            nn.LazyLinear(dim_h),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Hidden layers with multiple linear transformations
        self.neck = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_h // 2, dim_h // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ) for _ in range(3)
        ])
        # Final linear layer
        self.head = nn.Sequential(
            SpectralLinear(dim_h // 2 * 4, dim_h // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim_h // 2, dim_o, bias=False),
        )

    def forward(self, x):
        # Flatten the input features and pass through a linear transformation
        x = self.to_flat(x)
        # Similar to CSPNet, split the flattened features and aggregate different depths
        x = x.split(x.shape[1] // 2, dim=1)
        xs = [x[0]]
        for m in self.neck:
            x = m(x[1]) if isinstance(x, tuple) else m(x)
            xs.append(x)
        # Concatenate different depth features
        x = torch.cat(xs, dim=1)
        return self.head(x)


class Discriminator(nn.Module):
    def __init__(self, chs=None, amp=False):
        """
        Initializes the discriminator, including gradient reversal and convolutional layers.
        Args:
            chs: List of channels defining input and output dimensions for each layer
            amp: Whether to use automatic mixed precision training, default is False
        """
        super().__init__()
        if chs is None:
            chs = [64, 128, 256]
            self.chs = chs
            self.f_len = len(chs)
        # Initialize gradient reversal layer
        self.grl = GradientReversalLayer(alpha=1.0)
        self.amp = amp
        # Convolutional layers to extract features from different depths, defined by chs
        self.p = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chs[i] if i == 0 else chs[i] * 2, 64, kernel_size=11, stride=2, padding=5, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, chs[i + 1] if i + 1 < len(chs) else chs[i], kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(chs[i + 1] if i + 1 < len(chs) else chs[i]),
                nn.SiLU(inplace=True),
            ) for i in range(len(chs))
        ])
        # Aggregated feature head
        self.head = DiscriminatorHead(chs[-1], 256)
        # Optimizer for discriminator
        self.optim = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

    def forward(self, fs: list[torch.tensor]):
        with torch.cuda.amp.autocast(self.amp):
            assert len(fs) == self.f_len, f'Expected {self.f_len} feature maps, got {len(fs)}'
            # Pass features through the gradient reversal layer
            fs = [self.grl(f) for f in fs]
            # Aggregate features from different depths
            x = self.p[0](fs[0])
            for i in range(1, len(fs)):
                x = torch.cat((x, fs[i]), dim=1)
                x = self.p[i](x)
            # Make predictions using the head
            return self.head(x)
        
# from copy import deepcopy
# from ultralytics.utils.ops import resample_segments
# from ultralytics.data.dataset import Instances
class MultiDomainYoloDataset(YOLODataset):

    def __init__(self,
                 *args, data, task, **kwargs):
        super(MultiDomainYoloDataset,self).__init__(
       *args, data=data, task=task, **kwargs)
        
        self.domain_labels=[i for i,l in enumerate(self.labels)]

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        item= self.transforms(self.get_image_and_label(index))
        item["domain"]=self.domain_labels[index]
        return item

    # def get_image_and_label(self, index):
    #     """Get and return label information from the dataset."""
    #     label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
    #     label.pop("shape", None)  # shape is for rect, remove it
    #     label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
    #     label["ratio_pad"] = (
    #         label["resized_shape"][0] / label["ori_shape"][0],
    #         label["resized_shape"][1] / label["ori_shape"][1],
    #     )  # for evaluation
    #     if self.rect:
    #         label["rect_shape"] = self.batch_shapes[self.batch[index]]
    #     #label["domain"]=self.domain_labels[index]
    #     return self.update_labels_info(label)
    
    # def update_labels_info(self, label):
    #     """
    #     Custom your label format here.

    #     Note:
    #         cls is not with bboxes now, classification and semantic segmentation need an independent cls label
    #         Can also support classification and semantic segmentation by adding or removing dict keys there.
    #     """
    #     bboxes = label.pop("bboxes")
    #     segments = label.pop("segments", [])
    #     keypoints = label.pop("keypoints", None)
    #     bbox_format = label.pop("bbox_format")
    #     normalized = label.pop("normalized")

    #     # NOTE: do NOT resample oriented boxes
    #     segment_resamples = 100 if self.use_obb else 1000
    #     if len(segments) > 0:
    #         # list[np.array(1000, 2)] * num_samples
    #         # (N, 1000, 2)
    #         segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
    #     else:
    #         segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
    #     label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
    #     return label

class CustomTrainer(DetectionTrainer):
    """
    Custom trainer used for training with both source domain and target domain.
    Extends ultralytics.models.yolo.detect.train.DetectionTrainer.
    """
    def __init__(self, target_domain_data_cfg, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)
        try:
            if target_domain_data_cfg in ('yaml', 'yml') or self.args.task in ('detect', 'segment', 'pose'):
                self.t_data = check_det_dataset(target_domain_data_cfg)
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(target_domain_data_cfg)}' error ❌ {e}")) from e
        # Initialize target domain dataset and data loader
        self.t_trainset, self.t_testset = self.get_dataset() #self.get_dataset(self.t_data)
        self.t_iter = None
        self.t_train_loader = None
        # Initialize parameters for domain adaptation and model hooks
        self.model_hook_handler = []
        self.model_hook_layer_idx: list[int] = [2, 4, 6]
        self.roi_size = list(reversed([20 * 2 ** i for i in range(len(self.model_hook_layer_idx))]))
        self.model_hooked_features: None | list[torch.tensor] = None
        # Initialize discriminator and other helper models
        self.discriminator_model = None
        self.projection_model = None
        self.additional_models = []
        self.add_callback('on_train_start', self.init_helper_model)

    def init_helper_model(self, *args, **kwargs):
        self.discriminator_model = Discriminator(amp=self.amp).to(self.device)
        self.additional_models.append(self.discriminator_model)

    def get_t_batch(self):
        # Obtain a batch from the target domain dataset
        if self.t_iter is None:
            self.t_train_loader = self.get_dataloader(self.t_trainset, batch_size=self.batch_size, rank=RANK, mode='train')
            self.t_iter = iter(self.t_train_loader)
        try:
            batch = next(self.t_iter)
        except StopIteration:
            self.t_iter = iter(self.t_train_loader)
            batch = next(self.t_iter)
        return batch
    
    def build_yolo_dataset(self,cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
        """Build YOLO Dataset."""
        dataset = MultiDomainYoloDataset
        return dataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",  # augmentation
            hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
            rect=cfg.rect or rect,  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=cfg.task,
            classes=cfg.classes,
            data=data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )
    
    def build_dataset(self, img_path, mode="train", batch=None,rect=False):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        
        cfg=self.args
        return self.build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
      

    def activate_hook(self, layer_indices: list[int] = None):
        # Activate hooks to extract features from specific layers
        if layer_indices is not None:
            self.model_hook_layer_idx = layer_indices
        self.model_hooked_features = [None for _ in self.model_hook_layer_idx]
        self.model_hook_handler = [
            self.model.model[l].register_forward_hook(self.hook_fn(i)) for i, l in enumerate(self.model_hook_layer_idx)]

    def deactivate_hook(self):
        # Deactivate hooks
        if self.model_hook_handler is not None:
            for hook in self.model_hook_handler:
                hook.remove()
            self.model_hooked_features = None
            self.model_hook_handler = []

    def hook_fn(self, hook_idx):
        def hook(m, i, o):
            self.model_hooked_features[hook_idx] = o
        return hook

    def get_dis_output_from_hooked_features(self, batch):
        if self.model_hooked_features is not None:
            bbox_batch_idx = batch['batch_idx'].unsqueeze(-1)
            bbox = batch['bboxes']
            bbox = box_convert(bbox, 'cxcywh', 'xyxy')
            rois = []
            for fidx, f in enumerate(self.model_hooked_features):
                f_bbox = bbox * f.shape[-1]
                f_bbox = torch.cat((bbox_batch_idx, f_bbox), dim=-1)
                f_roi = roi_align(f, f_bbox.to(f.device), output_size=self.roi_size[fidx], aligned=True)
                rois.append(f_roi)
            dis_output = self.discriminator_model(rois)
            return dis_output
        else:
            return None

    def optimizer_step(self, optims: None | list[torch.optim.Optimizer] = None):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        
        # Unscales the gradients of the optimizer for mixed precision training.
        self.scaler.unscale_(self.optimizer)

        # If additional optimizers are provided (e.g., for the discriminator model), unscale their gradients as well.
        if optims is not None:
            for o in optims:
                # Check if the optimizer has gradients before unscaling.
                if o.param_groups[0]['params'][0].grad is not None:
                    self.scaler.unscale_(o)

        # Clip gradients that exceed the maximum allowed value to stabilize training.
        max_norm = 10.0
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)  # clip gradients of main model
        if len(self.additional_models) > 0:
            # Clip gradients of additional models, such as the discriminator, with a higher threshold.
            for m in self.additional_models:
                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=max_norm * 2)

        # Step the optimizer using the scaler, effectively updating the model parameters.
        self.scaler.step(self.optimizer)
        
        # If additional optimizers are provided, step them as well.
        if optims is not None:
            for o in optims:
                if o.param_groups[0]['params'][0].grad is not None:
                    self.scaler.step(o)

        # Update the scaler for the next iteration.
        self.scaler.update()

        # Reset the optimizer gradients to zero, preparing for the next iteration.
        self.optimizer.zero_grad()
        if optims is not None:
            for o in optims:
                o.zero_grad()

        # If EMA (Exponential Moving Average) is being used, update the EMA model.
        if self.ema:
            self.ema.update(self.model)

    def _do_train(self, world_size=1):
        # Set up distributed training settings if required.
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        # Initialize parameters for training.
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # Number of batches.
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # Warmup iterations.
        last_opt_step = -1

        # Run the callback for training start, which includes initializing the discriminator.
        self.run_callbacks('on_train_start')

        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')

        # Close mosaic augmentation after a certain number of epochs if required.
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

        # Start training loop, activating hooks for feature extraction.
        epoch = self.epochs  # Predetermined epoch value for potential resume cases.
        self.activate_hook()  # Activate hooks for collecting feature maps.

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()  # Set the model to training mode.
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)

            pbar = enumerate(self.train_loader)
            
            # Close mosaic augmentation at the specified epoch.
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)

            # Initialize training parameters, reset loss, optimizer.
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                # Run callback at the start of each batch.
                self.run_callbacks('on_train_batch_start')

                # Handle warmup learning rate adjustments.
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        x['lr'] = np.interp(ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward pass with mixed precision (AMP).
                with torch.cuda.amp.autocast(self.amp):
                    # Obtain object detection loss and feature hooks for the source domain.
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size  # Scale loss if in distributed training.
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items

                    # Get discriminator output for the source domain.
                    source_critics = self.get_dis_output_from_hooked_features(batch)

                    # Obtain and preprocess target domain batch.
                    t_batch = self.get_t_batch()
                    t_batch = self.preprocess_batch(t_batch)
                    t_loss, t_loss_item = self.model(t_batch)
                    
                    # Get discriminator output for the target domain.
                    target_critics = self.get_dis_output_from_hooked_features(t_batch)

                    # Combine source and target object detection loss.
                    self.loss += t_loss

                    # Train discriminator for domain adaptation, but skip it during specific epochs.
                    if 6 < epoch < self.args.epochs - 50:
                        # Compute discriminator loss using hinge loss.
                        threshold = 20
                        loss_d = (F.relu(torch.ones_like(source_critics) * threshold + source_critics)).mean()
                        loss_d += (F.relu(torch.ones_like(target_critics) * threshold - target_critics)).mean()
                    else:
                        loss_d = 0
                    
                    # Add discriminator loss to the total loss.
                    self.loss += loss_d * 2

                # Backward pass with scaled loss.
                self.scaler.scale(self.loss).backward()

                # Step optimizer if the accumulation condition is met.
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step(optims=[self.discriminator_model.optim])
                    last_opt_step = ni

                # Logging and monitoring progress.
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1])
                    )
                    self.run_callbacks('on_batch_end')
                    tb_module.WRITER.add_scalar('train/critic-source', source_critics.mean(), ni)
                    tb_module.WRITER.add_scalar('train/critic-target', target_critics.mean(), ni)
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

            # Scheduler and EMA updates.
            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            # Validation and model saving.
            if RANK in (-1, 0):
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Deactivate hooks, save model, then reactivate hooks for further training.
                if self.args.save or (epoch + 1 == self.epochs):
                    self.deactivate_hook()
                    self.save_model()
                    self.run_callbacks('on_model_save')
                    self.activate_hook()

            # Time tracking and cleanup.
            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()

            # Early Stopping Handling.
            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break

        # Final validation and hook deactivation.
        if RANK in (-1, 0):
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in {(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')

        self.deactivate_hook()
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')




def main():
    # Set hyperparameters and seed
    seed = 95 * 27
    kwargs = {
        'imgsz': 640,
        'epochs': 100,
        'val': False,
        'workers': 0,
        'batch': 1,
        'seed': seed,
    }
    seed_everything(seed)
    # Load pretrained YOLO model
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    # Initialize custom trainer
    custom_trainer = partial(CustomTrainer, target_domain_data_cfg='data/ACDC-rain.yaml')
    # Train using custom trainer
    model.train(custom_trainer, data='data/ACDC-fog.yaml', name='train_RMD', patience=0, **kwargs)
    # Validate model on rain and fog domains
    model.val(data='data/ACDC-rain.yaml', name='val_RMD_rain')
    model.val(data='data/ACDC-fog.yaml', name='val_RMD_fog')

    # Further training on specific domains, using pre-trained models
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    model.train(data='data/ACDC-fog.yaml', name='train_fog', **kwargs)
    model.val(data='data/ACDC-fog.yaml', name='val_fog_fog')
    model.val(data='data/ACDC-rain.yaml', name='val_fog_rain')

    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    model.train(data='data/ACDC-rain.yaml', name='train_rain', **kwargs)
    model.val(data='data/ACDC-rain.yaml', name='val_rain_rain')
    model.val(data='data/ACDC-fog.yaml', name='val_rain_fog')

    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    model.train(data='data/ACDC-fog_rain.yaml', name='train_fog_rain-full-epoch', **kwargs)
    model.val(data='data/ACDC-rain.yaml', name='val_fograin_rain-full-epoch')
    model.val(data='data/ACDC-fog.yaml', name='val_fograin_fog-full-epoch')


if __name__ == '__main__':
    main()
