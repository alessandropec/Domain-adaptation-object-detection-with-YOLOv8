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

    # More methods are provided to manage training, optimizer step, etc.


def main():
    # Set hyperparameters and seed
    seed = 95 * 27
    kwargs = {
        'imgsz': 640,
        'epochs': 100,
        'val': False,
        'workers': 2,
        'batch': 32,
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
