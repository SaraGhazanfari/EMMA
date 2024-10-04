import os

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPTextModel, CLIPVisionConfig

from llava.model.multimodal_encoder.tokenizer import get_tokenizer
# NUM_TEXT_TOKENS = [28, 77, 115, 230] #0.05, 0.1, 0.2, 0.4
HIDDEN_SIZE = 768

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.text_module_path = getattr(args, 'pretrain_mm_mlp_adapter', None)
        
        if self.text_module_path is None:
            self.text_module_path = getattr(args, '_name_or_path', None)
        else:
            self.text_module_path = '/'.join(self.text_module_path.split('/')[:-1])
            
        self.encoder_version = getattr(args, 'encoder_version', None)
        self.num_learnable_tokens = getattr(args, 'num_learnable_tokens', None)
        self.mm_text_select_layer = getattr(args, 'mm_text_select_layer', None)
        self.mm_text_select_feature = getattr(args, 'mm_text_select_feature', None)
        assert self.num_learnable_tokens == 0
        
        if self.encoder_version == 'v2':
            self.mm_text_num_tokens = 0
        else:
            self.mm_text_num_tokens = 77
            if self.mm_text_select_feature == 'cls':
                self.mm_text_num_tokens = 1
            
            elif self.mm_text_select_feature == 'patch':
                self.mm_text_num_tokens = 76
            
        # self.load_model()
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map, output_attentions=True)
        if self.encoder_version == 'v1' or self.encoder_version == 'v3':
            self.text_encoder = CLIPTextModel.from_pretrained(self.vision_tower_name, device_map=device_map)
            self.text_encoder.requires_grad_(False)
        
        self.vision_tower.requires_grad_(False)
        

        self.is_loaded = True
        self.create_text_modules()
        self.load_text_modules()

    def create_text_modules(self):
        self.text_projection = nn.Linear(HIDDEN_SIZE, self.config.hidden_size,
                                         device=self.device)
        num_image_tokens = int(self.config.image_size / self.config.patch_size) ** 2
        num_added_tokens = num_image_tokens + self.num_learnable_tokens + self.mm_text_num_tokens
        if self.encoder_version == 'v2' or self.encoder_version == 'v3':
            self.learnable_tokens = nn.Parameter(torch.zeros(num_image_tokens, self.config.hidden_size), requires_grad=True)
        
        fusion_parameter = torch.zeros((num_added_tokens, num_image_tokens), dtype=self.dtype,
                                       device=self.device)
        fusion_parameter[:num_image_tokens, :num_image_tokens] = torch.eye(num_image_tokens, dtype=self.dtype,
                                                                           device=self.device)
        self.image_text_infusion = nn.Linear(num_added_tokens, num_image_tokens, bias=True,
                                             device=self.device)
        self.image_text_infusion.weight = nn.Parameter(fusion_parameter.permute(1, 0), requires_grad=True)
        self.image_text_infusion.bias = nn.Parameter(torch.zeros(num_image_tokens, device=self.device),
                                                     requires_grad=True)

        self.tokenizer = get_tokenizer()

    def load_text_modules(self):
        if self.text_module_path:
            self.text_module_path = os.path.join(self.text_module_path, 'vision_tower.bin')
            state_dict = torch.load(self.text_module_path)
            text_projection, learnable_tokens, text_features, image_text_infusion_dict = dict(), None, dict(), dict()
            for key, value in state_dict.items():
                if key.startswith('text_projection'):
                    text_projection[key.replace('text_projection.', '')] = value
                elif key.startswith('model.vision_tower.text_projection'):
                    text_projection[key.replace('model.vision_tower.text_projection.', '')] = value
                elif key.startswith('learnable_tokens'):
                    learnable_tokens = value
                elif key.startswith('model.vision_tower.learnable_tokens'):
                    learnable_tokens = value
                elif key.startswith('image_text_infusion'):
                    image_text_infusion_dict[key.replace('image_text_infusion.', '')] = value
                elif key.startswith('model.vision_tower.image_text_infusion'):
                    image_text_infusion_dict[key.replace('model.vision_tower.image_text_infusion.', '')] = value
            self.text_projection.load_state_dict(text_projection)
            if self.encoder_version != 'v1' and learnable_tokens is not None:
                self.learnable_tokens = nn.Parameter(learnable_tokens.to(device=self.device))
            self.image_text_infusion.load_state_dict(image_text_infusion_dict)
            
            if self.encoder_version == 'v3':
                for module in [self.image_text_infusion, self.text_projection]:
                    for name, param in module.named_parameters():
                        param.requires_grad = False
            print('Text modules were loaded successfully!')
        else:
            print('There is no pretrained version of Text_Projection and Image_Text_Infusion')

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features
    
    def text_feature_select(self, instruct):
        text_features = self.text_encoder(instruct.to(device=self.device), output_hidden_states=True).hidden_states[
                self.mm_text_select_layer]
        if self.mm_text_select_feature == 'patch':
            text_features = text_features[:, 1:]
        elif self.mm_text_select_feature == 'cls':
            text_features = text_features[:, :1]
        elif self.mm_text_select_feature == 'cls_patch':
            text_features = text_features
        return text_features
        
    def forward(self, images, instruct=None):
        
        with torch.no_grad():
            if self.encoder_version != 'v2':
                 text_features = self.text_feature_select(instruct)
            if type(images) is list:
                image_features = []
                for image in images:
                    image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                          output_hidden_states=True)
                    image_feature = self.feature_select(image_forward_out).to(image.dtype)
                    image_features.append(image_feature)
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                       output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs).to(images.dtype)

        if self.encoder_version in ['v1', 'v3']:
            text_features = nn.GELU()(self.text_projection(text_features))
            
        elif self.encoder_version == 'v2': 
            text_features = self.learnable_tokens(torch.eye(self.num_learnable_tokens).to(device=self.device, dtype=self.dtype))   
            text_features = nn.GELU()(self.text_projection(text_features))
            text_features = text_features.to(self.dtype).unsqueeze(0).expand(images.size(0), -1, -1)
            
        infused_image_features = torch.cat((image_features, text_features), dim=1)
        
        infused_image_features = self.image_text_infusion(infused_image_features.permute(0, 2, 1)).permute(0, 2, 1)
        try:
            assert not torch.equal(image_features, infused_image_features)
        except:
            print('Image Feature is NOT changing')
        if self.encoder_version == 'v3':
            infused_image_features = infused_image_features + self.learnable_tokens
        return infused_image_features


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)