"""
Multi-Modal AI Framework
Implements advanced multi-modal learning capabilities for vision, text, and audio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Any, Union
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None
    
try:
    import torchvision.models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ModalityConfig:
    """Configuration for different modalities."""
    modality_type: str  # 'vision', 'text', 'audio', 'tabular'
    input_dim: int
    hidden_dim: int
    output_dim: int
    preprocessing: Dict[str, Any]
    model_config: Dict[str, Any]

class ModalityEncoder(ABC):
    """Abstract base class for modality encoders."""
    
    @abstractmethod
    def encode(self, input_data: Any) -> torch.Tensor:
        """Encode input data to feature representation."""
        pass
    
    @abstractmethod
    def get_output_dim(self) -> int:
        """Get output dimension of encoded features."""
        pass

class VisionEncoder(ModalityEncoder, nn.Module):
    """Vision encoder using CNN or Vision Transformer."""
    
    def __init__(self, config: ModalityConfig):
        super().__init__()
        self.config = config
        self.encoder_type = config.model_config.get('encoder_type', 'resnet')
        
        if self.encoder_type == 'resnet':
            self.encoder = self._build_resnet_encoder()
        elif self.encoder_type == 'vit':
            self.encoder = self._build_vit_encoder()
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")
        
        self.projection = nn.Linear(self._get_encoder_output_dim(), config.output_dim)
        
    def _build_resnet_encoder(self) -> nn.Module:
        """Build ResNet-based encoder."""
        if not TORCHVISION_AVAILABLE:
            # Fallback to simple CNN if torchvision not available
            return nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
        
        try:
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            # Remove final classification layer
            return nn.Sequential(*list(resnet.children())[:-1])
        except ImportError:
            # Fallback if import fails
            return nn.Sequential(
                nn.Conv2d(3, 256, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
    
    def _build_vit_encoder(self) -> nn.Module:
        """Build Vision Transformer encoder."""
        from transformers import ViTModel
        return ViTModel.from_pretrained('google/vit-base-patch16-224')
    
    def _get_encoder_output_dim(self) -> int:
        """Get output dimension from encoder."""
        if self.encoder_type == 'resnet':
            return 2048 if TORCHVISION_AVAILABLE else 256
        elif self.encoder_type == 'vit':
            return 768
        return self.config.hidden_dim
    
    def encode(self, input_data: torch.Tensor) -> torch.Tensor:
        """Encode vision input."""
        if self.encoder_type == 'resnet':
            features = self.encoder(input_data)
            features = features.view(features.size(0), -1)
        elif self.encoder_type == 'vit':
            features = self.encoder(input_data).last_hidden_state[:, 0]  # CLS token
        
        return self.projection(features)
    
    def get_output_dim(self) -> int:
        return self.config.output_dim

class TextEncoder(ModalityEncoder, nn.Module):
    """Text encoder using transformer models."""
    
    def __init__(self, config: ModalityConfig):
        super().__init__()
        self.config = config
        self.model_name = config.model_config.get('model_name', 'bert-base-uncased')
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name)
        
        # Freeze encoder parameters for transfer learning
        if config.model_config.get('freeze_encoder', True):
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        encoder_dim = self.encoder.config.hidden_size
        self.projection = nn.Linear(encoder_dim, config.output_dim)
        
    def encode(self, input_data: Union[str, List[str]]) -> torch.Tensor:
        """Encode text input."""
        if isinstance(input_data, str):
            input_data = [input_data]
        
        # Tokenize
        inputs = self.tokenizer(
            input_data,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Encode
        with torch.no_grad() if self.config.model_config.get('freeze_encoder', True) else torch.enable_grad():
            outputs = self.encoder(**inputs)
        
        # Use CLS token representation
        cls_embeddings = outputs.last_hidden_state[:, 0]
        return self.projection(cls_embeddings)
    
    def get_output_dim(self) -> int:
        return self.config.output_dim

class AudioEncoder(ModalityEncoder, nn.Module):
    """Audio encoder using CNN or transformer for audio features."""
    
    def __init__(self, config: ModalityConfig):
        super().__init__()
        self.config = config
        self.sample_rate = config.preprocessing.get('sample_rate', 16000)
        self.n_mels = config.preprocessing.get('n_mels', 128)
        self.encoder_type = config.model_config.get('encoder_type', 'cnn')
        
        if self.encoder_type == 'cnn':
            self.encoder = self._build_cnn_encoder()
        elif self.encoder_type == 'transformer':
            self.encoder = self._build_transformer_encoder()
        
        self.projection = nn.Linear(self._get_encoder_output_dim(), config.output_dim)
    
    def _build_cnn_encoder(self) -> nn.Module:
        """Build CNN-based audio encoder."""
        return nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def _build_transformer_encoder(self) -> nn.Module:
        """Build transformer-based audio encoder."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_mels,
            nhead=8,
            batch_first=True
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=6)
    
    def _get_encoder_output_dim(self) -> int:
        """Get output dimension from encoder."""
        if self.encoder_type == 'cnn':
            return 256
        elif self.encoder_type == 'transformer':
            return self.n_mels
        return self.config.hidden_dim
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> torch.Tensor:
        """Preprocess audio to mel-spectrogram."""
        # Convert to mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=512
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / log_mel_spec.std()
        
        return torch.tensor(log_mel_spec, dtype=torch.float32)
    
    def encode(self, input_data: np.ndarray) -> torch.Tensor:
        """Encode audio input."""
        # Preprocess audio
        mel_spec = self._preprocess_audio(input_data)
        
        if self.encoder_type == 'cnn':
            # Add batch and channel dimensions
            mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)
            features = self.encoder(mel_spec)
        elif self.encoder_type == 'transformer':
            # Transpose for transformer input (seq_len, features)
            mel_spec = mel_spec.transpose(0, 1).unsqueeze(0)
            features = self.encoder(mel_spec)
            features = features.mean(dim=1)  # Global average pooling
        
        return self.projection(features)
    
    def get_output_dim(self) -> int:
        return self.config.output_dim

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for fusion."""
    
    def __init__(self, input_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor) -> torch.Tensor:
        """Apply cross-modal attention."""
        attended, _ = self.attention(query, key, value)
        return self.norm(attended + query)

class MultiModalFusion(nn.Module):
    """Multi-modal fusion module with various fusion strategies."""
    
    def __init__(self, modality_dims: Dict[str, int], fusion_dim: int, 
                 fusion_strategy: str = 'attention'):
        super().__init__()
        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        self.fusion_strategy = fusion_strategy
        
        # Project each modality to fusion dimension
        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(dim, fusion_dim)
            for modality, dim in modality_dims.items()
        })
        
        if fusion_strategy == 'attention':
            self.fusion_layer = self._build_attention_fusion()
        elif fusion_strategy == 'concatenation':
            self.fusion_layer = self._build_concat_fusion()
        elif fusion_strategy == 'gated':
            self.fusion_layer = self._build_gated_fusion()
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
    
    def _build_attention_fusion(self) -> nn.Module:
        """Build attention-based fusion."""
        return CrossModalAttention(self.fusion_dim)
    
    def _build_concat_fusion(self) -> nn.Module:
        """Build concatenation-based fusion."""
        total_dim = len(self.modality_dims) * self.fusion_dim
        return nn.Sequential(
            nn.Linear(total_dim, self.fusion_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_dim, self.fusion_dim)
        )
    
    def _build_gated_fusion(self) -> nn.Module:
        """Build gated fusion mechanism."""
        num_modalities = len(self.modality_dims)
        return nn.Sequential(
            nn.Linear(self.fusion_dim * num_modalities, self.fusion_dim),
            nn.Sigmoid()
        )
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse multi-modal features."""
        # Project each modality
        projected_features = {}
        for modality, features in modality_features.items():
            if modality in self.modality_projections:
                projected_features[modality] = self.modality_projections[modality](features)
        
        if self.fusion_strategy == 'attention':
            # Use first modality as query, others as key/value
            modalities = list(projected_features.keys())
            query = projected_features[modalities[0]]
            
            fused = query
            for i in range(1, len(modalities)):
                key_value = projected_features[modalities[i]]
                fused = self.fusion_layer(fused, key_value, key_value)
            
            return fused
        
        elif self.fusion_strategy == 'concatenation':
            # Concatenate all modalities
            concatenated = torch.cat(list(projected_features.values()), dim=-1)
            return self.fusion_layer(concatenated)
        
        elif self.fusion_strategy == 'gated':
            # Gated fusion
            stacked = torch.stack(list(projected_features.values()), dim=-1)
            gates = self.fusion_layer(stacked.flatten(start_dim=-2))
            weighted = (stacked * gates.unsqueeze(-1)).sum(dim=-1)
            return weighted

class MultiModalModel(nn.Module):
    """Complete multi-modal learning model."""
    
    def __init__(self, modality_configs: Dict[str, ModalityConfig],
                 fusion_config: Dict[str, Any], num_classes: int):
        super().__init__()
        self.modality_configs = modality_configs
        self.num_classes = num_classes
        
        # Initialize encoders
        self.encoders = nn.ModuleDict()
        for modality, config in modality_configs.items():
            if modality == 'vision':
                self.encoders[modality] = VisionEncoder(config)
            elif modality == 'text':
                self.encoders[modality] = TextEncoder(config)
            elif modality == 'audio':
                self.encoders[modality] = AudioEncoder(config)
            else:
                raise ValueError(f"Unknown modality: {modality}")
        
        # Initialize fusion module
        modality_dims = {
            modality: encoder.get_output_dim()
            for modality, encoder in self.encoders.items()
        }
        
        self.fusion = MultiModalFusion(
            modality_dims=modality_dims,
            fusion_dim=fusion_config['fusion_dim'],
            fusion_strategy=fusion_config['strategy']
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_config['fusion_dim'], fusion_config['fusion_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_config['fusion_dim'] // 2, num_classes)
        )
    
    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Forward pass through multi-modal model."""
        # Encode each modality
        encoded_features = {}
        for modality, encoder in self.encoders.items():
            if modality in inputs:
                encoded_features[modality] = encoder.encode(inputs[modality])
        
        # Fuse modalities
        fused_features = self.fusion(encoded_features)
        
        # Classification
        logits = self.classifier(fused_features)
        return logits
    
    def get_modality_features(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Get individual modality features without fusion."""
        encoded_features = {}
        for modality, encoder in self.encoders.items():
            if modality in inputs:
                encoded_features[modality] = encoder.encode(inputs[modality])
        return encoded_features

class MultiModalTrainer:
    """Trainer for multi-modal models with advanced techniques."""
    
    def __init__(self, model: MultiModalModel, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.modality_weights = {}
        
    def train_with_modality_dropout(self, dataloader, optimizer, 
                                  dropout_prob: float = 0.1):
        """Train with modality dropout for robustness."""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Apply modality dropout
            inputs = self._apply_modality_dropout(batch['inputs'], dropout_prob)
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            logits = self.model(inputs)
            loss = F.cross_entropy(logits, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _apply_modality_dropout(self, inputs: Dict[str, Any], 
                              dropout_prob: float) -> Dict[str, Any]:
        """Randomly drop modalities during training."""
        if not self.training:
            return inputs
        
        dropped_inputs = {}
        for modality, data in inputs.items():
            if torch.rand(1).item() > dropout_prob:
                dropped_inputs[modality] = data.to(self.device)
        
        # Ensure at least one modality remains
        if not dropped_inputs:
            modality = list(inputs.keys())[0]
            dropped_inputs[modality] = inputs[modality].to(self.device)
        
        return dropped_inputs
    
    def compute_modality_importance(self, dataloader) -> Dict[str, float]:
        """Compute importance of each modality using gradient analysis."""
        self.model.eval()
        modality_gradients = {modality: 0.0 for modality in self.model.encoders.keys()}
        
        for batch in dataloader:
            inputs = {k: v.to(self.device) for k, v in batch['inputs'].items()}
            targets = batch['targets'].to(self.device)
            
            # Get modality features
            modality_features = self.model.get_modality_features(inputs)
            
            for modality, features in modality_features.items():
                features.requires_grad_(True)
                
                # Forward through fusion and classifier
                fused = self.model.fusion({modality: features})
                logits = self.model.classifier(fused)
                loss = F.cross_entropy(logits, targets)
                
                # Compute gradients
                grad = torch.autograd.grad(loss, features, retain_graph=True)[0]
                modality_gradients[modality] += grad.abs().mean().item()
        
        # Normalize
        total_gradient = sum(modality_gradients.values())
        return {k: v / total_gradient for k, v in modality_gradients.items()}

# Example usage and factory functions
def create_vision_text_model(num_classes: int = 10) -> MultiModalModel:
    """Create a vision-text multi-modal model."""
    vision_config = ModalityConfig(
        modality_type='vision',
        input_dim=224*224*3,
        hidden_dim=512,
        output_dim=256,
        preprocessing={'resize': (224, 224), 'normalize': True},
        model_config={'encoder_type': 'resnet'}
    )
    
    text_config = ModalityConfig(
        modality_type='text',
        input_dim=512,
        hidden_dim=768,
        output_dim=256,
        preprocessing={'max_length': 512},
        model_config={'model_name': 'bert-base-uncased', 'freeze_encoder': True}
    )
    
    fusion_config = {
        'fusion_dim': 256,
        'strategy': 'attention'
    }
    
    return MultiModalModel(
        modality_configs={'vision': vision_config, 'text': text_config},
        fusion_config=fusion_config,
        num_classes=num_classes
    )

def demo_multimodal_learning():
    """Demonstrate multi-modal learning capabilities."""
    # Create model
    model = create_vision_text_model(num_classes=10)
    
    # Create dummy inputs
    batch_size = 4
    vision_input = torch.randn(batch_size, 3, 224, 224)
    text_input = ["sample text"] * batch_size
    
    inputs = {
        'vision': vision_input,
        'text': text_input
    }
    
    # Forward pass
    with torch.no_grad():
        logits = model(inputs)
        print(f"Output logits shape: {logits.shape}")
        
        # Get individual modality features
        features = model.get_modality_features(inputs)
        for modality, feat in features.items():
            print(f"{modality} features shape: {feat.shape}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_multimodal_learning()
