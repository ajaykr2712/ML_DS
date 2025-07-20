# 🤖 Generative AI Project

A comprehensive implementation of various generative AI models including GPT, VAE, GAN, and Diffusion models.

## 🎯 Overview

This project provides production-ready implementations of state-of-the-art generative models with clear examples, extensive documentation, and modular architecture.

### 🚀 Features

- **GPT (Generative Pre-trained Transformer)**: Text generation with attention mechanisms
- **VAE (Variational Autoencoder)**: Latent space modeling for data generation
- **GAN (Generative Adversarial Network)**: Adversarial training for realistic data synthesis
- **Diffusion Models**: State-of-the-art image and data generation

## 📁 Project Structure

```
gen_ai_project/
├── config/                 # Configuration files
│   ├── gpt_config.yaml    # GPT model configuration
│   ├── vae_config.yaml    # VAE model configuration
│   ├── gan_config.yaml    # GAN model configuration
│   └── diffusion_config.yaml
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── training/          # Training scripts
│   └── evaluation/        # Evaluation utilities
├── data/                   # Dataset storage
├── notebooks/             # Jupyter notebooks
├── examples/              # Usage examples
└── requirements.txt       # Dependencies
```

## 🛠️ Installation

```bash
# Clone the repository
cd gen_ai_project

# Install dependencies
pip install -r requirements.txt

# Run setup
python setup.py develop
```

## 🚀 Quick Start

### Text Generation with GPT
```python
from src.models.gpt import GPTModel
from src.training.gpt_trainer import GPTTrainer

# Load configuration
config = load_config('config/gpt_config.yaml')

# Initialize model
model = GPTModel(config)

# Train model
trainer = GPTTrainer(model, config)
trainer.train(dataset)

# Generate text
generated_text = model.generate("The future of AI is")
print(generated_text)
```

### Image Generation with Diffusion
```python
from src.models.diffusion import DiffusionModel

# Initialize model
model = DiffusionModel.from_pretrained('config/diffusion_config.yaml')

# Generate images
images = model.sample(num_samples=4)
```

## 📊 Model Architectures

### GPT (Generative Pre-trained Transformer)
- Multi-head self-attention
- Position encoding
- Layer normalization
- Residual connections

### VAE (Variational Autoencoder)
- Encoder-decoder architecture
- Latent space sampling
- KL divergence regularization
- Reconstruction loss

### GAN (Generative Adversarial Network)
- Generator network
- Discriminator network
- Adversarial training
- Progressive training support

### Diffusion Models
- Forward diffusion process
- Reverse diffusion process
- U-Net architecture
- Noise scheduling

## 🎓 Examples

Check out the `examples/` directory for detailed usage examples:
- `text_generation_example.py`: Complete GPT implementation
- `image_generation_example.py`: VAE and GAN examples
- `diffusion_example.py`: Diffusion model usage

## 📈 Training

Each model comes with optimized training scripts:

```bash
# Train GPT model
python src/training/train_gpt.py --config config/gpt_config.yaml

# Train VAE model
python src/training/train_vae.py --config config/vae_config.yaml

# Train GAN model
python src/training/train_gan.py --config config/gan_config.yaml

# Train Diffusion model
python src/training/train_diffusion.py --config config/diffusion_config.yaml
```

## 🧪 Evaluation

Comprehensive evaluation metrics for each model type:

- **Text Generation**: Perplexity, BLEU score, Human evaluation
- **Image Generation**: FID, IS, LPIPS
- **General**: Reconstruction loss, Sampling quality

## 🔧 Configuration

Each model has detailed configuration files in the `config/` directory. Customize:
- Model architecture parameters
- Training hyperparameters
- Data preprocessing settings
- Evaluation metrics

## 📚 Documentation

- [Model Architecture Details](docs/architectures.md)
- [Training Guide](docs/training.md)
- [API Reference](docs/api.md)
- [Best Practices](docs/best_practices.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Attention Is All You Need (Transformer paper)
- Auto-Encoding Variational Bayes (VAE paper)
- Generative Adversarial Networks (GAN paper)
- Denoising Diffusion Probabilistic Models (DDPM paper)

---

**Built with ❤️ for the AI community**