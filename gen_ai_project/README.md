# Generative AI Project

## Overview

A comprehensive generative AI platform featuring multiple model implementations, training pipelines, and deployment strategies. This project demonstrates state-of-the-art generative models including GANs, VAEs, Transformers, and Diffusion models.

## Features

- **Multiple Model Architectures**
  - Generative Adversarial Networks (GANs)
  - Variational Autoencoders (VAEs)
  - Transformer-based Language Models
  - Diffusion Models
  - Autoregressive Models

- **Training Infrastructure**
  - Distributed training support
  - Mixed precision training
  - Gradient accumulation
  - Learning rate scheduling
  - Model checkpointing

- **Evaluation Framework**
  - Inception Score (IS)
  - Fréchet Inception Distance (FID)
  - BLEU scores for text generation
  - Perplexity metrics
  - Human evaluation protocols

- **Deployment Options**
  - REST API endpoints
  - Gradio web interface
  - Docker containerization
  - Cloud deployment configurations

## Project Structure

```
gen_ai_project/
├── config/              # Configuration files
├── data/               # Dataset management
├── examples/           # Usage examples
├── notebooks/          # Jupyter notebooks
├── src/               # Source code
│   ├── models/        # Model implementations
│   ├── training/      # Training scripts
│   ├── evaluation/    # Evaluation metrics
│   ├── utils/         # Utility functions
│   └── deployment/    # Deployment code
├── tests/             # Unit tests
├── requirements.txt   # Dependencies
├── setup.py          # Package setup
└── Dockerfile        # Container configuration
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd gen_ai_project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package**
   ```bash
   pip install -e .
   ```

## Quick Start

### Text Generation with GPT-like Model

```python
from src.models.transformer import GPTModel
from src.training.trainer import ModelTrainer

# Initialize model
model = GPTModel(vocab_size=50257, d_model=768, num_layers=12)

# Train model
trainer = ModelTrainer(model, config_path="config/gpt_config.yaml")
trainer.train(train_dataset, val_dataset)

# Generate text
generated_text = model.generate("The future of AI is", max_length=100)
print(generated_text)
```

### Image Generation with GAN

```python
from src.models.gan import StyleGAN
from src.training.gan_trainer import GANTrainer

# Initialize GAN
gan = StyleGAN(latent_dim=512, image_size=256)

# Train GAN
trainer = GANTrainer(gan, config_path="config/stylegan_config.yaml")
trainer.train(image_dataset)

# Generate images
generated_images = gan.generate(batch_size=16)
```

## Configuration

All model configurations are stored in the `config/` directory. Key configuration files:

- `gpt_config.yaml`: GPT model settings
- `vae_config.yaml`: VAE training parameters
- `gan_config.yaml`: GAN architecture settings
- `diffusion_config.yaml`: Diffusion model configuration

## Training

### Single GPU Training

```bash
python src/training/train.py --config config/model_config.yaml --model gpt
```

### Multi-GPU Training

```bash
python -m torch.distributed.launch --nproc_per_node=4 src/training/train.py \
    --config config/model_config.yaml --model gpt --distributed
```

## Evaluation

```bash
python src/evaluation/evaluate.py --model_path checkpoints/best_model.pt \
    --eval_dataset data/test_set.json --metrics fid,is,bleu
```

## Docker Deployment

```bash
# Build image
docker build -t gen-ai-app .

# Run container
docker run -p 8000:8000 gen-ai-app
```

## API Usage

Once deployed, the API provides endpoints for generation:

```bash
# Text generation
curl -X POST http://localhost:8000/generate/text \
    -H "Content-Type: application/json" \
    -d '{"prompt": "The future of AI", "max_length": 100}'

# Image generation
curl -X POST http://localhost:8000/generate/image \
    -H "Content-Type: application/json" \
    -d '{"style": "realistic", "size": 512}'
```

## Examples

See the `examples/` directory for detailed usage examples:

- `text_generation_example.py`: Complete text generation pipeline
- `image_generation_example.py`: Image synthesis with GANs
- `fine_tuning_example.py`: Fine-tuning pre-trained models
- `evaluation_example.py`: Model evaluation workflows

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{gen_ai_project,
  title={Generative AI Project: A Comprehensive Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/gen_ai_project}
}
```

## Support

For questions and support, please open an issue on the GitHub repository.
