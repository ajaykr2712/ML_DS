"""Setup script for the Generative AI Project."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements

setup(
    name="gen-ai-project",
    version="1.0.0",
    author="ML/DS Team",
    author_email="team@example.com",
    description="A comprehensive generative AI framework with multiple model implementations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/gen_ai_project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.5.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "gpu": [
            "triton>=2.0.0",
            "xformers>=0.0.20",
        ],
        "all": [
            "pytest>=7.4.0",
            "black>=23.5.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
            "triton>=2.0.0",
            "xformers>=0.0.20",
        ]
    },
    entry_points={
        "console_scripts": [
            "gen-ai-train=src.training.cli:main",
            "gen-ai-generate=src.generation.cli:main",
            "gen-ai-evaluate=src.evaluation.cli:main",
        ],
    },
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    include_package_data=True,
    zip_safe=False,
)
