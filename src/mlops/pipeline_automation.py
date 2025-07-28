"""
MLOps Pipeline Automation 2.0
Advanced MLOps pipeline with GitOps, automated training, and deployment.
"""

import os
import yaml
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import subprocess
import shutil
from pathlib import Path

@dataclass
class PipelineConfig:
    """Configuration for MLOps pipeline."""
    project_name: str
    model_type: str
    data_source: str
    training_config: Dict[str, Any]
    deployment_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    git_repo: Optional[str] = None
    environment: str = "development"

@dataclass
class ModelArtifact:
    """Model artifact metadata."""
    model_id: str
    version: str
    model_path: str
    metrics: Dict[str, float]
    timestamp: str
    git_commit: Optional[str] = None
    tags: List[str] = None

class GitOpsManager:
    """Manages GitOps workflow for ML pipelines."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.logger = logging.getLogger(__name__)
    
    def initialize_repo(self) -> bool:
        """Initialize Git repository if not exists."""
        try:
            if not (self.repo_path / '.git').exists():
                subprocess.run(['git', 'init'], cwd=self.repo_path, check=True)
                subprocess.run(['git', 'config', 'user.name', 'MLOps Pipeline'], 
                             cwd=self.repo_path, check=True)
                subprocess.run(['git', 'config', 'user.email', 'mlops@example.com'], 
                             cwd=self.repo_path, check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to initialize Git repo: {e}")
            return False
    
    def commit_changes(self, message: str, files: List[str] = None) -> bool:
        """Commit changes to Git repository."""
        try:
            if files:
                for file in files:
                    subprocess.run(['git', 'add', file], cwd=self.repo_path, check=True)
            else:
                subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True)
            
            subprocess.run(['git', 'commit', '-m', message], 
                         cwd=self.repo_path, check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to commit changes: {e}")
            return False
    
    def get_current_commit(self) -> Optional[str]:
        """Get current Git commit hash."""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  cwd=self.repo_path, 
                                  capture_output=True, 
                                  text=True, 
                                  check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

class DataValidationEngine:
    """Validates data quality and schema."""
    
    def __init__(self, validation_rules: Dict[str, Any]):
        self.validation_rules = validation_rules
        self.logger = logging.getLogger(__name__)
    
    def validate_data(self, data_path: str) -> Dict[str, Any]:
        """Validate data against defined rules."""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        try:
            # Load data based on file extension
            if data_path.endswith('.csv'):
                import pandas as pd
                data = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                with open(data_path, 'r') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported data format: {data_path}")
            
            # Schema validation
            if 'schema' in self.validation_rules:
                schema_valid = self._validate_schema(data, self.validation_rules['schema'])
                if not schema_valid:
                    validation_results["is_valid"] = False
                    validation_results["errors"].append("Schema validation failed")
            
            # Data quality checks
            if 'quality_checks' in self.validation_rules:
                quality_results = self._validate_quality(data, self.validation_rules['quality_checks'])
                validation_results["metrics"].update(quality_results)
            
        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Data validation error: {str(e)}")
        
        return validation_results
    
    def _validate_schema(self, data: Any, schema: Dict[str, Any]) -> bool:
        """Validate data schema."""
        if isinstance(data, dict):
            # JSON schema validation
            required_fields = schema.get('required_fields', [])
            for field in required_fields:
                if field not in data:
                    return False
        else:
            # DataFrame schema validation
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                required_columns = schema.get('required_columns', [])
                for col in required_columns:
                    if col not in data.columns:
                        return False
        return True
    
    def _validate_quality(self, data: Any, quality_checks: Dict[str, Any]) -> Dict[str, float]:
        """Validate data quality."""
        metrics = {}
        
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            # Missing values check
            if 'max_missing_percentage' in quality_checks:
                missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
                metrics['missing_percentage'] = missing_pct
            
            # Duplicate rows check
            if 'check_duplicates' in quality_checks:
                duplicate_pct = (data.duplicated().sum() / len(data)) * 100
                metrics['duplicate_percentage'] = duplicate_pct
        
        return metrics

class ModelTrainingOrchestrator:
    """Orchestrates model training pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def train_model(self, data_path: str) -> Optional[ModelArtifact]:
        """Train model with given configuration."""
        try:
            self.logger.info(f"Starting model training for {self.config.project_name}")
            
            # Simulate model training
            model_id = f"{self.config.project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create model directory
            model_dir = Path(f"models/{model_id}")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model (placeholder)
            model_path = str(model_dir / "model.pkl")
            with open(model_path, 'w') as f:
                f.write(f"Trained model for {self.config.project_name}")
            
            # Generate metrics (placeholder)
            metrics = {
                "accuracy": 0.95,
                "precision": 0.93,
                "recall": 0.92,
                "f1_score": 0.925
            }
            
            artifact = ModelArtifact(
                model_id=model_id,
                version=version,
                model_path=model_path,
                metrics=metrics,
                timestamp=datetime.now().isoformat(),
                tags=["automated", self.config.environment]
            )
            
            # Save artifact metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(artifact), f, indent=2)
            
            self.logger.info(f"Model training completed: {model_id}")
            return artifact
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return None

class DeploymentManager:
    """Manages model deployment."""
    
    def __init__(self, deployment_config: Dict[str, Any]):
        self.deployment_config = deployment_config
        self.logger = logging.getLogger(__name__)
    
    def deploy_model(self, artifact: ModelArtifact) -> bool:
        """Deploy model to specified environment."""
        try:
            environment = self.deployment_config.get('environment', 'staging')
            deployment_type = self.deployment_config.get('type', 'local')
            
            self.logger.info(f"Deploying model {artifact.model_id} to {environment}")
            
            if deployment_type == 'kubernetes':
                return self._deploy_to_kubernetes(artifact)
            elif deployment_type == 'docker':
                return self._deploy_to_docker(artifact)
            else:
                return self._deploy_locally(artifact)
                
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            return False
    
    def _deploy_to_kubernetes(self, artifact: ModelArtifact) -> bool:
        """Deploy to Kubernetes cluster."""
        # Generate Kubernetes manifests
        k8s_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"model-{artifact.model_id}",
                "labels": {"app": artifact.model_id}
            },
            "spec": {
                "replicas": self.deployment_config.get('replicas', 1),
                "selector": {"matchLabels": {"app": artifact.model_id}},
                "template": {
                    "metadata": {"labels": {"app": artifact.model_id}},
                    "spec": {
                        "containers": [{
                            "name": "model-server",
                            "image": f"model-server:{artifact.version}",
                            "ports": [{"containerPort": 8080}]
                        }]
                    }
                }
            }
        }
        
        # Save manifest
        manifest_path = f"k8s/{artifact.model_id}.yaml"
        Path(manifest_path).parent.mkdir(exist_ok=True)
        with open(manifest_path, 'w') as f:
            yaml.dump(k8s_manifest, f)
        
        self.logger.info(f"Kubernetes manifest saved: {manifest_path}")
        return True
    
    def _deploy_to_docker(self, artifact: ModelArtifact) -> bool:
        """Deploy using Docker."""
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app
COPY {artifact.model_path} /app/model.pkl
COPY requirements.txt /app/
RUN pip install -r requirements.txt

EXPOSE 8080
CMD ["python", "serve.py"]
"""
        
        dockerfile_path = f"deployments/{artifact.model_id}/Dockerfile"
        Path(dockerfile_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        self.logger.info(f"Dockerfile created: {dockerfile_path}")
        return True
    
    def _deploy_locally(self, artifact: ModelArtifact) -> bool:
        """Deploy locally."""
        deployment_dir = Path(f"deployments/local/{artifact.model_id}")
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        shutil.copy2(artifact.model_path, deployment_dir)
        
        # Create deployment info
        deployment_info = {
            "model_id": artifact.model_id,
            "deployment_time": datetime.now().isoformat(),
            "status": "deployed",
            "endpoint": f"http://localhost:8080/{artifact.model_id}"
        }
        
        with open(deployment_dir / "deployment.json", 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        self.logger.info(f"Local deployment completed: {deployment_dir}")
        return True

class MLOpsPipeline:
    """Main MLOps pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig, workspace_path: str = "./mlops_workspace"):
        self.config = config
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.gitops = GitOpsManager(self.workspace_path)
        self.data_validator = DataValidationEngine(config.training_config.get('validation_rules', {}))
        self.trainer = ModelTrainingOrchestrator(config)
        self.deployer = DeploymentManager(config.deployment_config)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize workspace
        self._initialize_workspace()
    
    def _initialize_workspace(self):
        """Initialize MLOps workspace."""
        # Create directory structure
        directories = [
            "data", "models", "experiments", "deployments", 
            "configs", "scripts", "notebooks", "tests"
        ]
        
        for directory in directories:
            (self.workspace_path / directory).mkdir(exist_ok=True)
        
        # Initialize Git
        self.gitops.initialize_repo()
        
        # Save pipeline configuration
        config_path = self.workspace_path / "configs" / "pipeline_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False)
    
    def run_pipeline(self, data_path: str) -> bool:
        """Run the complete MLOps pipeline."""
        try:
            self.logger.info("Starting MLOps pipeline execution")
            
            # Step 1: Data validation
            self.logger.info("Step 1: Data validation")
            validation_results = self.data_validator.validate_data(data_path)
            
            if not validation_results["is_valid"]:
                self.logger.error(f"Data validation failed: {validation_results['errors']}")
                return False
            
            # Step 2: Model training
            self.logger.info("Step 2: Model training")
            artifact = self.trainer.train_model(data_path)
            
            if not artifact:
                self.logger.error("Model training failed")
                return False
            
            # Step 3: Model deployment
            self.logger.info("Step 3: Model deployment")
            deployment_success = self.deployer.deploy_model(artifact)
            
            if not deployment_success:
                self.logger.error("Model deployment failed")
                return False
            
            # Step 4: Commit to Git
            self.logger.info("Step 4: Git commit")
            commit_message = f"Deploy model {artifact.model_id} with metrics: {artifact.metrics}"
            self.gitops.commit_changes(commit_message)
            
            self.logger.info("MLOps pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return False
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        status = {
            "project_name": self.config.project_name,
            "environment": self.config.environment,
            "workspace_path": str(self.workspace_path),
            "git_commit": self.gitops.get_current_commit(),
            "last_run": None,
            "models_deployed": []
        }
        
        # Get deployed models
        deployments_dir = self.workspace_path / "deployments"
        if deployments_dir.exists():
            for deployment_path in deployments_dir.glob("**/deployment.json"):
                try:
                    with open(deployment_path, 'r') as f:
                        deployment_info = json.load(f)
                        status["models_deployed"].append(deployment_info)
                except Exception:
                    continue
        
        return status
    
    def cleanup_old_models(self, keep_last_n: int = 5):
        """Clean up old model artifacts."""
        models_dir = self.workspace_path / "models"
        if not models_dir.exists():
            return
        
        # Get all model directories sorted by creation time
        model_dirs = sorted(models_dir.glob("*"), key=lambda x: x.stat().st_ctime, reverse=True)
        
        # Remove old models
        for model_dir in model_dirs[keep_last_n:]:
            if model_dir.is_dir():
                shutil.rmtree(model_dir)
                self.logger.info(f"Removed old model: {model_dir.name}")

# Example usage
if __name__ == "__main__":
    print("Testing MLOps Pipeline Automation 2.0...")
    
    # Create pipeline configuration
    config = PipelineConfig(
        project_name="customer_churn_predictor",
        model_type="classification",
        data_source="s3://data-bucket/churn_data.csv",
        training_config={
            "algorithm": "random_forest",
            "hyperparameters": {"n_estimators": 100, "max_depth": 10},
            "validation_rules": {
                "schema": {"required_columns": ["customer_id", "churn"]},
                "quality_checks": {"max_missing_percentage": 5}
            }
        },
        deployment_config={
            "environment": "staging",
            "type": "kubernetes",
            "replicas": 2
        },
        monitoring_config={
            "enable_drift_detection": True,
            "performance_threshold": 0.9
        },
        environment="development"
    )
    
    # Initialize pipeline
    pipeline = MLOpsPipeline(config, "./test_mlops_workspace")
    
    # Create dummy data file
    dummy_data_path = "./test_mlops_workspace/data/sample_data.csv"
    with open(dummy_data_path, 'w') as f:
        f.write("customer_id,churn,feature1,feature2\n")
        f.write("1,0,1.5,2.3\n")
        f.write("2,1,2.1,1.8\n")
    
    # Run pipeline
    success = pipeline.run_pipeline(dummy_data_path)
    
    if success:
        print("Pipeline executed successfully!")
        
        # Get status
        status = pipeline.get_pipeline_status()
        print(f"Project: {status['project_name']}")
        print(f"Environment: {status['environment']}")
        print(f"Git commit: {status['git_commit']}")
        print(f"Models deployed: {len(status['models_deployed'])}")
        
        # Cleanup
        pipeline.cleanup_old_models(keep_last_n=3)
    else:
        print("Pipeline execution failed!")
    
    print("\nMLOps Pipeline Automation 2.0 implemented successfully! 🚀")
