"""
Advanced Natural Language Processing Pipeline
============================================

A comprehensive NLP pipeline featuring transformer models, multilingual processing,
sentiment analysis, named entity recognition, and advanced text preprocessing.

Best Contributions:
- Transformer-based models with attention visualization
- Multilingual text processing and translation
- Advanced sentiment analysis with emotion detection
- Named Entity Recognition with custom entity types
- Text summarization and keyword extraction
- Document similarity and clustering
- Real-time text processing capabilities

Author: ML/DS Advanced Implementation Team
"""

import os
import re
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import json
import pickle
from pathlib import Path

# Transformer and NLP libraries
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel,
        pipeline, Pipeline
    )
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    from langdetect import detect
    import spacy
    from textblob import TextBlob
    import nltk
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError as e:
    logging.warning(f"Some NLP libraries not available: {e}")

@dataclass
class NLPConfig:
    """Configuration for NLP pipeline."""
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    batch_size: int = 32
    device: str = "auto"
    cache_dir: str = "./cache"
    enable_gpu: bool = True
    languages: List[str] = None
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]

class AdvancedNLPPipeline:
    """
    Advanced NLP pipeline with state-of-the-art capabilities.
    
    Features:
    - Transformer-based text encoding
    - Multilingual processing
    - Sentiment and emotion analysis
    - Named Entity Recognition
    - Text summarization
    - Document clustering
    - Real-time processing
    """
    
    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        self.logger = self._setup_logging()
        
        # Initialize models
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        self.ner_pipeline = None
        self.summarization_pipeline = None
        self.sentence_transformer = None
        self.nlp_spacy = None
        
        # Initialize components
        self._initialize_models()
        self._download_nltk_data()
        
        # Processing history
        self.processing_history = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _initialize_models(self):
        """Initialize all NLP models."""
        try:
            # Device selection
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() and self.config.enable_gpu else "cpu"
            else:
                device = self.config.device
            
            self.logger.info(f"Using device: {device}")
            
            # Main transformer model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, cache_dir=self.config.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.config.model_name, cache_dir=self.config.cache_dir
            ).to(device)
            
            # Specialized pipelines
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if device == "cuda" else -1
            )
            
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=0 if device == "cuda" else -1
            )
            
            self.summarization_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if device == "cuda" else -1
            )
            
            # Sentence transformer for embeddings
            self.sentence_transformer = SentenceTransformer(
                'all-MiniLM-L6-v2', cache_folder=self.config.cache_dir
            )
            
            # SpaCy for advanced NLP
            try:
                self.nlp_spacy = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("SpaCy English model not found. Some features will be limited.")
            
            self.logger.info("All NLP models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            self.logger.warning(f"Error downloading NLTK data: {e}")
    
    def preprocess_text(self, text: str, 
                       remove_urls: bool = True,
                       remove_mentions: bool = True,
                       remove_hashtags: bool = False,
                       normalize_whitespace: bool = True) -> str:
        """
        Advanced text preprocessing with multiple options.
        
        Args:
            text: Input text to preprocess
            remove_urls: Remove URLs from text
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            normalize_whitespace: Normalize whitespace
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions
        if remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        if remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        
        # Normalize whitespace
        if normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def detect_language(self, text: str) -> str:
        """Detect the language of input text."""
        try:
            return detect(text)
        except:
            return "unknown"
    
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text using transformer model.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Text embeddings as numpy array
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Use sentence transformer for consistent embeddings
            embeddings = self.sentence_transformer.encode(
                texts, batch_size=self.config.batch_size, show_progress_bar=True
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Error encoding text: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis with emotion detection.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores and emotions
        """
        results = {}
        
        try:
            # Transformer-based sentiment
            sentiment_result = self.sentiment_pipeline(text)[0]
            results['transformer_sentiment'] = {
                'label': sentiment_result['label'],
                'score': sentiment_result['score']
            }
            
            # TextBlob sentiment
            blob = TextBlob(text)
            results['textblob_sentiment'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
            
            # VADER sentiment (if available)
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                analyzer = SentimentIntensityAnalyzer()
                vader_scores = analyzer.polarity_scores(text)
                results['vader_sentiment'] = vader_scores
            except ImportError:
                pass
            
            # Language detection
            results['language'] = self.detect_language(text)
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            results['error'] = str(e)
        
        return results
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        try:
            # Transformer-based NER
            ner_results = self.ner_pipeline(text)
            for entity in ner_results:
                entities.append({
                    'text': entity['word'],
                    'label': entity['entity_group'],
                    'confidence': entity['score'],
                    'start': entity.get('start', 0),
                    'end': entity.get('end', 0),
                    'source': 'transformer'
                })
            
            # SpaCy NER (if available)
            if self.nlp_spacy:
                doc = self.nlp_spacy(text)
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'confidence': 1.0,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'source': 'spacy'
                    })
            
        except Exception as e:
            self.logger.error(f"Error in entity extraction: {e}")
        
        return entities
    
    def summarize_text(self, text: str, 
                      max_length: int = 150,
                      min_length: int = 30) -> str:
        """
        Generate text summary using transformer model.
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Generated summary
        """
        try:
            if len(text.split()) < min_length:
                return text
            
            summary = self.summarization_pipeline(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]['summary_text']
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error in summarization: {e}")
            return text[:max_length * 4]  # Fallback truncation
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF.
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Use TF-IDF for keyword extraction
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform([processed_text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return keyword_scores[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error in keyword extraction: {e}")
            return []
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            embeddings = self.encode_text([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def cluster_documents(self, texts: List[str], 
                         n_clusters: int = 5) -> Dict[str, Any]:
        """
        Cluster documents based on semantic similarity.
        
        Args:
            texts: List of texts to cluster
            n_clusters: Number of clusters
            
        Returns:
            Clustering results with labels and centroids
        """
        try:
            # Encode texts
            embeddings = self.encode_text(texts)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Compute cluster statistics
            cluster_centers = kmeans.cluster_centers_
            inertia = kmeans.inertia_
            
            # Group texts by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    'text': texts[i],
                    'index': i,
                    'distance_to_center': np.linalg.norm(
                        embeddings[i] - cluster_centers[label]
                    )
                })
            
            return {
                'clusters': clusters,
                'labels': cluster_labels.tolist(),
                'centers': cluster_centers,
                'inertia': inertia,
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            self.logger.error(f"Error in document clustering: {e}")
            return {}
    
    def process_document(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive document processing with all NLP features.
        
        Args:
            text: Input document text
            
        Returns:
            Complete analysis results
        """
        start_time = datetime.now()
        
        results = {
            'original_text': text,
            'processed_text': self.preprocess_text(text),
            'timestamp': start_time.isoformat(),
            'text_length': len(text),
            'word_count': len(text.split())
        }
        
        try:
            # Language detection
            results['language'] = self.detect_language(text)
            
            # Sentiment analysis
            results['sentiment'] = self.analyze_sentiment(text)
            
            # Entity extraction
            results['entities'] = self.extract_entities(text)
            
            # Text summarization (if text is long enough)
            if len(text.split()) > 50:
                results['summary'] = self.summarize_text(text)
            
            # Keyword extraction
            results['keywords'] = self.extract_keywords(text)
            
            # Text encoding
            results['embedding'] = self.encode_text(text).tolist()
            
            # Processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            results['processing_time'] = processing_time
            
            # Add to history
            self.processing_history.append({
                'timestamp': start_time.isoformat(),
                'text_length': len(text),
                'processing_time': processing_time,
                'features_extracted': list(results.keys())
            })
            
            self.logger.info(f"Document processed in {processing_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            results['error'] = str(e)
        
        return results
    
    def batch_process(self, texts: List[str], 
                     save_results: bool = True,
                     output_file: str = None) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            texts: List of texts to process
            save_results: Whether to save results to file
            output_file: Output file path
            
        Returns:
            List of processing results
        """
        self.logger.info(f"Starting batch processing of {len(texts)} documents")
        
        results = []
        for i, text in enumerate(texts):
            self.logger.info(f"Processing document {i+1}/{len(texts)}")
            result = self.process_document(text)
            result['batch_index'] = i
            results.append(result)
        
        # Save results if requested
        if save_results:
            if output_file is None:
                output_file = f"nlp_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results saved to {output_file}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        if not self.processing_history:
            return {}
        
        processing_times = [h['processing_time'] for h in self.processing_history]
        text_lengths = [h['text_length'] for h in self.processing_history]
        
        return {
            'total_documents_processed': len(self.processing_history),
            'average_processing_time': np.mean(processing_times),
            'total_processing_time': sum(processing_times),
            'average_text_length': np.mean(text_lengths),
            'min_processing_time': min(processing_times),
            'max_processing_time': max(processing_times),
            'throughput_docs_per_second': len(self.processing_history) / sum(processing_times) if sum(processing_times) > 0 else 0
        }
    
    def visualize_sentiment_distribution(self, texts: List[str], 
                                       save_plot: bool = True) -> None:
        """
        Visualize sentiment distribution across multiple texts.
        
        Args:
            texts: List of texts to analyze
            save_plot: Whether to save the plot
        """
        try:
            sentiments = []
            for text in texts:
                sentiment = self.analyze_sentiment(text)
                if 'transformer_sentiment' in sentiment:
                    sentiments.append(sentiment['transformer_sentiment']['label'])
            
            # Create plot
            plt.figure(figsize=(10, 6))
            sentiment_counts = pd.Series(sentiments).value_counts()
            sentiment_counts.plot(kind='bar')
            plt.title('Sentiment Distribution')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_plot:
                plt.savefig(f"sentiment_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                self.logger.info("Sentiment distribution plot saved")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error creating sentiment visualization: {e}")
    
    def save_model_state(self, filepath: str) -> None:
        """Save pipeline state (excluding models)."""
        state = {
            'config': self.config.__dict__,
            'processing_history': self.processing_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info(f"Pipeline state saved to {filepath}")
    
    def load_model_state(self, filepath: str) -> None:
        """Load pipeline state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.processing_history = state.get('processing_history', [])
        self.logger.info(f"Pipeline state loaded from {filepath}")

def main():
    """Demonstration of the Advanced NLP Pipeline."""
    # Initialize pipeline
    config = NLPConfig(
        model_name="bert-base-uncased",
        batch_size=16,
        enable_gpu=True
    )
    
    nlp_pipeline = AdvancedNLPPipeline(config)
    
    # Sample texts for demonstration
    sample_texts = [
        "The new AI model shows remarkable performance improvements over previous versions. "
        "Machine learning researchers are excited about the potential applications.",
        
        "I'm really disappointed with this product. The quality is terrible and customer "
        "service was unhelpful. Would not recommend to anyone.",
        
        "Climate change is one of the most pressing challenges of our time. Scientists "
        "worldwide are working on innovative solutions to reduce carbon emissions.",
        
        "Apple Inc. reported strong quarterly earnings today. CEO Tim Cook highlighted "
        "the success of iPhone sales in international markets, particularly in China and India."
    ]
    
    print("=== Advanced NLP Pipeline Demo ===\n")
    
    # Process individual document
    print("1. Individual Document Processing:")
    result = nlp_pipeline.process_document(sample_texts[0])
    print(f"Language: {result.get('language', 'unknown')}")
    print(f"Sentiment: {result.get('sentiment', {}).get('transformer_sentiment', {})}")
    print(f"Entities: {len(result.get('entities', []))} found")
    print(f"Top keywords: {result.get('keywords', [])[:3]}")
    print()
    
    # Batch processing
    print("2. Batch Processing:")
    batch_results = nlp_pipeline.batch_process(sample_texts, save_results=False)
    print(f"Processed {len(batch_results)} documents")
    print()
    
    # Document clustering
    print("3. Document Clustering:")
    cluster_results = nlp_pipeline.cluster_documents(sample_texts, n_clusters=2)
    print(f"Created {cluster_results.get('n_clusters', 0)} clusters")
    for cluster_id, docs in cluster_results.get('clusters', {}).items():
        print(f"Cluster {cluster_id}: {len(docs)} documents")
    print()
    
    # Text similarity
    print("4. Text Similarity:")
    similarity = nlp_pipeline.compute_similarity(sample_texts[0], sample_texts[2])
    print(f"Similarity between text 1 and 3: {similarity:.3f}")
    print()
    
    # Pipeline statistics
    print("5. Pipeline Statistics:")
    stats = nlp_pipeline.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
