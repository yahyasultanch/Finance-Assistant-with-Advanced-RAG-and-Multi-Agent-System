"""Fine-tune FinBERT on Financial PhraseBank."""
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinBERTFineTuner:
    """Fine-tuner for FinBERT on Financial PhraseBank."""

    def __init__(
        self,
        base_model: str = "ProsusAI/finbert",
        output_dir: str = "./models/finetuned_finbert",
        max_epochs: int = 3
    ):
        """
        Initialize fine-tuner.
        
        Args:
            base_model: Base FinBERT model
            output_dir: Output directory for fine-tuned model
            max_epochs: Maximum training epochs
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.max_epochs = max_epochs
        self.tokenizer = None
        self.model = None

    def load_data(self):
        """Load and prepare Financial PhraseBank dataset."""
        logger.info("Loading Financial PhraseBank dataset...")
        
        # Load dataset (all-agree subset for higher quality)
        dataset = load_dataset("financial_phrasebank", "sentences_allagree")
        
        # Split into train/validation/test
        train_test = dataset["train"].train_test_split(test_size=0.3, seed=42)
        test_val = train_test["test"].train_test_split(test_size=0.5, seed=42)
        
        train_dataset = train_test["train"]
        val_dataset = test_val["train"]
        test_dataset = test_val["test"]
        
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset

    def preprocess_data(self, dataset):
        """
        Preprocess dataset for training.
        
        Args:
            dataset: HuggingFace dataset
            
        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples["sentence"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        tokenized = dataset.map(tokenize_function, batched=True)
        tokenized = tokenized.rename_column("label", "labels")
        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        return tokenized

    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            Dict of metrics
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average="macro")
        
        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1
        }

    def train(self):
        """Fine-tune FinBERT on Financial PhraseBank."""
        logger.info(f"Starting fine-tuning of {self.base_model}")
        
        # Load tokenizer and model
        logger.info("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=3  # positive, neutral, negative
        )
        
        # Load and preprocess data
        train_dataset, val_dataset, test_dataset = self.load_data()
        
        train_dataset = self.preprocess_data(train_dataset)
        val_dataset = self.preprocess_data(val_dataset)
        test_dataset = self.preprocess_data(test_dataset)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=self.max_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            push_to_hub=False,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=50,
            save_total_limit=2,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(test_dataset)
        logger.info(f"Test results: {test_results}")
        
        # Save model
        logger.info(f"Saving fine-tuned model to {self.output_dir}")
        trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
        
        # Save test results
        import json
        results_path = self.output_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info("Fine-tuning complete!")
        
        return test_results


def main():
    """Main fine-tuning script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT on Financial PhraseBank")
    parser.add_argument(
        "--base_model",
        type=str,
        default="ProsusAI/finbert",
        help="Base FinBERT model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/finetuned_finbert",
        help="Output directory"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=3,
        help="Maximum training epochs"
    )
    
    args = parser.parse_args()
    
    # Run fine-tuning
    finetuner = FinBERTFineTuner(
        base_model=args.base_model,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs
    )
    
    results = finetuner.train()
    
    print("\n" + "="*50)
    print("Fine-tuning Results")
    print("="*50)
    print(f"Test Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Test Macro-F1: {results['eval_macro_f1']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
