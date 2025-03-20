import os
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import pandas as pd
import torch
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def prepare_prescription_data():
    """Prepare prescription dataset for training."""
    try:
        # Example prescription data structure
        prescription_data = {
            'input_text': [
                "Patient prescribed Amoxicillin 500mg three times daily for 7 days",
                "Take Metformin 1000mg twice daily with meals",
                # Add more examples here
            ],
            'output_text': [
                {
                    "medication": "Amoxicillin",
                    "dosage": "500mg",
                    "frequency": "three times daily",
                    "duration": "7 days"
                },
                {
                    "medication": "Metformin",
                    "dosage": "1000mg",
                    "frequency": "twice daily",
                    "instructions": "with meals"
                }
            ]
        }
        
        # Convert to dataset format
        dataset = Dataset.from_dict({
            'input_text': prescription_data['input_text'],
            'output_text': [str(out) for out in prescription_data['output_text']]
        })
        
        return dataset
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise

def train_model():
    """Fine-tune FLAN-T5 model on prescription data."""
    try:
        # Load model and tokenizer
        model_name = "google/flan-t5-base"  # Using FLAN-T5 as it's open source and powerful
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Prepare dataset
        dataset = prepare_prescription_data()
        
        # Split dataset
        dataset = dataset.train_test_split(test_size=0.2)
        
        # Tokenization function
        def preprocess_function(examples):
            inputs = examples["input_text"]
            targets = examples["output_text"]
            
            model_inputs = tokenizer(
                inputs,
                max_length=128,
                padding="max_length",
                truncation=True
            )
            
            labels = tokenizer(
                targets,
                max_length=128,
                padding="max_length",
                truncation=True
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Tokenize datasets
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./prescription_model",
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model("./prescription_model_final")
        tokenizer.save_pretrained("./prescription_model_final")
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model() 