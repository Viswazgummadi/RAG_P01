#!/bin/bash

# RAG Workflow Script - Run the entire pipeline from data processing to inference

# Configuration
DATA_DIR="data2"
PROCESSED_DATA_DIR="processed_data"
VECTOR_DB_DIR="vector_db"
TRAINING_DATA_DIR="training_data"
MODELS_DIR="models"

# Check for CUDA availability
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "CUDA available! Using GPU for processing."
    CUDA_AVAILABLE=1
else
    echo "CUDA not available. Using CPU for processing."
    CUDA_AVAILABLE=0
fi

# Create directories if they don't exist
mkdir -p $PROCESSED_DATA_DIR
mkdir -p $VECTOR_DB_DIR
mkdir -p $TRAINING_DATA_DIR
mkdir -p $MODELS_DIR/checkpoints
mkdir -p $MODELS_DIR/final

# Step 1: Process the raw data
echo "Step 1: Processing raw data..."
python preprocess.py --input_dir $DATA_DIR --output_file $PROCESSED_DATA_DIR/processed_data.jsonl

# Step 2: Build the vector database
echo "Step 2: Building vector database..."
python vector_db.py --jsonl_file $PROCESSED_DATA_DIR/processed_data.jsonl --output_dir $VECTOR_DB_DIR

# Step 3: Create training data for model fine-tuning
echo "Step 3: Creating training data..."
python scripts/prepare_training_data.py --jsonl_file $PROCESSED_DATA_DIR/processed_data.jsonl --output_file $TRAINING_DATA_DIR/training_data.jsonl --num_examples 10

# Step 4: Fine-tune the model
echo "Step 4: Fine-tuning the model..."
if [ $CUDA_AVAILABLE -eq 1 ]; then
    BATCH_SIZE=4
    GRAD_ACC=8
else
    # Smaller batch size for CPU training
    BATCH_SIZE=1
    GRAD_ACC=16
fi

python scripts/train_model.py \
    --dataset_file $TRAINING_DATA_DIR/training_data.jsonl \
    --model_name "mistralai/Mistral-7B-v0.1" \
    --output_dir $MODELS_DIR/checkpoints \
    --batch_size $BATCH_SIZE \
    --grad_accumulation $GRAD_ACC \
    --epochs 3 \
    --use_lora

# Step 5: Evaluate the model
echo "Step 5: Evaluating the model..."
python scripts/evaluate_model.py \
    --model_path $MODELS_DIR/checkpoints/final \
    --test_file $TRAINING_DATA_DIR/training_data.jsonl \
    --output_file $MODELS_DIR/evaluation_results.json \
    --use_lora \
    --base_model "mistralai/Mistral-7B-v0.1"

# Step 6: Run the web interface
echo "Step 6: Starting the web interface..."
echo "You can interact with both the pure model and RAG-enhanced model"
python ui/app.py \
    --model_path $MODELS_DIR/checkpoints/final \
    --vector_store_path $VECTOR_DB_DIR \
    --use_lora \
    --base_model "mistralai/Mistral-7B-v0.1"

echo "Pipeline completed successfully!"
