import argparse
import json
import os
import torch
import logging
from datetime import datetime
from PIL import Image
from transformers import pipeline, BitsAndBytesConfig, MllamaForConditionalGeneration, AutoProcessor, AutoTokenizer
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset

# Argument parser setup
parser = argparse.ArgumentParser(description="Process a dataset of images and questions.")
parser.add_argument("dataset_path", type=str, help="Path to the JSON dataset file")
parser.add_argument("images_folder", type=str, help="Path to the folder containing the images")
parser.add_argument("output_path", type=str, help="Path to the output JSON file")
parser.add_argument("--log_path", type=str, help="Path to the log file", default=None)
args = parser.parse_args()

# Set up logging
if args.log_path is None:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S.log"))
else:
    log_file = args.log_path

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=bnb_config)

# NOTE: Pipeline support for image text to text is bleeding edge so it's not
# available in the latest release. The following code will work once it is
# Create a pipeline for visual question answering
# pipe = pipeline(
#     task="image-text-to-text",
#     model=model_id,
#     device_map="auto",
#     quantization_config=bnb_config,
#     torch_dtype=torch.bfloat16,
# )

# Load the dataset
dataset = load_dataset('json', data_files=args.dataset_path, field='questions')

for out in pipe(KeyDataset(dataset, "file")):
    logging.info(out)

# List to store results
results = []

# Total number of questions
total_questions = len(dataset)
trial_questions = max(1, total_questions // 100)  # Ensure at least one question is processed
progress_interval = trial_questions // 10

logging.info(f"Processing {trial_questions} questions from the dataset of {total_questions} questions")

# Define the batch size
batch_size = 8

# Iterate through the dataset using an iterator
for i, batch in enumerate(dataset.to_iterable_dataset(batch_size=batch_size)):
    batch_images = []
    batch_texts = []
    batch_question_ids = []

    for question in batch:
        try:
            image_id = question['image_id']
            question_text = question['question']
            question_id = question['question_id']
            
            # Construct the image file path
            image_path = os.path.join(args.images_folder, f"COCO_val2014_{image_id:012d}.jpg")
            
            # Load the image
            image = Image.open(image_path)
            
            batch_images.append(image)
            batch_texts.append(question_text)
            batch_question_ids.append(question_id)
        except Exception as e:
            logging.error(f"An error occurred while preparing batch at question {i * batch_size}: {e}")

    # Process the batch
    try:
        outputs = pipe(images=batch_images, questions=batch_texts)
        for question_id, output in zip(batch_question_ids, outputs):
            answer = output['generated_text']
            results.append({
                "question_id": question_id,
                "answer": answer
            })
    except Exception as e:
        logging.error(f"An error occurred while processing batch at question {i * batch_size}: {e}")

    # Log progress and save checkpoint every 10%
    if (i * batch_size) % progress_interval == 0:
        logging.info(f"Processed {((i * batch_size) / trial_questions) * 100:.0f}% of the dataset")
        # Save checkpoint
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Checkpoint saved to {args.output_path}")

# Save final results to a JSON file
with open(args.output_path, 'w') as f:
    json.dump(results, f, indent=4)

logging.info(f"Final results saved to {args.output_path}")
