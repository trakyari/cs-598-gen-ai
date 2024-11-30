# Prerequisites

- Validation images from the VQA dataset - https://visualqa.org/download.html
- Place images into folder called `val2014`
- Run resize.py to resize images

# Run

- Run `python vision.py` to evaluate the model

# Files

- `question_ids.json`: list of question ids that were answered by the model
- `second_pass.json`: list of answer objects in the format used by the VQA evaluation logic answered by Llama 3.2
- `vision.py`: the evaluation script for the vision task
- `resize.py`: the script used to resize the images
