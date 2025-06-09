## File overview
Below are the files in the repository and their descriptions:
- anonymize_turbine_number.py: A script to anonymize turbine numbers in the Tonstad dataset.
- config.py: File for initializing the configuration settings, and enables editing of the configuration during runtime.
- config.yaml: Configuration file for the project, containing settings such as model parameters and data augmentations.
- data_preparation.py: A script for preparing the dataset, like setting augmentations and loading the data.
- label_images.py: A script to label images in the dataset into rough snow categories ("None", "Low", "Medium", "High", "Unknown").
- label_ordering.py: A script to label the ordering of images in the dataset.
- main.py: The main script that runs the project, orchestrating the data preparation, model training, and evaluation.
- model.py: Contains the model architecture and loss functions.
- pyproject.toml: A configuration file for the project, specifying dependencies.
- trainer.py: A script for training the model, including the training loop and evaluation metrics.
- utils.py: Contains utility functions used throughout the project.

## Config Overview
The configuration file `config.yaml` contains the following parameters:

### Training:
- epochs: Number of training epochs.
- train_portion: Fraction of the dataset used for training.
- val_portion: Fraction of the dataset used for validation.
- batch_size: Size of the training batches.
- learning_rate: Learning rate for the optimizer.
- weight_decay: Weight decay for regularization.
- device: Device to run the model on (CPU or GPU).
- early_stopping: Whether to use early stopping during training.
- loss: Loss function to use for training.
- quick_dev_run: Whether to run a quick development cycle with a smaller dataset.

### Image:
- height: Height of the input images. Ignored for ViT models.
- hidth: Width of the input images. Ignored for ViT models.
- reference_image: Whether to inject reference images into normal images.
- normalize: Whether to normalize the images. ignored for ViT models.
- horizontal_flip: Chance for horizontal flipping of images.
- brightness: Chance for brightness adjustment.
- contrast: Chance for contrast adjustment.
- saturation: Chance for saturation adjustment.
- hue: Chance for hue adjustment.

### Dataset:
- split: Dataset split to use (e.g., "All, turbine, angle").

### Model:
- snowranker: kernel_size: Size of the kernel for the SnowRanker model.
- vit: pre_trained: Which pre-trained model to use for ViT.
- vit: dropout: Dropout rate for the ViT model.

### Wandb:
api_key: API key for Weights & Biases.
username: Username for Weights & Biases.
project: Project name for Weights & Biases.
disabled: Whether to disable Weights & Biases logging.
run_name: Name of the run in Weights & Biases.
