# Pasta Shape Recognizer

An image classification model from data collection, cleaning, model training, deployment and API integration. <br/>
The model can classify 20 different types of pasta shapes. The types are following: <br/>

1. Spaghetti
2. Fettuccine
3. Penne
4. Rigatoni
5. Fusilli
6. Farfalle
7. Linguine
8. Tagliatelle
9. Lasagna
10. Ravioli
11. Tortellini
12. Orecchiette
13. Conchiglie
14. Rotini
15. Bucatini
16. Cannelloni
17. Macaroni
18. Orzo
19. Cavatappi
20. Gemelli

## Build from Source

### Clone the repo
```bash
git clone https://github.com/zzarif/Pasta-Shape-Recognizer.git
```

### Install CUDA Toolkit

- Go to the NVIDIA CUDA Toolkit download page: https://developer.nvidia.com/cuda-downloads
- Select "Windows" as the operating system and choose the appropriate version and installer type.
- Download and run the installer, following the installation instructions.

### Initialize and activate virtual environment

```bash
virtualenv --no-site-packages venv
source venv/Scripts/activate
```

### Install PyTorch with CUDA support

Run the following command to install PyTorch with CUDA support using `pip3`:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

_Note: Replace `cu121` with the appropriate CUDA version tag that matches your installed CUDA version (e.g., `cu121` for CUDA `12.1`)._

### Verify the installation

Run the following code to check if PyTorch is using the GPU:

```python
import torch
print(torch.cuda.is_available())
```

If the output is `True`, then PyTorch is successfully set up to use the GPU.  `torch.cuda.get_device_name(0)` should also show your GPU config.

### Install Dependencies

```bash
pip3 install fastai fastbook nbdev gradio
```

## Dataset Preparation

**Data Collection:** Downloaded from DuckDuckGo using term name.
**DataLoader:** Used fastai DataBlock API to set up the DataLoader.
**Data Augmentation:** fastai provides default data augmentation which operates in GPU.
Details can be found in `notebooks/data_prep.ipynb`

## Training and Data Cleaning

**Training:** Fine-tuned a `resnet34` model for 9 epochs (5+2+2) and got upto **~85.6%** accuracy.
**Data Cleaning:** This part took the highest time. Since I collected data from browser, there were many noises. Also, there were images that contained. I cleaned and updated data using fastai ImageClassifierCleaner. I cleaned the data each time after training or finetuning, except for the last time which was the final iteration of the model.

## Model Deployment
I deployed the model to HuggingFace Spaces Gradio App. The implementation can be found in `deployment` folder or [here](https://huggingface.co/spaces/zzarif/Pasta-Shape-Recognizer).
![Pasta-Shape-Recognizer](deployment/deployed_model_hf_spaces.png)

## API integration with GitHub Pages

The deployed model API is integrated [here](https://zzarif.github.io/Pasta-Shape-Recognizer/) in GitHub Pages Website. Implementation and other details can be found in `docs` folder.

### Contact Me
Would appreciate your feedback. For any further queries feel free to reach out to me at [zibran.zarif.amio@gmail.com](mailto:zibran.zarif.amio@gmail.com)
