from fastai.vision.all import *
import gradio as gr

import platform
import pathlib
plt = platform.system() 
if plt == 'Linux': 
    pathlib.WindowsPath = pathlib.PosixPath
    
pasta_shape_labels = (
    "bucatini",
    "cannelloni",
    "cavatappi",
    "conchiglie",
    "farfalle",
    "fettuccine",
    "fusilli",
    "gemelli",
    "lasagna",
    "linguine",
    "macaroni",
    "orecchiette",
    "orzo",
    "penne",
    "ravioli",
    "rigatoni",
    "rotini",
    "spaghetti",
    "tagliatelle",
    "tortellini"
)

model = load_learner('models/pasta_shape_recognizer_v2.pkl')

def recognize_image(image):
    pred, idx, probs = model.predict(image)
    return dict(zip(pasta_shape_labels, map(float, probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label(num_top_classes=5)
examples = [
    'deployment/unknown00.png',
    'deployment/unknown01.png',
    'deployment/unknown02.png',
    'deployment/unknown03.png',
    'deployment/unknown04.png',
    'deployment/unknown05.png',
    'deployment/unknown06.png',
    'deployment/unknown07.png',
    'deployment/unknown08.png',
    'deployment/unknown09.png',
    'deployment/unknown10.png',
    'deployment/unknown11.png',
    'deployment/unknown12.png',
    'deployment/unknown13.png'
]

iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False, share=True)