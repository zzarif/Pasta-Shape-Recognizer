from fastai.vision.all import *
import gradio as gr

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
    'app/unknown00.png',
    'app/unknown01.png',
    'app/unknown02.png',
    'app/unknown03.png',
    'app/unknown04.png',
    'app/unknown05.png',
    'app/unknown06.png',
    'app/unknown07.png',
    'app/unknown08.png',
    'app/unknown09.png',
    'app/unknown10.png',
    'app/unknown11.png',
    'app/unknown12.png',
    'app/unknown13.png'
]

iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False, share=True)