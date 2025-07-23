import gradio as gr
import pandas as pd
import numpy as np
from dataset import DeepCartDataset, DeepCartTorchDataset
import autoencoder

from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Our prediction tool 
data = None 
model = None

# Simulated product data
product_data = [
    {"id": "p1", "name": "Data1 Hat", "price": "$19.99", "url": "https://picsum.photos/id/1015/200/200"},
    {"id": "p2", "name": "Snazzy Shirt", "price": "$29.99", "url": "https://picsum.photos/id/1016/200/200"},
    {"id": "p3", "name": "Stylish Shoes", "price": "$49.99", "url": "https://picsum.photos/id/1018/200/200"},
]

product_data2 = [
    {"id": "p3", "name": "Data2 Shoes", "price": "$49.99", "url": "https://picsum.photos/id/1018/200/200"},
    {"id": "p4", "name": "Fancy Jacket", "price": "$99.99", "url": "https://picsum.photos/id/1020/200/200"},
]

product_data3 = [
    {"id": "p1", "name": "Data3 Hat", "price": "$19.99", "url": "https://picsum.photos/id/1015/200/200"},
    {"id": "p4", "name": "Fancy Jacket", "price": "$99.99", "url": "https://picsum.photos/id/1020/200/200"},
]

# Our poor-man's shopping interface, a set of gallery images and item selections
gallery_images = []
cart = set()
products = product_data

def initialize():     
    # ds = DeepCartDataset(tag="small")
    # ds.load("data/processed")
    # ds.split()
    # data = DeepCartTorchDataset(matrix=ds.test)
    # model = autoencoder.load_model("../models/")
    global products 

    products = product_data

def add_image(new_image):
    gallery_images.append(new_image)
    return gr.Gallery.update(value=gallery_images)

def change_mode(mode): 
    print(f"Mode changed to {mode}")

  
# Build the initial dataset samples
def build_dataset_samples():
    return [[item["url"]] for item in products]

def update_dataset(): 
    return gr.Dataset(samples=build_dataset_samples())

def generate(topk=5):
    # if model: 
    #     top_ks = model.recommend(data, top_k=topk)
    #     print(top_ks)
        
    global products 

    products = product_data3
    return update_dataset()

# Callback to toggle item in cart and display current selection
def on_click(evt: gr.SelectData):
    global products 

    index = evt.index
    product = product_data[index]
    print(f"product {product} clicked")

    pid = product["id"]

    if pid in cart:
        cart.remove(pid)
    else:
        cart.add(pid)

    # Update displayed items 
    products=product_data2

    # Prepare cart contents as string
    selected_items = [f"{p['name']} ({p['price']})" for p in products if p["id"] in cart]
    cart_text = "🛒 Your Cart:\n\n" + "\n".join(selected_items) if selected_items else "🛒 Your cart is empty."

    return cart_text, update_dataset()

def main(): 
    """
    Our Gradio application. 

    NOTE: The base code for the Gradio gallery is sourced from a dialogue with gpt-4o regarding the best
    way to emulate a shopping cart interface. See https://chatgpt.com/share/6880fcab-233c-8013-8df7-0b1195abb52c
    for the exchange. 
    """
    initialize() 

    demo = gr.Blocks()
    with demo: 

        # Header         
        gr.Markdown(value="# 🛒 DeepCart")
        gr.Markdown(value="##  Probing the depths of the Amazon electronics storefront")
        gr.Markdown(value="You've seen the best products Amazon has to offer, but have you seen the worst? Interact with our collection of the worst products the storefront has to offer and see more terrible products based on your preferences!")

        # Image gallery 
        # with gr.Row():
        #     image_input = gr.Image()
        #     gallery = gr.Gallery(columns=3, height="auto")

        # image_input.change(fn=add_image, inputs=image_input, outputs=gallery)
        
        ds = gr.Dataset(
            components = [
                gr.Image(type="filepath", label="Product")
                ],
            samples=build_dataset_samples(), 
            label="Recommendations")
        cart_display = gr.Textbox(label="Cart Contents", lines=6)
        ds.select(fn=on_click, outputs=[cart_display, ds])

        # Settings 
        with gr.Row():                 
            mode_picker = gr.Dropdown(choices=["Positive", "Negative"], value="Positive", label='Recommender Mode', interactive=True)
            mode_picker.change(fn=change_mode, inputs=[mode_picker])
            topk_slider = gr.Slider(label="Recommendations to generate", value=5, maximum=50, step=5)
            topk_slider.change(fn=generate, inputs=topk_slider, outputs=ds)

    demo.launch(share=False)

if __name__ == "__main__":
    main()

#import gradio as gr

# philosophy_quotes = [
#     ["I think therefore I am."],
#     ["The unexamined life is not worth living."]
# ]

# startup_quotes = [
#     ["Ideas are easy. Implementation is hard"],
#     ["Make mistakes faster."]
# ]

# def show_startup_quotes():
#     return gr.Dataset(samples=startup_quotes)

# with gr.Blocks() as demo:
#     textbox = gr.Textbox()
#     dataset = gr.Dataset(components=[textbox], samples=philosophy_quotes)
#     button = gr.Button()

#     button.click(show_startup_quotes, None, dataset)

# demo.launch()