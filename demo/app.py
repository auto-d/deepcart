import gradio as gr
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score, confusion_matrix, ConfusionMatrixDisplay

gallery_images = []

def add_image(new_image):
    gallery_images.append(new_image)
    return gr.Gallery.update(value=gallery_images)

def change_mode(mode): 
    print(f"Mode changed to {mode}")

# Simulated product data
product_data = [
    {"id": "p1", "name": "Cool Hat", "price": "$19.99", "url": "https://picsum.photos/id/1015/200/200"},
    {"id": "p2", "name": "Snazzy Shirt", "price": "$29.99", "url": "https://picsum.photos/id/1016/200/200"},
    {"id": "p3", "name": "Stylish Shoes", "price": "$49.99", "url": "https://picsum.photos/id/1018/200/200"},
    {"id": "p4", "name": "Fancy Jacket", "price": "$99.99", "url": "https://picsum.photos/id/1020/200/200"},
]

# Cart: Set of product IDs
cart = set()

# Build the initial dataset samples
def build_dataset_samples():
    return [[item["url"]] for item in product_data]

# Callback to toggle item in cart and display current selection
def on_click(evt: gr.SelectData):
    index = evt.index
    product = product_data[index]
    pid = product["id"]

    if pid in cart:
        cart.remove(pid)
    else:
        cart.add(pid)

    # Prepare cart contents as string
    selected_items = [f"{p['name']} ({p['price']})" for p in product_data if p["id"] in cart]
    cart_text = "🛒 Your Cart:\n\n" + "\n".join(selected_items) if selected_items else "🛒 Your cart is empty."

    return cart_text

def main(): 
    """
    Our Gradio application. 

    NOTE: The base code for the Gradio gallery is sourced from a dialogue with gpt-4o regarding the best
    way to emulate a shopping cart interface. See https://chatgpt.com/share/6880fcab-233c-8013-8df7-0b1195abb52c
    for the exchange. 
    """
    demo = gr.Blocks()
    with demo: 

        # Header         
        gr.Markdown(value="# 🛒 DeepCart")
        gr.Markdown(value="##  Probing the depths of the Amazon electronics storefront")
        gr.Markdown(value="You've seen the best products Amazon has to offer, but have you seen the worst? Interact with our collection of the worst products the storefront has to offer and see more terrible products based on your preferences!")

        # Settings 
        with gr.Row():                 
            mode_picker = gr.Dropdown(choices=["Positive", "Negative"], value="Positive", label='Recommender Mode', interactive=True)
            mode_picker.change(fn=change_mode, inputs=[mode_picker])
            topk_slider = gr.Slider(label="Recommendations to generate", value=5, maximum=50, step=5)            

        with gr.Row():
            image_input = gr.Image()
            gallery = gr.Gallery(columns=3, height="auto")

        image_input.change(fn=add_image, inputs=image_input, outputs=gallery)

        dataset = gr.Dataset(components=[gr.Image(type="filepath", label="Product")],
                            samples=build_dataset_samples(), label="Recommendations")
        cart_display = gr.Textbox(label="Cart Contents", lines=6)
        dataset.select(fn=on_click, outputs=cart_display)

    demo.launch(share=False)

if __name__ == "__main__":
    main()