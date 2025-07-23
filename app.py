import gradio as gr
import pandas as pd
import numpy as np
from dataset import DeepCartDataset, DeepCartTorchDataset
import autoencoder
import similarity

from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Our prediction tool 
ref_data = None
user_data = None 
model = None

# Our poor-man's shopping interface, a set of gallery images and item selections
products = []
top_k = 5
selected = 0

def initialize():         
    """
    Initialize global state and populate initial recommendation s
    """
    global user_data
    global ref_data
    global model 
    global products 

    # Retrieve product reference information 
    ref_data = DeepCartDataset("small")
    ref_data.load("data/processed")

    # Recover our model and create an empty dataset for our user(s)
    model = autoencoder.load_model("models/") 
    user_data = model.prepare_new_dataset()
    
    # Bootstrap our predictions with the model's best guesses    
    recs = model.recommend(user_data, k=20)
    products = []
    for index, item in recs.iterrows(): 
        details = get_item_details(item.item_id)
        products.append(details) 

def get_item_details(item_id):
    """
    Get item details from reference data
    """
    global ref_data 
    
    df = ref_data.items[ref_data.items.item_id == item_id]
    if (len(df)) > 0: 
        urls = df.images.values[0]
        
        # Default stock image to avoid jarring user experience
        url = "https://cdn3d.iconscout.com/3d/premium/thumb/product-3d-icon-download-in-png-blend-fbx-gltf-file-formats--tag-packages-box-marketing-advertisement-pack-branding-icons-4863042.png"
        if len(urls) > 0: 
            try:
                url = urls[0]['large']
            except Exception as e: 
                pass 
        
        return {
            "id": item_id, 
            "name": df.title.values[0], 
            "price": df.price.values[0], 
            "rating": df.average_rating.values[0],
            "url": url 
        }

def change_mode(mode): 
    """
    Toggle mode
    """
    print(f"Mode changed to {mode}")
  
# Build the initial dataset samples
def build_dataset_samples():
    """
    Revise the samples
    """
    return [item["url"] for item in products]

def generate(topk=5):
    """
    Generate new recommendations when the top_k changes
    """
    global user_data
    global model 
    global products 

    if model: 
        top_ks = model.recommend(user_data, k=topk)
        print(top_ks)

    return build_dataset_samples()

def on_click(evt: gr.SelectData):
    """
    Callback to toggle item in cart and display current selection
    """
    global products 
    global selected 

    stars = ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]

    selected = evt.index
    product = products[selected]   
    
    rating = int(product['rating']) 
    rating = max(1, rating)
    rating = min(rating, 5)
    star_rating = stars[rating-1]
    
    product_text = f"**Product Name**:{product['name']}\n\n"\
        f"**Price**: ${product['price']}\n\n"\
        f"**Average Rating**: {star_rating}\n\n"\
        f"**Product ID**:{product['id']}"

    return product_text

def submit_rating(r):
    """
    New rating received
    """
    global user_data
    global ref_data
    global model 
    global products 
    global selected 

    # Bootstrap our predictions with the model's best guesses    
    selected_item = products[selected]['id']
    item_index = user_data.i_map.get(selected_item)
    user_data.ui[0][item_index] = r/5

    recs = model.recommend(user_data, k=20)
    products = []
    for index, item in recs.iterrows(): 
        details = get_item_details(item.item_id)
        products.append(details) 

    return build_dataset_samples()

def main(): 
    """
    Our Gradio application. 

    NOTE: The base code for the Gradio gallery is sourced from a dialogue with gpt-4o regarding the best
    way to emulate a shopping cart interface. See https://chatgpt.com/share/6880fcab-233c-8013-8df7-0b1195abb52c
    for the exchange. 
    """
    global products 

    demo = gr.Blocks()
    with demo: 

        # Header         
        gr.Markdown(value="# 🛒 DeepCart")
        gr.Markdown(value="##  Probing the depths of the Amazon electronics storefront")
        gr.Markdown(value="You've seen the best products Amazon has to offer, but have you seen the worst? Interact with our collection of the worst products the storefront has to offer and see more terrible products based on your preferences!")

        topk_slider = gr.Slider(label="Recommendations to generate", value=5, maximum=50, step=5)

        # Use of gr.Radio with emojis courtesy of gpt-4o, see
        # https://chatgpt.com/share/68813240-1cf4-8013-b0d5-393e661c9508

        initialize()         

        gallery = gr.Gallery(label="Products", columns=3, height="auto", 
                             value = build_dataset_samples())         
        product_info = gr.Markdown(label="Product Info")

        gallery.select(fn=on_click, outputs=product_info)
        topk_slider.change(fn=generate, inputs=topk_slider, outputs=gallery)

        with gr.Column(): 
            rating = gr.Radio(choices=["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], label="Rate this product!")
            output = gr.Textbox("Output")
            rating.change(fn=submit_rating, inputs=rating, outputs=output)

    demo.launch(share=False)

if __name__ == "__main__":
    main()