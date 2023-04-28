from flask import Flask,request,render_template
from flask_ngrok import run_with_ngrok
import torch 
from diffusers import StableDiffusionPipeline
import base64
from io import BytesIO


#Load the model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", revision="fp16",torch_dtype=torch.float16
    
)

pipe.to("cuda") 

app = Flask("__main__")
run_with_ngrok(app)


app.route("/")
def home():
    return render_template("Home.html")


@app.route("/submit-prompt", methods=["POST"])
def generate():
    prompt = request.form("prompt-input")
    image = pipe(prompt).images[0]
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())
    img_str = "data:image/png;base64, " + str(img_str)[2:-1]
    return render_template('index.html', generated_image=img_str)




    
    
    
    


