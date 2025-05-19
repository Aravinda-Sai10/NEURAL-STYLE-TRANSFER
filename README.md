                                              NEURAL STYLE TRANSFER

This is a **Streamlit-based web application** that performs **Neural Style Transfer** using a pre-trained VGG19 model. Upload a **content image** and a **style image**, and watch as your content gets transformed into a beautiful artistic rendering based on the style!

---

# FEATURES:

- Upload content and style images 
- Transfers artistic style using VGG19 CNN from TensorFlow
- Adjustable weights for content and style
- Progress bar to visualize the training process
- Download the final stylized image as PNG
- Custom CSS styling for a clean interface

---

# APP PREVIEW:

![APP PREVIEW](screenshots/OUTPUT%201.png)
# Content Image:
![content Image](https://github.com/user-attachments/assets/c86889ae-02a1-4289-a0c6-ce027b5626a0)

# Style Image:
![style Image](https://github.com/user-attachments/assets/d1ee1fb4-c17b-4627-bf54-bf9fe8c7c0a2)

# WORKING:
![OUTPUT 2](https://github.com/user-attachments/assets/6f03b46b-fe2d-4e15-8e2c-a84dd15f30d5)
![OUTPUT 3](https://github.com/user-attachments/assets/2ad2c812-baf1-4f5b-aeda-022308e4a8ed)

---

## 📂 FILE STRUCTURE:

neural-style-transfer/
├── app.py                  # Main Streamlit app
├── style.css               #  CSS for UI styling
├── requirements.txt        # Python dependencies
├── .gitignore              # Files to ignore in Git
└── screenshots/
    └── output.png  

# TECHNOLOGIES USED:

1.Streamlit

2.TensorFlow

3.Pillow (PIL)

4.NumPy

#  HOW TO RUN:

1.Clone this repository:
  
   git clone https://github.com/Aravinda-Sai10/Neural-Style-Transfer.git
   cd Neural-Style-Transfer
   
2.  Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  

3. Install dependencies:
   pip install -r requirements.txt

4. Run the app:
   streamlit run app.py

---

# HOW IT WORKS:

1.Load the content and style image.
2.Use a pre-trained VGG19 model to extract content and style features.
3.Compute content and style losses.
4.Optimize a copy of the content image to minimize the total loss.
5.Display and download the stylized output.

---
# USES:
1.**AI Art Generation** – Turn ordinary photos into stylized artwork using famous painting styles (e.g., Van Gogh, Monet).

2.**Image Style Blending** – Mix textures, brush strokes, and colors from one image with the structure of another.

3.**Educational Demo** – Learn how convolutional neural networks (CNNs) extract content and style features using a pre-trained VGG19 model.


