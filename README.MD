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

# SCREENSHOTS:

![APP PREVIEW](screenshots/OUTPUT%201.png)

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

Streamlit

TensorFlow

Pillow (PIL)

NumPy

#  HOW TO RUN:

1. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  

2. Install dependencies:
   pip install -r requirements.txt

3. Run the app:
   streamlit run app.py

---

# HOW IT WORKS:

1.Load the content and style image.
2.Use a pre-trained VGG19 model to extract content and style features.
3.Compute content and style losses.
4.Optimize a copy of the content image to minimize the total loss.
5.Display and download the stylized output.

---
