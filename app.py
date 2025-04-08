# from flask import Flask, render_template, request

# app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         # Process the image upload and prediction
#         pass
#     return render_template("index.html")

# if __name__ == "__main__":
#     app.run(debug=True)

# from flask import Flask, render_template, request
# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from werkzeug.utils import secure_filename
# from PIL import Image

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'

# # Load the pre-trained model (make sure 'model.h5' is in your project directory)
# model = load_model('model.h5')

# def prepare_image(img_path):
#     """
#     Open an image file, convert to RGB, resize it to the model's expected input,
#     normalize pixel values, and add a batch dimension.
#     """
#     img = Image.open(img_path).convert('RGB')
#     img = img.resize((224, 224))  # Update the size based on your model's requirements
#     img_array = np.array(img) / 255.0  # Normalize pixel values
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     return img_array

# @app.route("/", methods=["GET", "POST"])
# def index():
#     tumor_type = None
#     confidence = None
#     filepath = None

#     if request.method == "POST":
#         # Ensure a file was submitted
#         if 'file' not in request.files or request.files['file'].filename == "":
#             return render_template("index.html", tumor_type=tumor_type, confidence=confidence, filepath=filepath)

#         file = request.files['file']
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         # Process the image and prepare it for prediction
#         img = prepare_image(filepath)

#         # Make prediction using the model
#         prediction = model.predict(img)
#         print("Prediction raw output:", prediction)  # Print raw model output to terminal

#         # For this example, assume a binary classification:
#         # Class 0: No Tumor, Class 1: Tumor Detected
#         class_idx = np.argmax(prediction, axis=1)[0]
#         confidence = float(np.max(prediction)) * 100

#         if class_idx == 1:
#             tumor_type = "Tumor Detected"
#         else:
#             tumor_type = "No Tumor Detected"

#         # Print result in terminal
#         print("Tumor Type:", tumor_type, "with confidence:", round(confidence, 2), "%")

#         # Render the template with results
#         return render_template("index.html", tumor_type=tumor_type, confidence=round(confidence, 2), filepath=filename)
    
#     return render_template("index.html", tumor_type=tumor_type, confidence=confidence, filepath=filepath)

# if __name__ == "__main__":
#     # Create upload folder if it doesn't exist
#     if not os.path.exists(app.config['UPLOAD_FOLDER']):
#         os.makedirs(app.config['UPLOAD_FOLDER'])
#     app.run(debug=True)


from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the pre-trained model (ensure 'model.h5' is in your project directory)
model = load_model('model_1.h5')

def prepare_image(img_path):
    """
    Open an image file, convert to RGB, resize it to the model's expected input,
    normalize pixel values, and add a batch dimension.
    """
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))  # Update size based on your model's requirements
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    tumor_type = None
    confidence = None
    filepath = None

    if request.method == "POST":
        # Ensure a file was submitted
        if 'file' not in request.files or request.files['file'].filename == "":
            return render_template("index.html", tumor_type=tumor_type, confidence=confidence, filepath=filepath)

        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image and prepare it for prediction
        img = prepare_image(filepath)

        # Make prediction using the model
        prediction = model.predict(img)
        print("Prediction raw output:", prediction)  # Print raw model output to terminal

        # Mapping of class indices to tumor categories
        labels = {0: "No Tumor", 1: "Meningioma", 2: "Pituitary", 3: "Glioma"}
        class_idx = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction)) * 100
        tumor_type = labels.get(class_idx, "Unknown")

        # Print result in terminal
        print("Tumor Type:", tumor_type, "with confidence:", round(confidence, 2), "%")

        # Render the template with results
        return render_template("index.html", tumor_type=tumor_type, confidence=round(confidence, 2), filepath=filename)
    
    return render_template("index.html", tumor_type=tumor_type, confidence=confidence, filepath=filepath)

if __name__ == "__main__":
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
