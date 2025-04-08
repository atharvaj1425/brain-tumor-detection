
    Open an image file, convert to RGB, resize it to the model's expected input,
    normalize pixel values, and add a batch dimension.
    """
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))  # Update size based on your model's requirements
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array