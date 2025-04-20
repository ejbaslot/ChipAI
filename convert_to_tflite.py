import tensorflow as tf

# Load the .h5 model
model = tf.keras.models.load_model('ChipAI/models/chili_pepper_classifier.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('ChipAI/models/chili_pepper_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Model successfully converted to TFLite!")
