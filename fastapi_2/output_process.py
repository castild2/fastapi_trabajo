import numpy as np

CLASS_MAPPING = sorted(['Millionarie Shortbread', 'Lemmon Muffin',
                 'Salted Caramel Brownie', 'Chocolate Tiffin', 'Rocky Road',
                 'Bakewell Tart', 'Bluberry Muffin', 'Raspberry Almond Bake'])


def process_output(prediction):
    predicted_class = int(np.argmax(prediction[0]))
    predicted_class_name = CLASS_MAPPING[predicted_class]
    predicted_proba = prediction[0][predicted_class]
    result = {'class': predicted_class_name, 'prob': float(predicted_proba)}
    return result
