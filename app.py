from flask import Flask, request, jsonify
from flask_cors import CORS
from func import predict_result
import keras

#app instance
app = Flask(__name__)
CORS(app)

model = keras.saving.load_model("final_model.h5")

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    # Check if the request contains JSON data
    if request.method == 'POST' and request.is_json:
        # Get the JSON data from the request
        data = request.get_json()

        # Check if the 'text' field is in the JSON data
        if 'text' in data:
            # Retrieve the text from the 'text' field
            input_text = data['text']

            global model

            # Predict ambiance
            prediction = predict_result(model, input_text)

            # Print the predicted ambiance label
            print("Input Text:", input_text)
            print("Predicted Sentiment:", prediction)

            # Return results as JSON
            return jsonify({
                'result': prediction,
                'status': 200,
            }), 200
        else:
            # If 'text' field is missing in the JSON data
            return jsonify({
                'error': 'Missing or invalid text field in the request',
                'status': 400,
            }), 400  # Return HTTP 400 Bad Request
    else:
        # If the request is not a POST request or not JSON
        return jsonify({
            'error': 'Invalid request',
            'status': 400,
        }), 400  # Return HTTP 400 Bad Request


if __name__ == "__main__":
    app.run(debug=True, port=8080)
