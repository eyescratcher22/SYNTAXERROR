from flask import Flask, request, jsonify, send_file
import pump_find_new as mock
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route('/generate-image',methods=['POST'])
def generate_image():
    name=request.json.get('name')
    # print(name)
    image_path=mock.generate_pump_image(name)
    return send_file(image_path, mimetype='image/png')


if __name__ =='__main__':
    app.run(debug=True, port=3000)