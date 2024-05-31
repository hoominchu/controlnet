from flask import Flask, request, jsonify
# import scribble_to_image_server as cn_s2i
# import hed2image
# import pose2image
# import canny2image
import scribble2image
# import base64
# import numpy as np
from PIL import Image
import base64
import io
import numpy as np
from io import BytesIO
# import json
from flask_cors import CORS
from waitress import serve

app = Flask(__name__)
CORS(app)


# a_prompt = 'best quality, extremely detailed'
# n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
# num_samples = 1
# image_resolution = 512
# detection_resolution = 512
# ddim_steps = 20
# guess_mode = False
# strength = 1
# scale = 9
# seed = 1156259033
# eta = 0.0

@app.route('/api/control_s2i', methods=['POST'])
def api_endpoint():
    b64_str = request.form.get('input_image')

    control_method = request.form.get('control_method')

    base64_decoded = base64.b64decode(b64_str)

    image = Image.open(io.BytesIO(base64_decoded))
    input_image = np.array(image)

    det = request.form.get('det')
    prompt = request.form.get('prompt')
    a_prompt = request.form.get('a_prompt')
    n_prompt = request.form.get('n_prompt')
    num_samples = int(request.form.get('num_samples'))
    image_resolution = int(request.form.get('image_resolution'))
    detect_resolution = int(request.form.get('detect_resolution'))
    ddim_steps = int(request.form.get('ddim_steps'))
    guess_mode = False
    strength = int(request.form.get('strength'))
    scale = int(request.form.get('scale'))
    seed = int(request.form.get('seed'))
    eta = float(request.form.get('eta'))

    # Do something with the arguments

    result = None

    if control_method == 'scribble':
        # (det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps,
        #  guess_mode, strength, scale, seed, eta)
        result = scribble2image.process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution,
                                ddim_steps, guess_mode, strength, scale, seed, eta)

    # elif control_method == 'pose':
    #     result = pose2image.process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution,
    #                                 detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
    # elif control_method == 'canny':
    #     result = canny2image.process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution,
    #                                  detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
    # else:
    #     result = cn_s2i.process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution,
    #                             detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)

    generated_image = Image.fromarray(result[1].astype('uint8'))
    buffered = BytesIO()
    generated_image.save(buffered, format="JPEG")
    generated_b64_bytes = base64.b64encode(buffered.getvalue())
    generated_b64_string = generated_b64_bytes.decode('utf-8')

    response_json = jsonify({'success': True, 'imageb64': generated_b64_string})

    return response_json, 200


@app.route('/api/status', methods=['GET'])
def api_status():
    response_json = jsonify({'running': True})
    return response_json, 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    # serve(app, host='0.0.0.0', port=5000)
