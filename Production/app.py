import io
from PIL import Image
from flask import Flask,request, jsonify,render_template
from flask_cors import CORS
import torch
from modelDefination.Basnet import Basnet
from modelDefination.Test_Transform import test_transform 


app = Flask(__name__)
CORS(app)

CLASS_MAPPING = {
    0: 'ञ',  # character_10_yna
    1: 'ट',  # character_11_taamatar
    2: 'ठ',  # character_12_thaa
    3: 'ड',  # character_13_daa
    4: 'ढ',  # character_14_dhaa
    5: 'ण',  # character_15_adna
    6: 'त',  # character_16_tabala
    7: 'थ',  # character_17_tha
    8: 'द',  # character_18_da
    9: 'ध',  # character_19_dha
    10: 'क', # character_1_ka
    11: 'न', # character_20_na
    12: 'प', # character_21_pa
    13: 'फ', # character_22_pha
    14: 'ब', # character_23_ba
    15: 'भ', # character_24_bha
    16: 'म', # character_25_ma
    17: 'य', # character_26_yaw
    18: 'र', # character_27_ra
    19: 'ल', # character_28_la
    20: 'व', # character_29_waw
    21: 'ख', # character_2_kha
    22: 'श', # character_30_motosaw
    23: 'ष', # character_31_petchiryakha
    24: 'स', # character_32_patalosaw
    25: 'ह', # character_33_ha
    26: 'क्ष',# character_34_chhya
    27: 'त्र', # character_35_tra
    28: 'ज्ञ', # character_36_gya
    29: 'ग', # character_3_ga
    30: 'घ', # character_4_gha
    31: 'ङ', # character_5_kna
    32: 'च', # character_6_cha
    33: 'छ', # character_7_chha
    34: 'ज', # character_8_ja
    35: 'झ', # character_9_jha
    36: '०', # digit_0
    37: '१', # digit_1
    38: '२', # digit_2
    39: '३', # digit_3
    40: '४', # digit_4
    41: '५', # digit_5
    42: '६', # digit_6
    43: '७', # digit_7
    44: '८', # digit_8
    45: '९'  # digit_9
}




@app.route('/')
def index():
    return "Hello, World!"

@app.route('/health_check')
def hello():
    return "API is runningg"

# try: 
#     model = Basnet(num_classes=46)
#     MODEL_PTH = "../models/Basnet-SA-V2(HD).pth"
#     model.load_state_dict(torch.load(MODEL_PTH), map_location=torch.device('cpu'))  # map_location ley chai chalako machine ko device add garxa 
#     model.eval()
#     print('Model loaded successfully')

    
# except Exception as e:
#     print(f'Error loading model: {e}')  

@app.route('/predict', methods=['POST'])
def predict():

    # model = None

    # try: 
    #     model = Basnet(num_classes=46)
    #     MODEL_PTH = "../models/Basnet-SA-V2(HD).pth"
    #     model.load_state_dict(torch.load(MODEL_PTH))
    #     model.eval()
    #     print('Model loaded successfully')

    
    # except Exception as e:
    #     print(f'Error loading model: {e}') 

    model = Basnet(num_classes=46)
    MODEL_PTH = "../models/Basnet-SA-V2(HD).pth"
    model.load_state_dict(torch.load(MODEL_PTH, map_location=torch.device('cpu')))  # map_location ley chai chalako machine ko device add garxa
    model.eval()
    print('Model loaded successfully')


    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    test_transform_instance  = test_transform()
    image = request.files['image'] # not a actual image just raw bytes

    actual_image = Image.open(io.BytesIO(image.read()))  # making a image from raw bytes

    transformed_image = test_transform_instance(actual_image) # transforming accroidn to test_transform

    with torch.no_grad():
        output = model(transformed_image.unsqueeze(0)) # Add batch dimension
        predicted_index = torch.argmax(output, dim=1).item()
        predicted_class = CLASS_MAPPING.get(predicted_index, "Unknown") # if outof index then return Unknown
        return jsonify({'prediction': predicted_class, 'predicted_index': predicted_index })

    

if __name__ == '__main__':
    app.run(debug=True)