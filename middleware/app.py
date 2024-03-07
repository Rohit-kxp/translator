from flask import Flask, render_template ,request
from selenium import webdriver
from bs4 import BeautifulSoup
from flask_cors import CORS
from transformers import BertTokenizer
import torch
import joblib
app = Flask(__name__)
CORS(app)

model = joblib.load('model\model.joblib')
@app.route('/')
def hello_world():
    return render_template("index.html")
# @app.route('/receive-url', methods=['POST'])
# def receive_url():
#     data = request.get_json()
#     url = data.get('url')   
#     print(url)
#     current = webdriver.Chrome()  
#     current.get(url)
#     html_data=current.page_source
#     current.quit()
#     soup = BeautifulSoup(html_data, 'html.parser')
#     text_data = soup.get_text()
#     text_data = text_data.replace("\n","")
#     for text in text_data.split('.'):
#         predictions = model.predict([text])
#         if predictions[0] >= 0.50000:
#             print("Detected")
#         else :
#             print("Not Detected")
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        from_text = request.form.get('from_text')
        from_lang = request.form.get('from_language')
        print(from_lang)
        to_lang = request.form.get('to_language')
#         dict = {
#     "Arabic": "ar_AR",
#     "Czech": "cs_CZ",
#     "German": "de_DE",
#     "English": "en_XX",
#     "Spanish": "es_XX",
#     "Estonian": "et_EE",
#     "Finnish": "fi_FI",
#     "French": "fr_XX",
#     "Gujarati": "gu_IN",
#     "Hindi": "hi_IN",
#     "Italian": "it_IT",
#     "Japanese": "ja_XX",
#     "Kazakh": "kk_KZ",
#     "Korean": "ko_KR",
#     "Lithuanian": "lt_LT",
#     "Latvian": "lv_LV",
#     "Burmese": "my_MM",
#     "Nepali": "ne_NP",
#     "Dutch": "nl_XX",
#     "Romanian": "ro_RO",
#     "Russian": "ru_RU",
#     "Sinhala": "si_LK",
#     "Turkish": "tr_TR",
#     "Vietnamese": "vi_VN",
#     "Chinese": "zh_CN",
#     "Afrikaans": "af_ZA",
#     "Azerbaijani": "az_AZ",
#     "Bengali": "bn_IN",
#     "Persian": "fa_IR",
#     "Hebrew": "he_IL",
#     "Croatian": "hr_HR",
#     "Indonesian": "id_ID",
#     "Georgian": "ka_GE",
#     "Khmer": "km_KH",
#     "Macedonian": "mk_MK",
#     "Malayalam": "ml_IN",
#     "Mongolian": "mn_MN",
#     "Marathi": "mr_IN",
#     "Polish": "pl_PL",
#     "Pashto": "ps_AF",
#     "Portuguese": "pt_XX",
#     "Swedish": "sv_SE",
#     "Swahili": "sw_KE",
#     "Tamil": "ta_IN",
#     "Telugu": "te_IN",
#     "Thai": "th_TH",
#     "Tagalog": "tl_XX",
#     "Ukrainian": "uk_UA",
#     "Urdu": "ur_PK",
#     "Xhosa": "xh_ZA",
#     "Galician": "gl_ES",
#     "Slovene": "sl_SI"
# }
        # from_lang_code = dict[from_lang]
        # to_lang_code=dict[to_lang]
        tokenizer=BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        print("check 1")
        tokenizer.src_lang=from_lang
        print("check 2")
        encoded_ar = tokenizer(from_text,return_tensors="pt",add_special_tokens=True,lang=from_lang)
        print("check 3")
        generated_tokens=model.generate(**encoded_ar,
                                        forced_bos_token_id=tokenizer.get_lang_id(to_lang))
        data = tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
        return render_template('index.html',data)
    return render_template('index.html')   
if __name__ == '__main__':
    app.run(port=4000,debug=True)