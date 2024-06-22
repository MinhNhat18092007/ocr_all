import streamlit as st
from PIL import Image
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from test import copyStateDict
from PIL import Image
import cv2
from skimage import io
import numpy as np
import test
import file_utils
import pandas as pd
from craft import CRAFT
from collections import OrderedDict
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import glob
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import file_utils
import os
import tkinter.messagebox as messagebox
import imgproc


# Tạo yêu cầu đến mô hình
def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")



'''

'''
# CRAFT
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cpu', default=True, type=str2bool, help='Use cpu for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='data_image', type=str, help='đường dẫn tới ảnh đầu vào')
parser.add_argument('--refine', default=True, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str,
                    help='pretrained refiner model')

args = parser.parse_args()



#########################################################################################
csv_columns = ['x_top_left', 'y_top_left', 'x_top_right', 'y_top_right', 'x_bot_right', 'y_bot_right', 'x_bot_left',
                'y_bot_left']
# load net
net = CRAFT()  # initialize
print('Đang thực hiện load weight (' + args.trained_model + ')')
'''
nhảy sang file test, đưa vào train model
'''
if args.cpu:
    net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
else:
    net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

if args.cpu:
    net = net.cpu()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

net.eval()
# LinkRefiner Đoạn này code không chạy qua nên không cần đọc vì weight đã load ở cái bên trên
# còn refine để mặc định bên trên là False nên sẽ bị bỏ qua
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
refine_net = None
if args.refine:
    from refinenet import RefineNet

    refine_net = RefineNet()
    print('Đang thực hiện load weight (' + args.refiner_model + ')')
    if args.cpu:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
        refine_net = refine_net.cpu()
        refine_net = torch.nn.DataParallel(refine_net)
    else:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

    refine_net.eval()
    args.poly = True


config = Cfg.load_config_from_name('vgg_transformer')
config['export'] = 'transformerocr_checkpoint.pth'
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False

detector = Predictor(config)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

    
# Tạo tiêu đề và phần tải lên hình ảnh
st.title("Trích xuất thông tin từ căn cước công dân")
uploaded_file = st.file_uploader("Tải lên ảnh căn cước công dân", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị hình ảnh tải lên
    image = Image.open(uploaded_file)
    image.save("uploaded_image.jpg")
    st.image(image, caption='Hình ảnh căn cước', use_column_width=True)
    import tempfile
    # Lưu trữ tạm thời và lấy đường dẫn
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        # Save the image to the temp file
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Hiển thị đường dẫn tạm thời
    print(f"Đường dẫn tạm thời của file: {temp_file_path}")
    # Nút chạy
    if st.button("Run"):
        print("ok")
        image_path = "uploaded_image.jpg"
        k = 1
        crop_folder = "crop_Word"
        result_folder = "Results"
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text, det_scores = test.test_net(net, image, args.text_threshold, args.link_threshold,
                                                                args.low_text, args.cpu, args.poly, args, refine_net)
        bbox_score = {}

        def crop_polygon(image, vertices, box_num1):
            # Tạo mặt nạ
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.int32(vertices)], 255)

            # Tìm bounding rect để crop vùng chứa đa giác
            rect = cv2.boundingRect(np.int32(vertices))

            # Crop và lấy hình ảnh con theo bounding rect
            cropped = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

            # Tạo mặt nạ cho vùng đã crop
            cropped_mask = mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

            # Lọc vùng bằng mặt nạ
            result = cv2.bitwise_and(cropped, cropped, mask=cropped_mask)
            crop_path = os.path.join(crop_folder, f"crop_{box_num1 + 1}.jpg")
            cv2.imwrite(crop_path, result)
            return result

        if len(bboxes) == 0:
            with open(f"data_text//text_{k}.txt", "w", encoding="utf-8") as f:
                f.write(" ")

        else:
            # for box_num, item in enumerate(bboxes):
            #     # Crop the bbox from the image
            #     pts = np.array(item, np.int32).reshape((-1, 1, 2))
            #     rect = cv2.boundingRect(pts)
            #     x, y, w, h = rect
            #     cropped_img = image[y:y+h, x:x+w].copy()
            #     crop_path = os.path.join(crop_folder, f"crop_{box_num + 1}.jpg")
            #     cv2.imwrite(crop_path, cropped_img)
            for box_num in range(len(bboxes)):
                item = bboxes[box_num]
                data = np.array([[int(item[0][0]), int(item[0][1]), int(item[1][0]), int(item[1][1]), int(item[2][0]),
                                    int(item[2][1]), int(item[3][0]), int(item[3][1])]])
                csvdata = pd.DataFrame(data, columns=csv_columns)
                csvdata.to_csv(f'data{k}.csv', index=False, mode='a', header=False)

            # save score text
            filename, file_ext = os.path.splitext(os.path.basename(image_path))
            mask_file = result_folder + "/res_" + filename + '_mask.jpg'  # tạo đường dẫn file bản đồ nhiệt

            cv2.imwrite(mask_file, score_text)  # in ra bản đồ nhiệt
            #
            file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)

            cropped_images = []
            for i, box in enumerate(bboxes):
                cropped = crop_polygon(image, box, i)
                cropped_images.append(cropped)


            print(f"Đã cắt {len(cropped_images)} vùng bounding box.")
            path = glob.glob("crop_Word/*.jpg")
            cv_img = [str(detector.predict(Image.open(f'crop_Word/crop_' + str(i + 1) + '.jpg'))) for i in
                        range(len(bboxes))]
            print(cv_img)
            # from google.generativeai.types import HarmCategory, HarmBlockThreshold
            # import google.generativeai as genai

            # genai.configure(api_key="AIzaSyAH4ayK6nL71wxPtuYOCe32OdZVZAANWic")

            # # Khởi tạo mô hình
            # model = genai.GenerativeModel(model_name='gemini-1.5-flash')

            # # Thiết lập safe_setting cho các loại harm có sẵn và hợp lệ
            # safety_settings = [
            #     {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
            #     {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            #     {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            #     {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE}
            # ]
            # print("ok")
            # response = model.generate_content(
            #     [f"tìm tên, ngày sinh, nơi cư trú, số căn cước, hạn sử dụng trong mảng sau  {cv_img}, chỉ trả về tên, ngày sinh, nơi cư trú, số căn cước,hạn sử dụng mà model tìm thấy được, không trả lời thêm gì, ví dụ 'NGUYỄN THANH SANG, 18/05/1981, 223/11 Kv Bỉnh- Dương, Long Hòa, Bình Thủy, Cần Thơ, 092081007131, 18/05/2041  '"],
            #     safety_settings=safety_settings
            # )
            # # print(f"tìm tên, ngày sinh, nơi cư trú, số căn cước, hạn sử dụng trong mảng sau  {cv_img}, chỉ trả về tên, ngày sinh, nơi cư trú, số căn cước,hạn sử dụng mà model tìm thấy được, không trả lời thêm gì, ví dụ 'NGUYỄN THANH SANG, 18/05/1981, 223/11 Kv Bỉnh- Dương, Long Hòa, Bình Thủy, Cần Thơ, 092081007131, 18/05/2041  '")
            # print(response.text)
            

            for box in bboxes:
                cv2.polylines(image, [np.int32(box)], isClosed=True, color=(0, 255, 0), thickness=1)
            
            plt.figure(figsize=(20, 20))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Detected Text Bounding Boxes')
            plt.show()
            print(f"đã load xong ảnh {k + 1}")

        # # Hiển thị kết quả
        st.subheader("Kết quả trích xuất:")
        st.text_area("ALL TEXT",cv_img)
        # st.text_area("Tên", response.text.get('name', ''))
        # st.text_area("Ngày sinh", response.text.get('dob', ''))
        # st.text_area("Nơi cư trú", response.text.get('address', ''))
        # st.text_area("Số căn cước", response.text.get('id_number', ''))
        # st.text_area("Hạn sử dụng", response.text.get('expiry', ''))
