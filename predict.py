import json
import os
import pickle
from datasets import load_dataset
from multiprocessing import Pool
from PIL import Image
import warnings
from tqdm import tqdm
import json
import hashlib
import shutil
import gc

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None  # 取消像素限制

names = ['CLEVR-Math(MathV360K)', 'FigureQA(MathV360K)', 'GEOS(MathV360K)', 'GeoQA+(MathV360K)', 'Geometry3K(MathV360K)', 'IconQA(MathV360K)', 'MapQA(MathV360K)', 'PMC-VQA(MathV360K)', 'Super-CLEVR(MathV360K)', 'TabMWP(MathV360K)', 'UniGeo(MathV360K)', 'VizWiz(MathV360K)', 'ai2d(cauldron,llava_format)', 'ai2d(gpt4v)', 'ai2d(internvl)', 'allava_instruct_laion4v', 'allava_instruct_vflan4v', 'aokvqa(cauldron,llava_format)', 'chart2text(cauldron)', 'chartqa(cauldron,llava_format)', 'chrome_writting', 'clevr(cauldron,llava_format)', 'diagram_image_to_text(cauldron)', 'dvqa(cauldron,llava_format)', 'figureqa(cauldron,llava_format)', 'geo170k(align)', 'geo170k(qa)', 'geo3k', 'geomverse(cauldron)', 'hateful_memes(cauldron,llava_format)', 'hitab(cauldron,llava_format)', 'hme100k', 'iam(cauldron)', 'iconqa(cauldron,llava_format)', 'iiit5k', 'image_textualization(filtered)', 'infographic(gpt4v)', 'infographic_vqa', 'infographic_vqa_llava_format', 'intergps(cauldron,llava_format)', 'k12_printing', 'llavar_gpt4_20k', 'lrv_chart', 'lrv_normal(filtered)', 'magpie_pro(l3_80b_mt)', 'magpie_pro(l3_80b_st)', 'magpie_pro(qwen2_72b_st)', 'mapqa(cauldron,llava_format)', 'mathqa', 'mavis_math_metagen', 'mavis_math_rule_geo', 'multihiertt(cauldron)', 'orand_car_a', 'raven(cauldron)', 'rendered_text(cauldron)', 'robut_sqa(cauldron)', 'robut_wikisql(cauldron)', 'robut_wtq(cauldron,llava_format)', 'scienceqa(cauldron,llava_format)', 'scienceqa(nona_context)', 'screen2words(cauldron)', 'sharegpt4o', 'sharegpt4v(coco)', 'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)', 'sroie', 'st_vqa(cauldron,llava_format)', 'tabmwp(cauldron)', 'tallyqa(cauldron,llava_format)', 'textcaps', 'textocr(gpt4v)', 'tqa(cauldron,llava_format)', 'vision_flan(filtered)', 'vistext(cauldron)', 'visual7w(cauldron,llava_format)', 'visualmrc(cauldron)', 'vqarad(cauldron,llava_format)', 'vsr(cauldron,llava_format)', 'websight(cauldron)']
# names = ['CLEVR-Math(MathV360K)', 'GEOS(MathV360K)',]
output_dir = "/dev/shm/data/LLaVA-OneVision-Processed/json_files"  # 设置保存文件的目录
image_folder = '/dev/shm/data/LLaVA-OneVision-Processed/images'
# 读取 JSON 文件
with open('/home/yueke/data/llava_one_vision.json', 'r') as file:
    data = json.load(file)

image_files = ['a8871481d8bd625bb265453d7445a9ef',
               '3b82aee7ca9aab657dafd3732eaff99e',]

for name in names:
    dataset = load_dataset("/home/yueke/data/LLaVA-OneVision-Data", name=name, split="train")
    for item in dataset:
        image = item['image']
        image_id = item['id']
        if image_id in image_files:
            image.save(os.path.join(image_folder, image["image"]))