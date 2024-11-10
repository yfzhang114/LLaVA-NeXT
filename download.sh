# 下载文件
wget http://images.cocodataset.org/zips/train2017.zip -O train2017.zip
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -O gqa_images.zip
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -O textvqa_train_val_images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -O vg_images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -O vg_images2.zip

# 创建目标目录
mkdir -p coco/train2017 gqa/images ocr_vqa/images textvqa/train_val_images vg/VG_100K vg/VG_100K_2

# 解压文件到指定目录
unzip train2017.zip -d coco/train2017
unzip gqa_images.zip -d gqa/images
unzip textvqa_train_val_images.zip -d textvqa/train_val_images
unzip vg_images.zip -d vg/VG_100K
unzip vg_images2.zip -d vg/VG_100K_2

# 针对 OCR-VQA 数据集的特殊处理
# 假设 OCR-VQA 文件已经下载在 ocr_vqa_source_folder
# 将其重命名为 .jpg 并移动到 ocr_vqa/images
for file in ocr_vqa_source_folder/*; do
    if [[ "${file}" != *.jpg ]]; then
        mv "$file" "ocr_vqa/images/$(basename "${file%.*}.jpg")"
    else
        mv "$file" ocr_vqa/images/
    fi
done

# 清理下载的压缩包（如果不再需要）
rm -f train2017.zip gqa_images.zip textvqa_train_val_images.zip vg_images.zip vg_images2.zip

echo "所有文件已解压并整理到指定目录结构。"
