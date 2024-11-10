import os
import json
import csv

# 设置log文件夹路径
log_dir = "logs/scaling-01B-C4-M32-lr1e-4-ori"
checkpoints = [f for f in os.listdir(log_dir) if f.startswith("checkpoint")]

# 按数字顺序排序
checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))

# 需要提取的字段
metrics = {
    "MME": ("mme", "mme_percetion_score,none"),
    "MMER-C": ("mmerealworld_cn", "mme_realworld_score,none"),
    "GQA": ("gqa_lite", "exact_match,none"),
    "VQAv2": ("vqav2_val_lite", "exact_match,none"),
    "realworldqa": ("realworldqa", "exact_match,flexible-extract"),
    "OCRBench": ("ocrbench", "ocrbench_accuracy,none"),
    "TextVQA": ("textvqa_val_lite", "exact_match,none"),
    "WebSRC": ("websrc_val", "websrc_squad_f1,none"),
    "ChartQA": ("chartqa_lite", "relaxed_overall,none"),
    "AI2D": ("ai2d_lite", "exact_match,flexible-extract"),
    "DOCVQA": ("docvqa_val_lite", "anls,none"),
    "POPE": ("pope_adv", "pope_accuracy,none"),
}

# 初始化表头
header = ["Steps"] + [str(int(cp.split('-')[-1])) for cp in checkpoints]

# 初始化数据表格
table = {metric: [metric] + [""] * len(checkpoints) for metric in metrics.keys()}
step_averages = []
# 递归查找以 _results.json 结尾的文件
def find_results_json(directory):
    result_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith("_results.json"):
                result_files.append(os.path.join(root, file))
    return result_files

# 遍历每个checkpoint文件夹并提取数据
for idx, checkpoint in enumerate(checkpoints):
    checkpoint_path = os.path.join(log_dir, checkpoint)
    
    # 查找所有以 _results.json 结尾的文件
    result_files = find_results_json(checkpoint_path)
    
    if result_files:
        # 假设每个checkpoint只处理第一个找到的 _results.json 文件
        result_file = result_files[0]
        with open(result_file, 'r') as f:
            data = json.load(f)["results"]
            
            # 填充对应的指标
            metric_values = []
            for metric, (dataset, field) in metrics.items():
                if dataset in data and field in data[dataset]:
                    if 'MME' == metric:
                        score = data['mme']['mme_cognition_score,none'] * 100 + data['mme']['mme_percetion_score,none'] * 100
                        score /= 2334
                    else:
                        score = data[dataset][field] * 100
                    table[metric][idx + 1] = score
                    metric_values.append(score)
                    
        # 计算每个 step 的均值
        if metric_values:
            step_avg = sum(metric_values) / len(metric_values)
            step_averages.append(step_avg)
        else:
            step_averages.append(None)

max_avg = max(filter(lambda x: x is not None, step_averages))
max_idx = step_averages.index(max_avg)

# 查找最早在 5% 范围内的 step
threshold = max_avg * 0.95
closest_idx = next(i for i, avg in enumerate(step_averages) if avg is not None and avg >= threshold)

# 查找最早在 1% 范围内的 step
threshold = max_avg * 0.99
closest_idx_99 = next(i for i, avg in enumerate(step_averages) if avg is not None and avg >= threshold)

# 查找最早在 1% 范围内的 step
threshold = max_avg * 0.90
closest_idx_90 = next(i for i, avg in enumerate(step_averages) if avg is not None and avg >= threshold)


table["Average"] = ["Average"] + step_averages

summary_row = ["Summary"]
summary_row = ["Summary"]
for idx in range(len(header) - 1):
    idx_summary = ''
    if idx == max_idx:
        idx_summary += f"Max Avg: {max_avg:.2f}\n"
    if idx == closest_idx:
        idx_summary += f"Closest 5% Step: {step_averages[closest_idx]:.2f}\n"
    if idx == closest_idx_99:
        idx_summary = f"Closest 1% Step: {step_averages[closest_idx_99]:.2f}"
    if idx == closest_idx_90:
        idx_summary = f"Closest 10% Step: {step_averages[closest_idx_90]:.2f}"
    summary_row.append(idx_summary)

table['Summary'] = summary_row
# 将结果写入CSV文件
output_file = "results_summary.csv"
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)  # 写入表头
    for metric, row in table.items():
        writer.writerow(row)  # 写入每一行

print(f"CSV file saved as {output_file}")
print(f"Maximum average found at step: {header[max_idx + 1]} with value: {max_avg:.2f}")
print(f"First step within 5% of maximum average: {header[closest_idx + 1]} with value: {step_averages[closest_idx]:.2f}")