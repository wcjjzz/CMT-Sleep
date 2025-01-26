import fnmatch
import os

def hypnogram_txt2new(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 当前时间
    current_time = 0.0
    # 上一睡眠阶段标签
    previous_stage = None
    # 注释列表
    annotations = []
    # 单个标签持续时间
    duration = 5.0

    for line in lines[1:]:
        # 去除空格转换数值类型，确定睡眠阶段
        # Wake:(5)  stage1-4:(3->0)  REM:(4)
        stage = int(line.strip())
        if stage == 5:
            stage_name = "Sleep stage W"
        elif stage == 4:
            stage_name = "Sleep stage R"
        elif stage == 3:
            stage_name = "Sleep stage 1"
        elif stage == 2:
            stage_name = "Sleep stage 2"
        elif stage == 1:
            stage_name = "Sleep stage 3"
        elif stage == 0:
            # stage_name = "unknown Sleep stage ?"
            stage_name = "Sleep stage 4"
        else:
            stage_name = "Sleep stage ?"

        # 判断睡眠阶段是否变换
        if (previous_stage is None) or (previous_stage != stage):
            # 添加注释
            annotations.append(f"{current_time},{duration},{stage_name}")
            # 更新时间
            current_time += duration
            # 更新当前睡眠阶段
            previous_stage = stage
        else:
            # 更新时间
            current_time += duration
            # 取最新注释的持续时间并更新
            last_annotation = annotations[-1]
            parts = last_annotation.split(',')
            new_duration = float(parts[1]) + duration
            annotations[-1] = f"{parts[0]},{new_duration},{parts[2]}"

    # 将注释写入输出文件中
    with open(output_file, 'w') as f:
        f.write("# MNE-Annotations\n")
        f.write("# onset, duration, description\n")
        for annotation in annotations:
            f.write(annotation + '\n')

def convert_directory(directory):
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, 'Hyp*R&K*.txt'):  # 确保处理.txt文件
            input_file = os.path.join(directory, filename)
            output_file = os.path.join(directory, 'mne_new_txt_' + filename)
            hypnogram_txt2new(input_file, output_file)
            print(f"Converted {input_file} to {output_file}")

# 指定包含输入文件的目录（绝对路径，需更改）
directory = r"D:\hkk\项目_可解释性睡眠分期\原始数据集\DREAMS"

# 调用函数转换目录中的每个文件
convert_directory(directory)