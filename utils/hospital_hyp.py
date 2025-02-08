import fnmatch
import os


def hypnogram_txt2new(input_file, output_file):
    # 打开文件，开始操作
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # 定义空注释列表
    annotations = []
    # 定义当前时间
    current_time = 0.0
    # 定义当前睡眠阶段
    previous_stage = None
    # 定义每个标签代表持续时间(秒)
    duration = 30.0

    # 读取
    for line in lines[1:]:  # 忽略文件的第一行（表头）
        # 去除每一行的空格
        line = line.strip()

        # 分割每一行，得到 Onset, Duration, Annotation
        parts = line.split(',')

        if len(parts) < 3:
            continue  # 跳过格式不正确的行

        # 获取Annotation中的最后一个字符（假设它是睡眠阶段标识符）
        stage = parts[2].strip()[-1]  # 提取Annotation最后一个字符

        # 对应每个睡眠阶段的标识符
        stage_name = {
            'W': "Sleep stage W",
            '1': "Sleep stage 1",
            '2': "Sleep stage 2",
            '3': "Sleep stage 3",
            '4': "Sleep stage 4",
            'R': "Sleep stage R"
        }.get(stage, "unknown Sleep stage ?")

        # 若睡眠阶段变化，结束填写前阶段注释
        if previous_stage != stage:
            annotations.append([current_time, duration, stage_name.strip()])
            previous_stage = stage
            # 更新当前时间
            current_time += duration

        # 若睡眠阶段未改变则更新持续时间
        else:
            # 更新注释
            annotations[-1][1] += duration
            # 更新当前时间
            current_time += duration

    # 格式化输出并写入文件
    with open(output_file, 'w') as file:
        file.write("# MNE-Annotations\n")
        file.write("# Onset, Duration, Description\n")
        for annotation in annotations:
            # 右对齐Onset和Duration列，保留1位小数
            onset_str = f"{annotation[0]:>7.1f}"
            duration_str = f"{annotation[1]:>8.1f}"
            description = annotation[2]

            # 格式化输出，每列数据通过逗号分隔，且逗号对齐
            file.write(f"{onset_str}, {duration_str}, {description}\n")


def convert_directory_AASM(directory):
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, '*.txt'):  # 确保处理.txt文件
            input_file = os.path.join(directory, filename)
            output_file = os.path.join(directory, filename[:-4] + '_mne_Annotation.txt')
            hypnogram_txt2new(input_file, output_file)
            print(f"Converted {input_file} to {output_file}")


def convert_directory_RK(directory):
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, '*.txt'):  # 确保处理.txt文件
            input_file = os.path.join(directory, filename)
            output_file = os.path.join(directory, 'mne_new_txt_' + filename)
            hypnogram_txt2new(input_file, output_file)
            print(f"Converted {input_file} to {output_file}")


def main():
    # 指定包含输入文件的目录（绝对路径，需更改）会将新注释文件生成到当前文件夹中
    directory = r"C:\Users\LYT\Desktop\数据集\hospital_depression\depression_pinyin - 副本"
    # 调用函数转换目录中的每个文件
    convert_directory_AASM(directory)


if __name__ == '__main__':
    main()
