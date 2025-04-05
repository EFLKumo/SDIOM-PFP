import ijson
import json
import os
import re
from datetime import datetime, time

def sanitize_filename(filename):
    """
    用下划线替换无效字符，对用作文件名的字符串进行消毒。
    """
    # 用下划线替换文件名中不允许的字符。
    # 对于 Windows，这些字符包括\ / :* ?" < > |
    # 如果需要，我们可以扩展这个列表。
    return re.sub(r'[\\/:*?"<>|]', '_', filename)

def proceed_inbound_data(input_file_path, output_dir):
    """
    流式处理大型 JSON 文件 (JSON Lines 格式)，只处理 "地铁入站" 数据，并按 station 分类输出到以 station 命名的 TXT 文件中。
    假设输入文件每行一个 JSON 对象。
    输出 TXT 文件内容格式为： time1,time2,time3,...

    Args:
        input_file_path (str): 输入 JSON 文件路径
        output_dir (str): 输出 TXT 文件的目录
    """

    station_data = {} # 临时使用字典按 station 存储数据，用于在处理完所有行后写入文件
    discarded_data_count = 0

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f: # 逐行读取文件
            try:
                # 尝试解析每一行作为一个 JSON 对象
                objects = ijson.items(line, '') # 解析单行 JSON
                for obj in objects: # 迭代单行 JSON 中的顶层对象 (通常只有一个)
                    if isinstance(obj, dict) and "data" in obj:
                        for record in obj["data"]:
                            deal_type = record.get("deal_type")
                            time_str = record.get("deal_date") # 获取时间字符串
                            station = record.get("station")
                            line_name = record.get("company_name")

                            # 清洗字段为 None 的数据
                            if (time_str is None or line_name is None or station is None):
                                print(f"忽略非法数据: time={time_str}, line_name={line_name}, station={station}")
                            else:
                                if deal_type == "地铁入站":
                                    try:
                                        time_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").time() # 将字符串转换为 time 对象
                                        cutoff_time = time(12, 0, 0)
                                        if time_obj >= cutoff_time:
                                            print(f"舍弃数据 - 站点: {station}, 时间: {time_str} (时间晚于 12:00)")
                                            discarded_data_count += 1
                                        else:
                                            if station not in station_data:
                                                station_data[station] = []
                                            station_data[station].append(time_str) # 保存原始时间字符串
                                            print(f"已处理入站 time = {time_str}, station = {station}")
                                    except ValueError:
                                        print(f"忽略非法时间格式数据: {time_str}")

                                else:
                                    print("忽略出站数据")

            except ijson.common.IncompleteJSONError:
                print(f"跳过无法解析的 JSON 行 (IncompleteJSONError): {line.strip()}")
            except json.JSONDecodeError:
                print(f"跳过无法解析的 JSON 行 (JSONDecodeError): {line.strip()}")

    # 循环遍历 station_data，为每个 station 创建单独的 TXT 文件
    for station_name, time_list in station_data.items():
        sanitized_station = sanitize_filename(station_name)
        output_file_path = os.path.join(output_dir, f"{sanitized_station}.txt") # 输出为 TXT 文件
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            output_content = ",".join(time_list) # 将 time_list 转换为逗号分隔的字符串
            f_out.write(output_content)
        print(f"地铁入站数据已保存到 {output_file_path}")

if __name__ == "__main__":
    input_file = "2018record.json"
    output_directory = "output_stations"
    proceed_inbound_data(input_file, output_directory)
