import os
import logging
from datetime import datetime, timedelta

def advanced_delete_zero_byte_files(directory_path, min_age_days=0, dry_run=False):
    """
    高级版本：删除指定目录下所有大小为0的文件，可设置文件最小存在时间和模拟运行
    
    参数:
        directory_path (str): 要清理的目录路径
        min_age_days (int): 文件最小存在天数（默认0，即所有零字节文件）
        dry_run (bool): 模拟运行，只显示将要删除的文件而不实际删除
        
    返回:
        tuple: (成功删除的文件数, 失败的文件数, 模拟运行时的文件数)
    """
    # 初始化计数器
    deleted_count = 0
    failed_count = 0
    would_delete_count = 0
    
    # 验证目录是否存在
    if not os.path.exists(directory_path):
        logging.error(f"目录不存在: {directory_path}")
        return (0, 1, 0)
    
    if not os.path.isdir(directory_path):
        logging.error(f"路径不是目录: {directory_path}")
        return (0, 1, 0)
    
    try:
        # 计算最小修改时间阈值
        min_age_time = datetime.now() - timedelta(days=min_age_days)
        
        # 遍历目录中的所有文件
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                try:
                    # 检查文件大小是否为0
                    if os.path.getsize(file_path) == 0:
                        # 检查文件是否足够旧（如果设置了最小天数）
                        if min_age_days > 0:
                            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                            if file_mtime > min_age_time:
                                continue  # 文件太新，跳过
                        
                        if dry_run:
                            # 模拟运行模式，只计数不删除
                            logging.info(f"将删除零字节文件: {file_path}")
                            would_delete_count += 1
                        else:
                            # 实际删除文件
                            os.remove(file_path)
                            logging.info(f"已删除零字节文件: {file_path}")
                            deleted_count += 1
                except OSError as e:
                    logging.error(f"无法处理文件 {file_path}: {e}")
                    failed_count += 1
                except Exception as e:
                    logging.error(f"处理文件时出错 {file_path}: {e}")
                    failed_count += 1
    
    except Exception as e:
        logging.error(f"遍历目录时出错: {e}")
        return (deleted_count, failed_count + 1, would_delete_count)
    
    return (deleted_count, failed_count, would_delete_count)



def delete_zero_size_files(path: str):
    """
    删除指定路径下大小为 0 字节的文件
    :param path: 目标目录路径
    """
    if not os.path.isdir(path):
        print(f"Error: {path} 不是一个有效目录")
        return

    deleted_count = 0
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)

        if os.path.isfile(filepath):  # 只处理普通文件
            if os.path.getsize(filepath) == 0:
                try:
                    os.remove(filepath)
                    print(f"已删除无效文件: {filepath}")
                    deleted_count += 1
                except Exception as e:
                    print(f"删除失败 {filepath}: {e}")

    print(f"共删除 {deleted_count} 个无效文件")


# 示例调用
if __name__ == "__main__":
    target_path = "/mnt/data/PMT/R8520_406/"   # 修改为你的路径
    delete_zero_size_files(target_path)


## 使用示例
#if __name__ == "__main__":
#    # 配置日志
#    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
#    # 指定要清理的目录路径
#    target_directory = "/mnt/data/PMT/R8520_406/"  # 替换为实际路径
    
#    # 执行清理（只删除存在至少7天的零字节文件）
#    deleted, failed, would_delete = advanced_delete_zero_byte_files(
#        target_directory, 
#        min_age_days=7,
#        dry_run=False  # 设置为True可以先预览将要删除的文件
#    )
    
#    # 输出结果
#    print(f"操作完成: 成功删除 {deleted} 个文件, {failed} 个文件处理失败")
