import logging
import os

def test_log_creation():
    log_dir = './logs'
    log_file = os.path.join(log_dir, 'test_log.log')

    try:
        # 1. 确保 ./logs 目录存在
        os.makedirs(log_dir, exist_ok=True)
        print(f"Directory '{log_dir}' ensured.")

        # 2. [关键] 配置日志记录到文件
        # 我们在这里使用 basicConfig，因为这是这个脚本中的 *第一次* 调用
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=log_file,
            filemode='w'  # 'w' = 覆盖旧日志
        )

        # 3. 写入日志
        logging.info("--- 这是一个测试日志 ---")
        logging.info("如果您看到了这个文件，说明文件权限和 os.makedirs 都正常工作。")
        logging.info("测试成功。")

        print(f"Log test complete. Please check for the file: {log_file}")

    except Exception as e:
        print(f"An error occurred during the logging test:")
        print(e)

if __name__ == "__main__":
    test_log_creation()