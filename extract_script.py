import os

def extract_python_code(root_dir):
    # 遍历目录
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                print("-" * 60)
                print(f"FILE: {file}")
                print(f"PATH: {file_path}")
                print("-" * 60)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        print(f.read())
                except Exception as e:
                    print(f"读取文件出错: {e}")
                
                print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    # "." 表示当前目录，你也可以换成具体的路径
    extract_python_code(".")