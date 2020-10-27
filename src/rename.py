import os
path = '../test_image/test/test'
file_list = os.listdir(path)

for file in file_list:
    # 补0 10表示补0后名字共10位
    filename = file.zfill(10)
    print(filename)
    new_name = ''.join(filename)
    os.rename(path + '\\' + file, path + '\\' + new_name)
