import os
import glob
import shutil
path= 'F:/BaiduNetdiskDownload/Facescrub_data/Facescrub/Facescrub'
filelist=os.listdir(path)

for files in filelist:#遍历所有文件
    #print files
    Olddir=os.path.join(path,files)#匹配路径和文件名
    if os.path.isdir(Olddir):#如果是文件夹则跳过
        continue
    filename=os.path.splitext(files)[0]#将文件名和后缀名分离

    filename_key=filename[:11]# 截取文件名中的关键字符串
    newdir=os.path.join(path,filename_key+'*')# 通过glob 的通用匹配查询关键词特征的文件，并匹配路径
    #print newdir
    same_sation=glob.glob(newdir)#查询具有通用匹配字符的文件
    for i in same_sation:#历遍所有文件
        newfiledir=path+'\\'+filename_key#确定新的文件路径，此处根据关键词创建新的文件夹
        isexist=os.path.exists(newfiledir)# 判断文件夹是否存在
        if not isexist : # 如果不存在则创建文件路径
            os.mkdir(newfiledir)
        shutil.move(i,newfiledir)# 移动文件
