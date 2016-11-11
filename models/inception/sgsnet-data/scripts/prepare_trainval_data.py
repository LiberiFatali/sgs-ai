import os

## validation set
#valFile = open('ft_val.txt', 'r')
#lines = valFile.readlines()
#valFile.close() 

#oldValPath = '/home/hieunguyen/caffe/examples/sgsnet/data/images'
#newValPath = '/home/hieunguyen/tensorflow/models/inception/sgsnet-data/raw-data/validation'

#for l in lines:
  #oldFilePath = l.split(' ')[0]
  #newFilePath = oldFilePath.replace(oldValPath, newValPath)  
  #newFolderPath = os.path.dirname(newFilePath)
  #if not os.path.exists(newFolderPath):
    #os.makedirs(newFolderPath)
  ##print newFolderPath
  ##print oldFilePath
  ##print newFilePath
  #try:
    #if not os.path.islink(newFilePath):
      #os.symlink(oldFilePath, newFilePath)
  #except Exception, e:
    #print e, oldFilePath
    
  #break

## train set  
#trainFile = open('ft_train.txt', 'r')
#lines = trainFile.readlines()
#trainFile.close() 

#oldTrainPath = '/home/hieunguyen/caffe/examples/sgsnet/data/images'
#newTrainPath = '/home/hieunguyen/tensorflow/models/inception/sgsnet-data/raw-data/train'

#for l in lines:
  #oldFilePath = l.split(' ')[0]
  #newFilePath = oldFilePath.replace(oldTrainPath, newTrainPath)  
  #newFolderPath = os.path.dirname(newFilePath)
  #if not os.path.exists(newFolderPath):
    #os.makedirs(newFolderPath)
  ##print newFolderPath
  ##print oldFilePath
  ##print newFilePath
  #try:
    #if not os.path.islink(newFilePath):
      #os.symlink(oldFilePath, newFilePath)
  #except Exception, e:
    #print e, oldFilePath
    
  ##break
  

# ImageNet val set
with open('imagenet_2012_validation_synset_labels.txt', 'r') as valImgNetLabel:  
  lines = valImgNetLabel.read().splitlines() 
  
srcValPath = '/media/hieunguyen/DATA/Dataset/ImageNet/ILSVRC2015/Data/CLS-LOC/val'
destValPath = '/home/hieunguyen/tensorflow/models/inception/sgsnet-data/raw-data/validation'

files = [ f for f in os.listdir(srcValPath) if os.path.isfile(os.path.join(srcValPath, f)) ]
files = sorted(files)

#count = 0
for f, l in zip(files, lines):  
  old_f = os.path.join(srcValPath, f)
  new_f = os.path.join(destValPath, l, f)
  #print old_f, l
  #print new_f
  
  try:
    if not os.path.islink(new_f):
      os.symlink(old_f, new_f)
  except Exception, e:
    print e, old_f, new_f
  
  #count += 1
  #if count == 1:
    #break

  






