import tensorflow as tf
import os

print 'Running...'

folderpath = 'SgsImageNet_TFRecord_balance_inc/'

lstFileNames = [f for f in os.listdir(folderpath) if 'train' in f]
nTotal = 0
for f in lstFileNames:
  nCount = 0
  filename = f
  for serialized_example in tf.python_io.tf_record_iterator(folderpath + filename):
    nCount += 1
  nTotal += nCount
  
print '*** Total images in records: ', nTotal
