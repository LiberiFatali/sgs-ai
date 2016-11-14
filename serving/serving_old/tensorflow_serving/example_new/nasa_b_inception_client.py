# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""Send JPEG image to inception_inference server for classification.
"""

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import inception_inference_pb2
import os, urllib
import numpy as np

server = 'localhost:9000'

folder = '/media/hieunguyen/DATA/Dataset/crowdAI_NASA/test_release'
test_csv = '/media/hieunguyen/DATA/Dataset/crowdAI_NASA/test_release.csv'



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def main():
  host, port = server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = inception_inference_pb2.beta_create_InceptionService_stub(channel)
  
  filename = '/home/hieunguyen/Downloads/cat.jpg'
  with open(filename, 'rb') as fp:
    data = fp.read()
    request = inception_inference_pb2.InceptionRequest()
    request.jpeg_encoded = data
    result = stub.Classify(request, 30.0)  # 10 secs timeout
    print result
    
  ## list of test files
  #with open(test_csv) as csvTest:
    #list_test = csvTest.readlines()[1:]      
  #with open("result_test.csv", "w") as csvResult:
    #csvResult.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'
    #.format('image_url', 'city', 'none', 'unknown', 'aurora', 'astronaut', 'stars', 'black'))
    #for line in list_test:
      #url = line.strip()
      ## get filename
      #f = os.path.basename(url)
      ## Process files
      ##filenames = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]      
      #file_local = os.path.join(folder, f)
      ## download file
      #if not os.path.isfile(file_local):
	#urllib.urlretrieve (url, file_local)
      ## Send request
      #with open(file_local, 'rb') as fp:
	## See inception_inference.proto for gRPC request/response details.
	#data = fp.read()
	#request = inception_inference_pb2.InceptionRequest()
	#request.jpeg_encoded = data
	#result = stub.Classify(request, 30.0)  # 10 secs timeout
	##print url
	##print result.classes[0]
	##print result.scores[0]
	##result_dict = {c:s for c,s in zip(result.classes, result.scores)}
	#scores = softmax(result.scores)
	#result_dict = {'city':0, 'none':0, 'unknown':0, 
		      #'aurora':0, 'astronaut':0, 'stars':0, 'black':0, 
		      #'unused background':0}
	#for c,s in zip(result.classes, scores):
	  #result_dict[c] = s
	
	##print result_dict
	#csvResult.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'
	#.format(url, result_dict['city'], result_dict['none'], result_dict['unknown'], 
		#result_dict['aurora'], result_dict['astronaut'], result_dict['stars'], result_dict['black']))
      ##break


if __name__ == '__main__':
  print 'Running...'
  main()
  print 'Done!'
