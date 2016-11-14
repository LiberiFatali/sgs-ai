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

#!/usr/grte/v4/bin/python2.7

"""Send JPEG image to inception_inference server for classification.
"""

import os
import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy as np
import tensorflow as tf

from tensorflow_serving.example import inception_inference_pb2

# HieuNM
import json
import requests
import time
import web

from requests.adapters import HTTPAdapter
#from urllib3.poolmanager import PoolManager
from requests.packages.urllib3.poolmanager import PoolManager
import ssl
import datetime


tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'inception_inference service host:port')
#tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

HOST, PORT = FLAGS.server.split(':')

NUM_CLASSES = 5
WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
SYNSET_FILE = os.path.join(WORKING_DIR, 'sgsnet_synsets.txt')
METADATA_FILE = os.path.join(WORKING_DIR, 'sgsnet_metadata.txt')


class Tfserving_Classify:
  def GET(self):
    web.header('Access-Control-Allow-Origin', '*')    
    web.header('Content-Type', 'application/json')    
    incomingIP = web.ctx['ip']   
    #if incomingIP == '172.16.2.113':
    #  return        
    user_data = web.input()
    if 'url' in user_data: # to check if the url-object is created
      img_url = user_data.url
      img_id = str(time.time())
      if 'id' in user_data: 
	if user_data.id != '':
	  img_id = user_data.id		  
      # Download remote image to local
      if img_url != '':	
	try:
	  resp = sess.get(img_url, stream=False)
	  if resp.status_code == 200 and 'image' in resp.headers['content-type']:	    
	    resp_data = resp.content
	  else:
	    print str(datetime.datetime.now()) + '; '+ img_url.encode('ascii', 'replace')  + '; Cannot DOWNLOAD target file!'	    
	    output = {'File':img_url,'Label':'Cannot DOWNLOAD target file!','Label_Confidence': 'NA','Recognized_Text':'NA'}	    
	    yield json.dumps(output)
	    return	  
	except Exception, e:	  
	  print >> sys.stderr, e
	  sys.stdout.flush()	  
	  #output = {'File':img_url,'Label':'Cannot ACCESS target file or it is not an image!','Label_Confidence': 'NA','Recognized_Text':'NA'}
	  #yield json.dumps(output)
	  yield 'Error accessing remote file'
	  return
	
	try:
	  sLabel, fConf = classify(HOST, PORT, resp_data, 5)	
	  #strLabel = str(sLabel)
	  #strConf  = str(fConf)	
	  #output = {'File':img_url,'Label':strLabel,'Probability':strConf}
	  jsLabel = json.loads(json.dumps(sLabel))
	  jsConf = json.loads(json.dumps(fConf))
	  sAdultContent = "No"
	  if jsLabel[0] in ['ass', 'penis', 'vagina', 'womanbreast']:
	    if jsConf[0] >= 0.7:
	      sAdultContent = "Yes"
	    else: 
	      sAdultContent = "NotSure"
	  output = {"File":img_url,"Label":jsLabel,"Probability":jsConf,"Adult Content":sAdultContent}
	  del resp_data
	  yield json.dumps(output, sort_keys=True)
	  return
	except Exception, e:	  
	  print >> sys.stderr, e
	  sys.stdout.flush()
	  #output = {'File':img_url,'Label': e,'Label_Confidence': 'NA','Recognized_Text':'NA'}
	  #yield json.dumps(output)
	  yield 'Error classifying image'
	  return
	  
	##strLabel = str(sLabel)
	##strConf  = str(fConf)	
	##output = {'File':img_url,'Label':strLabel,'Probability':strConf}
	#jsLabel = json.loads(json.dumps(sLabel))
	#jsConf = json.loads(json.dumps(fConf))
	#sAdultContent = "No"
	#if jsLabel[0] in ['ass', 'penis', 'vagina', 'womanbreast']:
	  #if jsConf[0] >= 0.7:
	    #sAdultContent = "Yes"
	  #else: 
	    #sAdultContent = "NotSure"
	#output = {"File":img_url,"Label":jsLabel,"Probability":jsConf,"Adult Content":sAdultContent}
	#del resp_data
	#yield json.dumps(output, sort_keys=True)
	#return
	
    
  def POST(self):
    pass
  
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def classify(host, port, image_data, num_classes_returned=5):  
  #channel = implementations.insecure_channel(host, int(port))
  #stub = inception_inference_pb2.beta_create_InceptionService_stub(channel)
  ## Create label->synset mapping
  #synsets = []
  #with open(SYNSET_FILE) as f:
    #synsets = f.read().splitlines()
  ## Create synset->metadata mapping
  #texts = {}
  #with open(METADATA_FILE) as f:
    #for line in f.read().splitlines():
      #parts = line.split('\t')
      #assert len(parts) == 2
      #texts[parts[0]] = parts[1]
  ## Send request
  ##with open(FLAGS.image, 'rb') as f:
  ## See inception_inference.proto for gRPC request/response details.  
  #request = inception_inference_pb2.InceptionRequest()
  request.jpeg_encoded = image_data
  result = stub.Classify(request, 60.0)  # 60 secs timeout
  softmax_probs = softmax(result.scores)
  sLabel = []
  fConf = []
  #out_string = ''  
  for i in range(num_classes_returned):
    index = result.classes[i]
    score = result.scores[i]
    prob = softmax_probs[i]
    #print '%f %f : %s' % (prob, score, texts[synsets[index - 1]])
    #sLabel.append(texts[synsets[index - 1]])
    sLabel.append(index)
    fConf.append(prob)
    #out_string += str(prob) + ' ' + str(score) + ' ' + str(texts[synsets[index - 1]]) + ' ; '
  
  #sLabel = texts[synsets[result.classes[0] - 1]]
  #fConf = softmax_probs[0]
  return sLabel, fConf


# handle binary upload
class Tfserving_Upload: 
  def GET(self):
    pass
  
  def POST(self):
    web.header('Access-Control-Allow-Origin', '*')
    #web.header('Content-type', "text/plain; charset=utf-8")
    #web.header('Content-Type', 'application/json')    
    # parse inputs     
    post_data = web.data()   
    try:
      # load as an image
      #input_image_pil = Image.open(StringIO(post_data))
      sLabel, fConf = classify(HOST, PORT, post_data, 5)
      jsLabel = json.loads(json.dumps(sLabel))
      jsConf = json.loads(json.dumps(fConf))
      sAdultContent = "No"
      if jsLabel[0] in ['ass', 'penis', 'vagina', 'womanbreast']:
	if jsConf[0] >= 0.7:
	  sAdultContent = "Yes"
	else: 
	  sAdultContent = "NotSure"
      output = {"Label":jsLabel,"Probability":jsConf,"Adult Content":sAdultContent}
      
      return output
      
    except Exception, e:
      print >> sys.stderr, e
      sys.stdout.flush()
      return 'Cannot RECEIVE target file or it is not an image!'        


class SSLAdapter(HTTPAdapter):
    '''An HTTPS Transport Adapter that uses an arbitrary SSL version.'''
    def __init__(self, ssl_version=None, **kwargs):
        self.ssl_version = ssl_version

        super(SSLAdapter, self).__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(num_pools=connections,
                                       maxsize=maxsize,
                                       block=block,
                                       ssl_version=self.ssl_version)

if __name__ == '__main__':  
    # Map urls to class 
  urls = (  
    '/tfserving_classify', 'Tfserving_Classify',
    '/tfserving_upload', 'Tfserving_Upload'
    ) 
  
  #tf.app.run()
  # Use a `Session` instance to customize how `requests` handles making HTTP requests.
  sess = requests.Session()
  # `mount` a custom adapter that retries failed connections for HTTP and HTTPS requests.
  sess.mount("http://", requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=3))
  #sess.mount("http://", requests.adapters.HTTPAdapter(max_retries=3))
  sess.mount("https://", SSLAdapter(ssl_version=ssl.PROTOCOL_TLSv1, max_retries=3))
  
  
  channel = implementations.insecure_channel(HOST, int(PORT))
  stub = inception_inference_pb2.beta_create_InceptionService_stub(channel)
  request = inception_inference_pb2.InceptionRequest()
  
  # Create label->synset mapping
  synsets = []
  with open(SYNSET_FILE) as f:
    synsets = f.read().splitlines()
  # Create synset->metadata mapping
  texts = {}
  with open(METADATA_FILE) as f:
    for line in f.read().splitlines():
      parts = line.split('\t')
      assert len(parts) == 2
      texts[parts[0]] = parts[1]

  # Web application to process request   
  web.config.debug = False 
  app = web.application(urls, globals())  
  app.run()
