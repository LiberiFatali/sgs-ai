#!/usr/bin/python

"""Send JPEG image to inception_inference server for classification.
"""

## HieuNM
#import threading

## m-RNN
#import os, sys
#sys.path.append('/home/hieunguyen/m-RNN/TF-mRNN/py_lib')
#from common_utils import CommonUtiler
#from tf_mrnn_decoder import mRNNDecoder
#from vision import ImageFeatureExtractor

import datetime, json, requests, ssl, StringIO, time
import pytesseract
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.poolmanager import PoolManager
# PIL must be imported before tensorflow. It's totally unclear why
# and probably has some kind of byzantine and bizarre reason (likely
# because the Pillow version vs. the one tensorflow uses are different).
from PIL import Image
from urlparse import urlparse
#
import logging

# This is a placeholder for a Google-internal import.
from grpc.beta import implementations
import numpy as np
#import tensorflow as tf
import inception_inference_pb2
#
from celery import Celery
from celery.utils.log import get_task_logger
from functools import wraps
from flask import abort, Flask, request, Response
application = Flask(__name__)

### Setup Celery  
def make_celery(app):
  celery = Celery(app.import_name, 
		  broker=app.config['CELERY_BROKER_URL'])
  celery.conf.update(app.config)
  TaskBase = celery.Task
  class ContextTask(TaskBase):
    abstract = True
    def __call__(self, *args, **kwargs):
      with app.app_context():
	return TaskBase.__call__(self, *args, **kwargs)
  celery.Task = ContextTask
  return celery

application.config.update(
  CELERY_BROKER_URL='redis://localhost:6379/0',
  CELERY_RESULT_BACKEND='redis://localhost:6379/0',
  CELERYD_CONCURRENCY = 3,
  CELERYD_LOG_FILE = 'logs/sgsnet/sgsnet_client.log',
)
celery = make_celery(application)

logger = get_task_logger(__name__)
logger.setLevel(logging.INFO) 

@celery.task(name="celery_tasks.write_log")
def write_log(text):  
  logger.info(text)  
  return
###

### Setup protocol for inception gRPC server
HOST = "localhost"
#PORT = 9000
PORT = 8888
# set up protocol
#channel = implementations.insecure_channel(HOST, PORT)
#stub = inception_inference_pb2.beta_create_InceptionService_stub(channel)
##grpc_request = inception_inference_pb2.InceptionRequest()  

#WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
#SYNSET_FILE = os.path.join(WORKING_DIR, 'sgsnet_synsets.txt')
#METADATA_FILE = os.path.join(WORKING_DIR, 'sgsnet_metadata.txt')
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
###    

### Setup session object
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

## For caption
## set up paths
#mrnn_model_path = '/home/hieunguyen/m-RNN/TF-mRNN/trained_models/coco_caption/mrnn_GRU_570K.ckpt'
#mrnn_config_path = '/home/hieunguyen/m-RNN/TF-mRNN/model_conf/mrnn_GRU_conf.py'
#mrnn_vocab_path = '/home/hieunguyen/m-RNN/TF-mRNN/trained_models/coco_caption/mscoco_mc3_vocab'
#img_model_path = '/home/hieunguyen/m-RNN/TF-mRNN/external/tf_cnn_models/inception_v3.pb'

## initilize feature extractor and sentence decoder
#cu = CommonUtiler()
#config = cu.load_config(mrnn_config_path)
#ife = ImageFeatureExtractor(img_model_path)
#decoder = mRNNDecoder(config, 'demo', mrnn_vocab_path)

#decoder.load_model(mrnn_model_path)
#BEAM_SIZE = 3

  
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

# Use a `Session` instance to customize how `requests` handles HTTP requests
sess = requests.Session()
# custom adapter that retries failed connections for HTTP and HTTPS requests
sess.mount("http://", requests.adapters.HTTPAdapter(pool_connections=500, 
						    pool_maxsize=500,
						    max_retries=3))  
sess.mount("https://", SSLAdapter(pool_connections=500, pool_maxsize=500, 
				  max_retries=3, 
				  ssl_version=ssl.PROTOCOL_TLSv1))  
###

### RESTful APIs
# Token based authentication
SECRET_KEY = "373d90afb6a48262227a726408e6ee11"
def require_key_url(api_method):
  @wraps(api_method)
  def check_key_url(*args, **kwargs):
    #apikey = request.headers.get('ApiKey')
    apikey = request.args.get('key')
    input_url = request.args.get('url')
    if apikey and apikey == SECRET_KEY and input_url:          
      parsed_url = urlparse(input_url)  
      if bool(parsed_url.scheme) and bool(parsed_url.netloc):      	
	return api_method(*args, **kwargs)
      else:
	abort(400)
    else:
      abort(401)
  return check_key_url

# Classify the request for one image
ADULT_THRESHOLD = 0.7
@application.route("/tag", methods=["GET", "POST"])
#@require_key_url
def classify_image():
  start_time = time.time()
  #incoming_ip = request.remote_addr    
  # handle GET
  if request.method == 'GET':
    input_url = request.args.get('url')
    #
    try:
      resp = sess.get(input_url, stream=False)
      if resp.status_code == 200 and 'image' in resp.headers['content-type']:	    
	#resp_data = resp.content
	input_image_pil = Image.open(StringIO.StringIO(resp.content))
	if input_image_pil.format == 'JPEG':
	  resp_data = resp.content
	else:	# convert to JPEG format
	  #if len(input_image_pil.getbands()) < 3:
	    #input_image_pil = input_image_pil.convert("RGB")
	  output_obj = StringIO.StringIO()
	  input_image_pil.save(output_obj, format="JPEG")
	  resp_data = output_obj.getvalue()
	  output_obj.close()
	
	# classify with tfserving
	channel = implementations.insecure_channel(HOST, PORT)
	stub = inception_inference_pb2.beta_create_InceptionService_stub(channel)
	grpc_request = inception_inference_pb2.InceptionRequest() 
	grpc_request.jpeg_encoded = resp_data
	result = stub.Classify(grpc_request , 60.0)  # 60 secs timeout
	sLabel = [result.classes[i] for i in range(len(result.classes))]
	fConf = softmax(result.scores).tolist()
	sAdultContent = "No"
	if sLabel[0] in ['ass', 'penis', 'vagina', 'womanbreast']:
	  if fConf[0] >= ADULT_THRESHOLD:
	    sAdultContent = "Yes"
	  else: 
	    sAdultContent = "NotSure"
	output = {"File":input_url, "Label":sLabel, "Probability":fConf, 
		  "Adult_Content":sAdultContent}
	del resp_data     
      else:      
	output = {"File":input_url, "Label":"Cannot DOWNLOAD target file", 
		  "Probability":"NA", "Adult_Content":"NA"}	    
	#return json.dumps(output)      	  
	
      #response = Response(response=json.dumps(output, sort_keys=True), 
			  #mimetype='application/json')
      #response.headers.add('Access-Control-Allow-Origin', '*')   
      #run_time = time.time() - start_time    
      ##logger.info('{0} {1} {2}'.format(datetime.datetime.now(), output, '%.3f'%run_time))
      #write_log.delay('{0} {1}'.format(output, '%.3f'%run_time))
      #return response
    except Exception, e:	
      raise
  # handle POST
  elif request.method == 'POST':     
    # load posted data
    post_data = request.get_data()
    # open with Pillow
    input_image_pil = Image.open(StringIO.StringIO(post_data))
    if input_image_pil.format == 'JPEG':
      resp_data = post_data
    else:	# convert to JPEG format
      #if len(input_image_pil.getbands()) < 3:
	#input_image_pil = input_image_pil.convert("RGB")
      output_obj = StringIO.StringIO()
      input_image_pil.save(output_obj, format="JPEG")
      resp_data = output_obj.getvalue()
      output_obj.close()
    # classify with tfserving
    channel = implementations.insecure_channel(HOST, PORT)
    stub = inception_inference_pb2.beta_create_InceptionService_stub(channel)
    grpc_request = inception_inference_pb2.InceptionRequest() 
    grpc_request.jpeg_encoded = resp_data
    result = stub.Classify(grpc_request , 60.0)  # 60 secs timeout
    sLabel = [result.classes[i] for i in range(len(result.classes))]
    fConf = softmax(result.scores).tolist()
    sAdultContent = "No"
    if sLabel[0] in ['ass', 'penis', 'vagina', 'womanbreast']:
      if fConf[0] >= ADULT_THRESHOLD:
	sAdultContent = "Yes"
      else: 
	sAdultContent = "NotSure"
    #output = {"File":input_url, "Label":sLabel, "Probability":fConf, 
	      #"Adult_Content":sAdultContent}
    output = {"Label":sLabel, "Probability":fConf, "Adult_Content":sAdultContent}
    
    del resp_data 
    #return json.dumps(output)
  else:
    output = "Invalid HTTP Method"
  # return   
  response = Response(response=json.dumps(output, sort_keys=True), 
		      mimetype='application/json')
  response.headers.add('Access-Control-Allow-Origin', '*')   
  run_time = time.time() - start_time    
  #logger.info('{0} {1} {2}'.format(datetime.datetime.now(), output, '%.3f'%run_time))
  #write_log.delay('{0} {1}'.format(output, '%.3f'%run_time))
  return response        
    
###

###
# Extract text from image
@application.route("/text", methods=["GET", "POST"])
#@require_key_url
def ocr_image():
  start_time = time.time()
  
  if request.method == 'GET':
    input_url = request.args.get('url')
    resp = sess.get(input_url, stream=False)
    # download file at remote url
    if resp.status_code == 200 and 'image' in resp.headers['content-type']:	              
      resp_data = resp.content      
    else:      
      sRecognizedText = "Cannot DOWNLOAD target file"
  elif request.method == 'POST':
    # load posted data
    resp_data = request.get_data()
  else:
    sRecognizedText = "Invalid HTTP Method"
  # tesseract extracts text 
  if resp_data:
    input_image_pil = Image.open(StringIO.StringIO(resp_data))    
    sRecognizedText = pytesseract.image_to_string(input_image_pil, lang='some+eng')  
  
  # return   
  output = {"Recognized_Text":sRecognizedText}    
  response = Response(response=json.dumps(output, sort_keys=True), 
		      mimetype='application/json')
  response.headers.add('Access-Control-Allow-Origin', '*')   
  run_time = time.time() - start_time      
  #write_log.delay('{0} {1}'.format(output, '%.3f'%run_time))
  return response
  
  
###

###
###
## Caption input image
#@application.route("/caption", methods=["GET", "POST"])
##@require_key_url
#def caption_image():
  #start_time = time.time()
  ## lists that hold result
  #lstScores = []
  #lstCaption = []    
  ##
  #if request.method == 'GET':
    #input_url = request.args.get('url')
    #resp = sess.get(input_url, stream=False)
    ## download file at remote url
    #if resp.status_code == 200 and 'image' in resp.headers['content-type']:	              
      #resp_data = resp.content      
    #else:      
      #lstCaption = ["Cannot DOWNLOAD target file"]
  #elif request.method == 'POST':
    ## load posted data
    #resp_data = request.get_data()
  #else:
    #lstCaption = ["Invalid HTTP Method"]    
  ## tesseract extracts text 
  #if resp_data and len(lstCaption)==0:
    #input_image_pil = Image.open(StringIO.StringIO(resp_data))
    ## convert to RGB 3 channels if it's necessary
    #if len(input_image_pil.getbands()) != 3:
      #input_image_pil = input_image_pil.convert("RGB")
    ## convert to numpy array data
    #img_npdata = np.array(input_image_pil)
    ## extract visual feature for the image
    #visual_features = ife.extract_features(img_npdata, 
					   #flag_from_file=False)
    ## generate sentences
    #sentences = decoder.decode(visual_features, BEAM_SIZE)
    ## prepare result
    #for s in sentences:      
      #lstScores.append(s['score'])
      #lstCaption.append(' '.join(s['words']))
    #lstConf = softmax(lstScores).tolist()     
  
  ## return   
  #output = {"Caption":lstCaption, "Probability":lstConf}    
  #response = Response(response=json.dumps(output, sort_keys=True), 
		      #mimetype='application/json')
  #response.headers.add('Access-Control-Allow-Origin', '*')   
  #run_time = time.time() - start_time      
  #write_log.delay('{0} {1}'.format(output, '%.3f'%run_time))
  #return response
  
  
###

###

if __name__ == "__main__":    
  #application.run(host='0.0.0.0', port=2222)
  application.run(debug=False, use_reloader=False)
  