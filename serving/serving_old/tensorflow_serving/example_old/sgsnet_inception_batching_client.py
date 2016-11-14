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

#!/usr/bin/env python

"""
Send JPEG image to inception_batching_inference server for classification. This
client can handle images with very different aspect ratios from the
original training material by using alternative preprocessing methods to
simple resizing, such as padding (symmetrically adding 0s to the edges
of the image to bring it to a square shape) or centrally cropping. Both
of these methods prevent heavy distortion from being introduced into the
image, but also carry other disadvantages (see options).
"""

import os
import sys
import threading

# HieuNM
import json
import requests
import time
import web
import ssl
import pytesseract
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.poolmanager import PoolManager
from StringIO import StringIO
from urlparse import urlparse
# PIL must be imported before tensorflow. It's totally unclear why
# and probably has some kind of byzantine and bizarre reason (likely
# because the Pillow version vs. the one tensorflow uses are different).
from PIL import Image

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf
from tensorflow.python.platform.logging import warn

from tensorflow_serving.example import inception_batching_inference_pb2
from tensorflow_serving.example import inception_inference_pb2



FLAGS = tf.app.flags.FLAGS  
FLAGS.prep_method = 'resize'
FLAGS.image_size = 299
FLAGS.concurrency = 10

HOST = "localhost"
PORT = 9000
NUM_CLASSES = 5
WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
SYNSET_FILE = os.path.join(WORKING_DIR, 'sgsnet_synsets.txt')
METADATA_FILE = os.path.join(WORKING_DIR, 'sgsnet_metadata.txt')


def _prep_image(img, w=FLAGS.image_size, h=FLAGS.image_size):
  '''
  Preprocesses the requested image to the desired size, permitting server-side
  batching of Inception. The original preprocessing operation for Inception is
  to central crop by 87.5%, and then simply resize to the appropriate
  dimensions. Here, there are additional options for users who have retrained
  Inception using alternative preprocessing methods. The source image cropping
  of 87.5% is presumed completed; this is intended to be a more general prep
  function.

  Args:
    img: A PIL image.
    w: The desired width.
    h: The desired height.
  '''
  if FLAGS.prep_method == 'resize':
    return _resize_to(img, w=w, h=h)
  elif FLAGS.prep_method == 'crop':
    # resize to appropriate dimensions
    resized_im = _resize_to_min(img, w=w, h=h)
    # center crop
    return _center_crop_to(resized_im, w=w, h=h)
  elif FLAGS.prep_method == 'padresize':
    des_asp = float(w) / h
    # pad the image
    padded_im = _pad_to_asp(img, des_asp)
    # resize the image
    return _resize_to(padded_im, w=w, h=h)


def _resize_to(img, w=None, h=None):
  '''
  Resizes the image to a desired width and height. If either is undefined,
  it resizes such that the defined argument is satisfied and preserves aspect
  ratio. If both are defined, resizes to satisfy both arguments without
  preserving aspect ratio.

  Args:
    img: A PIL image.
    w: The desired width.
    h: The desired height.
  '''
  ow, oh = img.size
  asp = float(ow) / oh
  if w is None and h is None:
    # do nothing
    return img
  elif w is None:
    # set the width
    w = int(h * asp)
  elif h is None:
    h = int(w / asp)
  return img.resize((w, h), Image.BILINEAR)


def _read_image(imagefn):
  '''
  This function reads in an image as a raw file and then converts
  it to a PIL image. Note that, critically, PIL must be imported before
  tensorflow for black magic reasons.

  Args:
    imagefn: A fully-qualified path to an image as a string.

  Returns:
    The PIL image requested.
  '''
  try:
    pil_image = Image.open(imagefn)
  except Exception, e:
    warn('Problem opening %s with PIL, error: %s' % (imagefn, e.message))
    return None
  try:
    # ensure that the image file is closed.
    pil_image.load()
  except Exception, e:
    warn('Problem loading %s with PIL, error: %s' % (imagefn, e.message))
    return None
  return pil_image


def _resize_to_min(img, w=None, h=None):
  '''
  Resizes an image so that its size in both dimensions is greater than or
  equal to the provided arguments. If either argument is None, that dimension
  is ignored. If the image is larger in both dimensions, then the image is
  shrunk. In either case, the aspect ratio is preserved and image size is
  minimized. If the target of interest is in the center of the frame, but the
  image has an unusual aspect ratio, center cropping is likely the best option.
  If the image has an unusual aspect ratio but is irregularly framed, padding
  the image will prevent distortion while also including the entirety of the
  original image.

  Args:
    img: A PIL image.
    w: The minimum width desired.
    h: The minimum height desired.
  '''
  ow, oh = img.size
  if w is None and h is None:
    return img
  if w is None:
    # resize to the desired height
    return _resize_to(img, h=h)
  elif h is None:
    # resize to the desired width
    return _resize_to(img, w=w)
  if ow == w and oh == h:
    # then you need not do anything
    return img
  hf = h / float(oh)  # height scale factor
  wf = w / float(ow)  # width scale factor
  if min(hf, wf) < 1.0:
    # then some scaling up is necessary. Scale up by as much as needed,
    # leaving one dimension larger than the requested amount if required.
    scale_factor = max(hf, wf)
  else:
    # scale down by the least amount to ensure both dimensions are larger
    scale_factor = min(hf, wf)
  nw = int(ow * scale_factor)
  nh = int(oh * scale_factor)
  return _resize_to(img, w=nw, h=nh)


def _center_crop_to(img, w, h):
  '''
  Center crops image to desired size. If either dimension of the image is
  already smaller than the desired dimensions, the image is not cropped.

  Args:
    img: A PIL image.
    w: The width desired.
    h: The height desired.
  '''
  ow, oh = img.size
  if ow < w or oh < h:
    return img
  upper = (oh - h) / 2
  lower = upper + h
  left = (ow - w) / 2
  right = left + w
  return img.crop((left, upper, right, lower))


def _pad_to_asp(img, asp):
  '''
  Symmetrically pads an image to have the desired aspect ratio.

  Args:
    img: A PIL image.
    asp: The aspect ratio, a float, as w / h
  '''
  ow, oh = img.size
  oasp = float(ow) / oh
  if asp > oasp:
    # the image is too narrow. Pad out width.
    nw = int(oh * asp)
    left = (nw - ow) / 2
    upper = 0
    newsize = (nw, oh)
  elif asp < oasp:
    # the image is too short. Pad out height.
    nh = int(ow / asp)
    left = 0
    upper = (nh - oh) / 2
    newsize = (ow, nh)
  nimg = Image.new(img.mode, newsize)
  nimg.paste(img, box=(left, upper))
  return nimg


def prep_inception_from_file(image_file):
  '''
  Preprocesses an image from a fully-qualified file, in same
  manager as the batchless inception server (including the
  87.5% source crop) and wraps _prep_image to make the image
  the correct size.
  '''
  # Load the image.
  image = _read_image(image_file)
  if image is None:
    return None

  # In the original implementation of Inception export, the images are
  # centrally cropped by 87.5 percent before undergoing adjustments to
  # bring them into the correct size, which we replicate here.
  ow, oh = image.size  # obtain the original width and height
  nw = int(ow * 0.875)  # compute the new width
  nh = int(oh * 0.875)  # compute the new height
  image = _center_crop_to(image, w=nw, h=nh)  # center crop to 87.5%

  # preprocess the image to bring it to a square with edge length
  # FLAGS.image_size
  image = _prep_image(image)

  # Convert to a numpy array
  image = numpy.array(image)

  # Perform additional preprocessing to mimic the inputs to inception.
  # Scale image pixels. all pixels now reside in [0, 1), as in the
  # tensor representation following tf.image.decode_jpeg.
  image = image / 256.

  # Scale the image to the domain [-1, 1) (referred to incorrectly
  # as (-1, 1) in the original documentation).
  image = (image - 0.5) * 2.0
  return image


def _load_image(se, img_url):
  '''
  This function requests data from the remote url and then converts
  it to a PIL image. Note that, critically, PIL must be imported before
  tensorflow for black magic reasons.

  Args:
    se: a custom Session instance to make http request
    img_url: A URL to an image as a string.

  Returns:
    The PIL image targeted.
  '''
  parsed_url = urlparse(img_url)
  pil_image = None
  if se and bool(parsed_url.scheme) and bool(parsed_url.netloc):  
    try:
      resp = se.get(img_url, stream=False)
      if resp.status_code != 200 or 'image' not in resp.headers['content-type']:	    
	raise ValueError('Target is not an image or network problem')	
    except Exception, e:	  
      warn('Cannot DOWNLOAD target file at %s, error: %s' % (img_url, e.message))	
      return None	 
    
    try:
      pil_image = Image.open(StringIO(resp.content))
    except Exception, e:
      warn('Problem opening %s with PIL, error: %s' % (img_url, e.message))
      return None
    try:
      # ensure that the image file is closed.
      pil_image.load()
    except Exception, e:
      warn('Problem loading %s with PIL, error: %s' % (img_url, e.message))
      return None
    
  return pil_image

def prep_inception_from_url(se, image_url):
  '''
  Preprocesses an image from a fully-qualified file, in same
  manager as the batchless inception server (including the
  87.5% source crop) and wraps _prep_image to make the image
  the correct size.
  '''
  # Load the image.
  image = _load_image(se, image_url)
  if image is None:
    return None

  # Extract text using sgs trained Tesseract
  #sRecognizedText = None
  sRecognizedText = pytesseract.image_to_string(image, lang='some+eng')
  ## In the original implementation of Inception export, the images are
  ## centrally cropped by 87.5 percent before undergoing adjustments to
  ## bring them into the correct size, which we replicate here.
  #ow, oh = image.size  # obtain the original width and height
  #nw = int(ow * 0.875)  # compute the new width
  #nh = int(oh * 0.875)  # compute the new height
  #image = _center_crop_to(image, w=nw, h=nh)  # center crop to 87.5%

  # preprocess the image to bring it to a square with edge length
  # FLAGS.image_size
  image = _prep_image(image)

  # Convert to a numpy array
  image = numpy.array(image)

  # Perform additional preprocessing to mimic the inputs to inception.
  # Scale image pixels. all pixels now reside in [0, 1), as in the
  # tensor representation following tf.image.decode_jpeg.
  image = image / 256.

  # Scale the image to the domain [-1, 1) (referred to incorrectly
  # as (-1, 1) in the original documentation).
  image = (image - 0.5) * 2.0
  return image, sRecognizedText


def do_inference(astub, concurrency, jsListUrl, se):
  '''
  Performs inference over multiple images given a list of image urls
  in JSON format, with one image url per item. Prints the results of
  the top N classes.

  Args:    
    concurrency: Maximum number of concurrent requests.
    jsListUrl: posted JSON containing the fully-qualified path to a 
		single image per item.
    se: custom Requests Session

  Returns:
    None.
  '''
  
  num_images = len(jsListUrl)
  cv = threading.Condition()
  # this will store the ouput Inception. We require it to map filenames
  # to their labels in the case of batching.
  inference_results = []
  result_status = {'active': 0, 'error': 0, 'done': 0}
  def done(result_future, sRecognizedText, filename):
    '''
    Callback for result_future, modifies inference_results to hold the
    output of Inception.
    '''
    with cv:
      exception = result_future.exception()
      if exception:
        result_status['error'] += 1
        print exception
      else:
        result = result_future.result()
        indices = [result.classes[i] for i in range(NUM_CLASSES)]
	softmax_probs = softmax(result.scores).tolist()	
        #scores = [result.scores[i] for i in range(NUM_CLASSES)]
        #inf_res = [filename, indices, scores]
        inf_res = [filename, sRecognizedText, indices, softmax_probs]
        inference_results.append(inf_res)
      result_status['done'] += 1
      result_status['active'] -= 1
      cv.notify()

  for image_url in jsListUrl:
    image_array, sRecognizedText = prep_inception_from_url(se, image_url)
    if image_array is None:
      num_images -= 1
      continue
    request = inception_batching_inference_pb2.InceptionBatchingRequest()
    # this is not as efficient as i feel like it could be,
    # since you have to flatten the array then turn it into
    # a list before you extend the request image_data field.
    request.image_data.extend(image_array.flatten().tolist())
    
    with cv:
      while result_status['active'] == concurrency:
        cv.wait()
      result_status['active'] += 1
    result_future = astub.Classify.future(request, 10.0)  # 10 second timeout
    result_future.add_done_callback(
        lambda result_future, sRecognizedText=sRecognizedText, filename=image_url: done(result_future, sRecognizedText, filename))  # pylint: disable=cell-var-from-loop
  with cv:
    while result_status['done'] != num_images:
      cv.wait()
  return inference_results


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0)


def classify_postdata(astub, post_data, se, num_classes_returned=5):      
  # Load and preprocess the image.  
  pil_image = Image.open(StringIO(post_data))
  # Extract text using sgs trained Tesseract
  #sRecognizedText = None
  sRecognizedText = pytesseract.image_to_string(pil_image, lang='some+eng')  
  pil_image_resized = _prep_image(pil_image)
  image = numpy.array(pil_image_resized)    
  # Perform additional preprocessing to mimic the inputs to inception.
  # Scale image pixels. all pixels now reside in [0, 1), as in the
  # tensor representation following tf.image.decode_jpeg.
  image = image / 256.
  # Scale the image to the domain [-1, 1) (referred to incorrectly
  # as (-1, 1) in the original documentation).
  image = (image - 0.5) * 2.0
  if image is None:
    return
  # The image is now a numpy nd array with the appropraite size for
  # Inception, with each element constrained to the domain [-1, 1).

  # Create the request. See inception_inference.proto for gRPC request/
  # response details. Instead of using an encoded jpeg, we send the
  # data as a row-major flattened list of floats.  
  request = inception_batching_inference_pb2.InceptionBatchingRequest()
  request.image_data.extend(image.flatten().tolist())
  #request.image_data.extend(float(image_data))
  result = astub.Classify(request, 10.0)  # 10 secs timeout
  
  sLabel = []
  #fConf = []
  for i in range(NUM_CLASSES):
    index = result.classes[i]
    #score = result.scores[i]
    #print '%f : %s' % (score, texts[synsets[index - 1]])
    sLabel.append(texts[synsets[index - 1]])
  
  softmax_probs = softmax(result.scores)
  #sLabel = list(result.classes)
  fConf = softmax_probs.tolist()
  return sLabel, fConf, sRecognizedText


def classify_url(astub, img_url, se, num_classes_returned=5):    
  #request.jpeg_encoded = image_data
  #result = stub.Classify(request, 60.0)  # 60 secs timeout
  #softmax_probs = softmax(result.scores)
  #sLabel = list(result.classes)
  #fConf = softmax_probs.tolist()
  #return sLabel, fConf  
  
  # Load and preprocess the image.  
  image, sRecognizedText = prep_inception_from_url(se, img_url)  
  if image is None:
    return
  # The image is now a numpy nd array with the appropraite size for
  # Inception, with each element constrained to the domain [-1, 1).

  # Create the request. See inception_inference.proto for gRPC request/
  # response details. Instead of using an encoded jpeg, we send the
  # data as a row-major flattened list of floats.  
  request = inception_batching_inference_pb2.InceptionBatchingRequest()
  request.image_data.extend(image.flatten().tolist())
  #request.image_data.extend(float(image_data))
  result = astub.Classify(request, 10.0)  # 10 secs timeout
  
  sLabel = []
  #fConf = []
  for i in range(NUM_CLASSES):
    index = result.classes[i]
    #score = result.scores[i]
    #print '%f : %s' % (score, texts[synsets[index - 1]])
    sLabel.append(texts[synsets[index - 1]])
  
  softmax_probs = softmax(result.scores)
  #sLabel = list(result.classes)
  fConf = softmax_probs.tolist()
  return sLabel, fConf, sRecognizedText    


class Tfserving_Classify:
  def GET(self):
    web.header('Access-Control-Allow-Origin', '*')    
    web.header('Content-Type', 'application/json')  
    
    incomingIP = web.ctx['ip']            
    user_data = web.input()
    if 'url' in user_data: # to check if the url-object is created
      start_time = time.time()     
      img_url = user_data.url		
      #global nReceived
      #nReceived += 1
      #logger.info('{0} {1}'.format(start_time, nReceived))
      # get remote image 
      if img_url != '':	
	try:
	  # call Tensorflow Serving grpc server
	  #global CURRENT_STUB_IDX
	  #CURRENT_STUB_IDX += 1
	  #if CURRENT_STUB_IDX >= len(STUB):
	    #CURRENT_STUB_IDX = 0	  
	  #astub = STUB[CURRENT_STUB_IDX]
	  achannel = implementations.insecure_channel(HOST, PORT)
	  astub = inception_batching_inference_pb2.beta_create_InceptionService_stub(achannel)
	  
	  sLabel, fConf, sRecognizedText = classify_url(astub, img_url, sess, 5)		  	  
	  sAdultContent = "No"
	  if sLabel[0] in ['ass', 'penis', 'vagina', 'womanbreast']:
	    if fConf[0] >= 0.7:
	      sAdultContent = "Yes"
	    else: 
	      sAdultContent = "NotSure"
	  output = {"File":img_url, "Label":sLabel, "Probability":fConf, "Recognized_Text":sRecognizedText, "Adult Content":sAdultContent}
	  
	  proc_time = time.time() - start_time
	  #print img_url, proc_time
	  #del resp_data
	  return json.dumps(output, sort_keys=True)	  
	except Exception, e:	  
	  print >> sys.stderr, e
	  sys.stdout.flush()
	  #output = {'File':img_url,'Label': e,'Label_Confidence': 'NA','Recognized_Text':'NA'}
	  #yield json.dumps(output)
	  return ' '.join(['Error classifying image', str(e)])
    
  def POST(self):
    web.header('Access-Control-Allow-Origin', '*')
    #web.header('Content-type', "text/plain; charset=utf-8")
    #web.header('Content-Type', 'application/json')    
    # parse inputs     
    post_data = web.data()   
    try:
      start_time = time.time()
      # load as JSON
      #print post_data
      jsListUrl = json.loads(post_data)      
      
      #global CURRENT_STUB_IDX
      #CURRENT_STUB_IDX += 1
      #if CURRENT_STUB_IDX >= len(STUB):
	#CURRENT_STUB_IDX = 0
      ##print 'CURRENT_STUB_IDX: ', CURRENT_STUB_IDX
      #astub = STUB[CURRENT_STUB_IDX]
      achannel = implementations.insecure_channel(HOST, PORT)
      astub = inception_batching_inference_pb2.beta_create_InceptionService_stub(achannel)
      
      inference_results = do_inference(astub, FLAGS.concurrency, jsListUrl, sess)
      output = []
      for filename, recognized_texts, indices, probs in inference_results:	
	alabel = [texts[synsets[idx - 1]] for idx in indices]
	sAdultContent = "No"
	if alabel[0] in ['ass', 'penis', 'vagina', 'womanbreast']:
	  if probs[0] >= 0.7:
	    sAdultContent = "Yes"
	  else: 
	    sAdultContent = "NotSure"
	  
	ares = {"File":filename, "Label":alabel, "Probability":probs, "Recognized_Text":recognized_texts, "Adult Content":sAdultContent}
	output.append(ares)
	
	#print '%s Inference:' % filename
	#for idx, prob in zip(indices, probs):
	  #print '\t%f : %s' % (prob, texts[synsets[idx - 1]])		  
      
      run_time = time.time() - start_time
      print 'Total seconds: ', run_time
      
      #return output
      return json.dumps(output, sort_keys=True)
	
    except Exception, e:
      print >> sys.stderr, e
      sys.stdout.flush()
      return ' '.join(['Cannot RECEIVE target file or it is not an image!', str(e)])


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

      #global CURRENT_STUB_IDX
      #CURRENT_STUB_IDX += 1
      #if CURRENT_STUB_IDX >= len(STUB):
	#CURRENT_STUB_IDX = 0
      #astub = STUB[CURRENT_STUB_IDX]
      achannel = implementations.insecure_channel(HOST, PORT)
      astub = inception_batching_inference_pb2.beta_create_InceptionService_stub(achannel)      
      
      sLabel, fConf, sRecognizedText = classify_postdata(astub, post_data, sess, 5)
      jsLabel = json.loads(json.dumps(sLabel))
      jsConf = json.loads(json.dumps(fConf))
      sAdultContent = "No"
      if jsLabel[0] in ['ass', 'penis', 'vagina', 'womanbreast']:
	if jsConf[0] >= 0.7:
	  sAdultContent = "Yes"
	else: 
	  sAdultContent = "NotSure"
      output = {"Label":jsLabel, "Probability":jsConf, "Recognized_Text":sRecognizedText, "Adult Content":sAdultContent}
      
      #return output
      return json.dumps(output, sort_keys=True)
      
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
  
  # Use a `Session` instance to customize how `requests` handles making HTTP requests.
  sess = requests.Session()
  # `mount` a custom adapter that retries failed connections for HTTP and HTTPS requests.
  sess.mount("http://", requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=500, max_retries=3))
  #sess.mount("http://", requests.adapters.HTTPAdapter(max_retries=3))
  sess.mount("https://", SSLAdapter(pool_connections=100, pool_maxsize=500, ssl_version=ssl.PROTOCOL_TLSv1, max_retries=3))  
  
  # Set up protocol    
  #HOST_PORT = ["localhost:9000", "localhost:9001"]  
  #HOST_PORT = ["localhost:9000"]
  #CHANNEL = []
  #STUB = []
  #for hp in HOST_PORT:
    #ahost, aport = hp.split(':')
    #achannel = implementations.insecure_channel(ahost, int(aport))
    #astub = inception_batching_inference_pb2.beta_create_InceptionService_stub(achannel)
    #CHANNEL.append(achannel)
    #STUB.append(astub)
  
  #CURRENT_STUB_IDX = -1
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

    # setup log file
  if 'logger' not in locals():
    #logging.getLogger("urllib3").setLevel(logging.WARNING)
    #
    logger = logging.getLogger()
    log = logging.FileHandler('/home/hieunguyen/sgsnet_2222.log')
    formatter = logging.Formatter()
    log.setFormatter(formatter)
    logger.addHandler(log)
    logger.setLevel(logging.INFO)  
  
  #nReceived = 0

  # Web application to process request   
  web.config.debug = False
  app = web.application(urls, globals())  
  app.run()