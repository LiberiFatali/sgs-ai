#from test_proj import application
from sgsnet_tfserving_single_client import application

if __name__ == "__main__":
  #application.run()
  application.run(debug=False, use_reloader=False)