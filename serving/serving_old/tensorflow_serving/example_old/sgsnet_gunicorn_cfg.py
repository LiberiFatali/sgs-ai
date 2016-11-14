#
# Server socket
#
#   bind - The socket to bind.
#
#       A string of the form: 'HOST', 'HOST:PORT', 'unix:PATH'.
#       An IP is a valid HOST.
#
#   backlog - The number of pending connections. This refers
#       to the number of clients that can be waiting to be
#       served. Exceeding this number results in the client
#       getting an error when attempting to connect. It should
#       only affect servers under significant load.
#
#       Must be a positive integer. Generally set in the 64-2048
#       range.
#
bind = '0.0.0.0:2222'
#bind = 'unix:sgsnet_tfserving_single_client.sock'
backlog = 2048
# The number of worker processes that this server
# should keep alive for handling requests.
# A positive integer generally in the 2-4 x $(NUM_CORES) range.
# You will want to vary this a bit to find the best for your
# particular application work load.
#workers = 32
workers = 20
# The type of workers to use. The default sync class
# should handle most 'normal' types of work loads
worker_class = 'sync'
"""
The number of worker threads for handling requests.
Run each worker with the specified number of threads.
A positive integer generally in the 2-4 x $(NUM_CORES) range.
You will want to vary this a bit to find the best for your particular application work load.
"""
threads = 1
# Keep track of process id
#pidfile='sgsnet_gunicorn_pid'
# Make it daemonized
#daemon = True
daemon = False
# Add access loggging
accesslog='logs/gunicorn/sgsnet_gunicorn_access.log'
# Add error logging
errorlog='logs/gunicorn/sgsnet_gunicorn_error.log'

#
#   spew - Install a trace function that spews every line of Python
#       that is executed when running the server. This is the
#       nuclear option.
#
#       True or False
#
spew = False
"""Load application code before the worker processes are forked.

By preloading an application you can save some RAM resources as well as speed up 
server boot times. Although, if you defer application loading to each worker process, 
you can reload your application code easily by restarting workers.
"""
preload = True
