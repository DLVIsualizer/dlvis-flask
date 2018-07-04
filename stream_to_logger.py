# Source come from
# https://www.electricmonk.nl/log/2011/08/14/redirect-stdout-and-stderr-to-a-logger-in-python/


import logging
import sys
import time
import ntpath


# class StreamToLogger(object):
#    """
#    Fake file-like stream object that redirects writes to a logger instance.
#    """
#    def __init__(self, logger, log_level=logging.INFO):
#       self.logger = logger
#       self.log_level = log_level
#       self.linebuf = ''
#
#    def write(self, buf):
#       for line in buf.rstrip().splitlines():
#          self.logger.log(self.log_level, line.rstrip())
#
# logging.basicConfig(
#    level=logging.DEBUG,
#    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
#    filename="stderr.log",
#    filemode='a'
# )
#
# stderr_logger = logging.getLogger('STDERR')
# sl = StreamToLogger(stderr_logger, logging.ERROR)
# sys.stderr = sl

# 기타 로깅 가이드
# http://hamait.tistory.com/880
class FunctionProfileLogger(object):
	def __init__(self, logger, log_level=logging.INFO):
		self.logger = logger
		self.log_level = log_level
		self.linebuf = ''
		self.functionMaps = {}
	
	# logger write redirection
	def write(self, buf):
		for line in buf.rstrip().splitlines():
			self.logger.log(self.log_level, line.rstrip())
	
	def startFunction(self, frame):
		if self.isDebugging():
			self.functionMaps[frame.f_code.co_name] = start_time = time.perf_counter()
		else:
			pass
	
	def endFuction(self, frame, additional=""):
		if self.isDebugging():
			elapsed = (time.perf_counter() - self.functionMaps[frame.f_code.co_name])
			self.logger.debug("%25s %25s %10.5fs (%s)" %
			                  (ntpath.basename(frame.f_code.co_filename), frame.f_code.co_name,elapsed,additional))
		else:
			pass
	
	def isDebugging(self):
		return self.logger.isEnabledFor(logging.DEBUG)


# "functionTime" 로거 생성
funcionLog = logging.getLogger("functionTime")
funcionLog.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s/%(levelname)s::%(message)s')

file_handler = logging.FileHandler('profileFunction.log')
file_handler.setFormatter(formatter)
funcionLog.addHandler(file_handler)

fl = FunctionProfileLogger(funcionLog, logging.DEBUG)
