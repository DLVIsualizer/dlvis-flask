activate_this = '/var/www/flask/venv/bin/activate_this.py'
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))

import sys

sys.path.insert(0, "/var/www/flask/")

from run_keras_server import app as application
