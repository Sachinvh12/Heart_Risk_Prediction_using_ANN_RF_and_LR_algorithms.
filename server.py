from flask import Flask,render_template,url_for
from flask_bootstrap import Bootstrap
import eda as ed
import mla
app=Flask(__name__)
Bootstrap(app)
@app.route('/')

def index():
    return render_template('index.html')



@app.route('/eda')
def eda():
	return ed.funcy()

@app.route('/ml')
def ml():
	return mla.func()

