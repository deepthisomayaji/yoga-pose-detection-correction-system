from flask import Flask,flash, render_template,Response,request,redirect,url_for
import ProjectClasses as project
import pandas as pd
import csv
app = Flask(__name__)
project = project.frameworking()

#welcome page
@app.route('/')
def index():
    return render_template('index.html')

#options page
@app.route('/options')
def options():
    return render_template('options.html')

#developers page
@app.route('/developers')
def developers():
    return render_template('developers.html')

#contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

#poses
@app.route('/trikonasana')
def trikonasana():
    return render_template('trikonasana.html')
@app.route('/ardha_uttasana')
def ardha_uttasana():
    return render_template('ardha_uttasana.html')
@app.route('/lunges')
def lunges():
    return render_template('lunges.html')
@app.route('/Vyaghrasana')
def Vyaghrasana():
    return render_template('Vyaghrasana.html')
@app.route('/ardha_chandrasana')
def ardha_chandrasana():
    return render_template('ardha_chandrasana.html')
@app.route('/planks')
def planks():
    return render_template('planks.html')

df = pd.read_csv('report.csv')
df.to_csv('report.csv',index=None)
#yoga1begin page
@app.route('/common_video')
def common_video():
    data = pd.read_csv('report.csv')
    return render_template("common_video.html",tables=[data.to_html()], titles=[''])

@app.route('/new', methods=['POST'])
def new():
    new.yoganame = request.form['nameyoga']
    data = pd.read_csv('report.csv')
    return render_template("common_video.html",tables=[data.to_html()], titles=[''])

def write_to_file (dd):
	with open('database.txt',mode = 'a') as database:
		email = dd["email"]
		subject = dd["subject"]
		message = dd["message"]
		file = database.write(f'\n {email},{subject},{message}')
@app.route('/submit_form', methods=['POST'])
def submit_form():
	if request.method == 'POST':
		dd = request.form.to_dict()
		write_to_file (dd)
		return 'form_submitted'
	else:
		return 'Some thing went wrong ... Try again'


@app.route('/video_feed')
def video_feed():
    yoganame = new.yoganame
    return Response(project.gen_frames(yoganame), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)