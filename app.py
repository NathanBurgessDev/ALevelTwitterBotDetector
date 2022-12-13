from flask import Flask, render_template, url_for, request, render_template
from BotDetectorSingleData import createGML, createConvertedGraph, predict_User



app = Flask(__name__)

posts = [
    {
        'author': 'Nathan Burgess',
        'title': 'Bot Recognition and Igraph Analysis of Networks',
        'content': '''This website hosts an A level Computer Science project on machine learning algorithms being used to make a prediction to whether a twitter user is a human, or a bot.
        More information for this will be shown on the about page and if you wish to test it for yourself you may go to the Result page.''',
        
        'date_posted': 'December 21, 2020'
    }
]

# All of the HTML files inherit from layout.html
# This allows each HTML file to be much smaller and simpler to read

@app.route("/")
@app.route("/home")
def index():
    return render_template('home.html', posts=posts)


@app.route("/hi/")
def who():
    return "<h1>who are you?</h1>"

@app.route("/hi/<username>")
def greet(username):
    return f"Hi there, {username}!"

@app.route("/about") #Objective 4.2
def about():
    return render_template('about.html', title="About") #objective 4.2 accomplished in this HTML

@app.route("/detect")
def result2():
    return render_template('my-form.html')

#render_template for html file
#input to that html file is the returned processed_text

@app.route('/detect', methods=['POST'])
def result():
    text = request.form['text']
    processed_text = predict_User(text) #Objective 4
    if processed_text == "Failed":
        return render_template("error.html")
    else:
        if processed_text == "0":
            new_result = "that " + text + " is a human"
        else:
            new_result = "that " + text + " is a bot"

        return render_template('result.html', result = new_result, user = text, title = "About")


if __name__ == '__main__':
    app.run(debug=True)
    
