from flask import Flask , render_template ,jsonify
from gemini import gemini_bp
from openai_routes import openai_bp

app = Flask(__name__)
app.register_blueprint(openai_bp)
app.register_blueprint(gemini_bp)



@app.route("/" , methods = ["GET","POST"])
def main():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)    
