from flask import Flask, request, redirect, send_file, url_for, render_template_string, render_template
from io import BytesIO  
import hashlib
cache = {}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
@app.route("/hash", methods=["GET", "POST"])
def hash():
    if request.method == 'POST':
        photo1 = request.files['photo1']
        photo2 = request.files['photo2']
        hash_type = request.form['hash_type']
        
        if photo1.filename == '' or photo2.filename == '':
            return "No selected file", 400
        cache["photo1"] = BytesIO(photo1.read())
        cache["photo2"] = BytesIO(photo2.read())
        print(cache["photo1"])
        print(cache["photo2"])
        if hash_type == 'md5':
            hash1 = hashlib.md5(cache["photo1"].getvalue()).hexdigest()
            hash2 = hashlib.md5(cache["photo2"].getvalue()).hexdigest()
        elif hash_type == 'sha1':
            hash1 = hashlib.sha1(cache["photo1"].getvalue()).hexdigest()
            hash2 = hashlib.sha1(cache["photo2"].getvalue()).hexdigest()
        elif hash_type == 'sha256':
            hash1 = hashlib.sha256(cache["photo1"].getvalue()).hexdigest()
            hash2 = hashlib.sha256(cache["photo2"].getvalue()).hexdigest()
        else:
            return "Invalid hash type", 400
        return render_template('hash-results.html', hash1=hash1, hash2=hash2, filename1='/preview/photo1', filename2='/preview/photo2')
    return redirect(url_for("home"))


@app.route("/preview/<filename>")
def preview(filename):
    if filename in cache:
        cache[filename].seek(0)  
        return send_file(cache[filename], mimetype="image/jpeg")  

@app.route("/")
def home():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)