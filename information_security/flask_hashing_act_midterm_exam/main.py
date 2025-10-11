from flask import Flask, request, redirect, send_file, url_for, render_template_string, render_template
from io import BytesIO  
import hashlib
import dotenv
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
        
        hash_comparison = compare_hashes(hash1, hash2)
        if hash_comparison:
            return render_template('hash-results.html', hash1=hash1, hash2=hash2, filename1='/preview/photo1', filename2='/preview/photo2', message="The files are identical.")
        return render_template('hash-results.html', hash1=hash1, hash2=hash2, filename1='/preview/photo1', filename2='/preview/photo2', message="Files have been tampered/modified.")
    return redirect(url_for("home"))


def compare_hashes(hash1, hash2):
    return hash1 == hash2



def encrypt_text(message, key):
    encrypted = ""
    key_length = len(key)

    for i, char in enumerate(message):
        # Get ASCII value of message char
        message_ascii = ord(char)
        # Get ASCII value of key char (repeating)
        key_ascii = ord(key[i % key_length])
        # Shift message char by key's ASCII value (mod 256 for byte range)
        encrypted_char = chr((message_ascii + key_ascii) % 256)
        encrypted += encrypted_char

    return encrypted

@app.route("/encrypt", methods=["GET", "POST"])
def encrypt():
    if request.method == "POST":
        message = request.form["message"]
        key = request.form["key"]
        encrypted_message = encrypt_text(message, key)
        return render_template("act1.html", encrypted_message=encrypted_message, mode="encrypt")
    return render_template("act1.html", mode="encrypt")



def decrypted_text(message, key):
    decrypted = ""
    key_length = len(key)
    
    for i, char in enumerate(message):
        # Get ASCII value of message char
        message_ascii = ord(char)
        # Get ASCII value of key char (repeating)
        key_ascii = ord(key[i % key_length])
        # Shift message char by key's ASCII value (mod 256 for byte range)
        decrypted_char = chr((message_ascii - key_ascii) % 256)
        decrypted += decrypted_char
    
    return decrypted


@app.route("/decrypt", methods=["GET", "POST"])
def decrypt():
    if request.method == "POST":
        message = request.form["message"]
        key = request.form["key"]
        decrypted_message = decrypted_text(message, key)
        return render_template("act1.html", decrypted_message=decrypted_message, mode="decrypt")
    return render_template("act1.html", mode="decrypt")

@app.route("/preview/<filename>")
def preview(filename):
    if filename in cache:
        cache[filename].seek(0)  
        return send_file(cache[filename], mimetype="image/jpeg")  

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/verify", methods=["GET", "POST"])
def verify():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        if username == "" or password == "":
            return render_template("login.html", error="Please enter both username and password.")
        elif username == "Infosec" and password == "1nf053c!":
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid credentials. Please try again.")
    return render_template("login.html")

@app.route("/")
def login():
    return render_template("login.html")

if __name__ == "__main__":
    # app.run(host="172.16.0.29", port=8080, debug=False)
    app.run(host="0.0.0.0", port=8080, debug=False)