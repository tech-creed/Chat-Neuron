from flask import make_response,Flask, flash, redirect, render_template, request, url_for, session
from app import *
import secrets
import bcrypt

@app.route('/')
def home():
    return render_template('index.html')

#SignUp
@app.route('/register', methods=["POST", "GET"])
def signup():
    message = ''
    if "email" in session:
        return redirect('/dashboard')

    if request.method == "POST":
        user = request.form.get("name")
        email = request.form.get("email")
        
        password1 = request.form.get("password")
        password2 = request.form.get("cpassword")

        user_found = db.user.find_one({"name": user})
        email_found = db.user.find_one({"email": email})
        
        user_key = secrets.token_urlsafe(16)

        if user_found:
            message = 'There already is a user by that name'
            return render_template('login.html', message=message)
        if email_found:
            message = 'This email already exists in database'
            return render_template('login.html', message=message)
        if password1 != password2:
            message = 'Passwords should match!'
            return render_template('login.html', message=message)

        else:
            hashed = bcrypt.hashpw(password2.encode('utf-8'), bcrypt.gensalt())
            user_input = {'name': user, 'email': email, 'password': hashed, 'key': user_key}
            db.user.insert_one(user_input)
            
            user_data = db.user.find_one({"email": email})
            new_email = user_data['email']
   
            return redirect('/dashboard')

    return render_template('register.html')

#Login
@app.route('/login', methods=["POST", "GET"])
def login():
    message = 'Please login to your account'
    if "email" in session:
        return redirect('/dashboard')

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
       
        email_found = db.user.find_one({"email": email})
        if email_found:
            email_val = email_found['email']
            passwordcheck = email_found['password']
            
            if bcrypt.checkpw(password.encode('utf-8'), passwordcheck):
                session["email"] = email_val
                return redirect('/dashboard')
            else:
                if "email" in session:
                    return redirect(url_for("logged_in"))
                message = 'Wrong password'
                return render_template('login.html', message=message)
        else:
            message = 'Email not found'
            return render_template('login.html', message=message)

    return render_template('login.html', message=message)

#logout
@app.route("/logout", methods=["POST", "GET"])
def logout():
    if "email" in session:
        session.pop("email", None)
        return redirect('/login')
    else:
        return redirect('/login')


@app.route('/documentation')
def doc():
    return render_template('doc.html')

@app.route('/doc')
def doc_page():
    return render_template('doc-page.html')

@app.route('/doc/<string:id>')
def specific_section(id):
    print(id)
    return redirect('http://127.0.0.1:8000/doc'+id)

#-------------------------------------------------------------------------#
# Dashboard
@app.route("/dashboard", methods=["POST", "GET"])
def dashboard():
    if "email" in session:
        user = db.user.find({'email': session['email']})
        for x in user:
            name = x
        return render_template('dashboard.html', name=name['name'], key=name['key'])
    else:
        return redirect('/login')

@app.route('/upload', methods=['GET'])
def uploadget():
    if "email" in session:
        return render_template('upload.html')
    else:
        return redirect('/login')