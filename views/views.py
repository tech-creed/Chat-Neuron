from flask import make_response,Flask, flash, redirect, render_template, request, url_for, session
from app import *
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
            user_input = {'name': user, 'email': email, 'password': hashed}
            db.user.insert_one(user_input)
            
            user_data = db.user.find_one({"email": email})
            new_email = user_data['email']
   
            return redirect('/dashboard')

    return render_template('register.html')