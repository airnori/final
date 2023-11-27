# 필요한 모듈을 불러옵니다.
from flask import Flask, render_template, redirect, url_for, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
import requests
from sqlalchemy.orm import Session

app = Flask(__name__)  # Flask 앱 인스턴스를 생성합니다.
app.config['SECRET_KEY'] = 'your_secret_key'  # 앱의 보안 키를 설정합니다.
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'  # 데이터베이스 URI를 설정합니다.
db = SQLAlchemy(app)  # SQLAlchemy 객체를 생성합니다.

login_manager = LoginManager()  # 로그인 매니저 객체를 생성합니다.
login_manager.init_app(app)  # 로그인 매니저를 앱과 연동합니다.
login_manager.login_view = 'login'  # 로그인 페이지의 뷰 함수를 설정합니다.

# User 모델을 정의합니다. 이는 SQLAlchemy를 이용해 데이터베이스에 저장됩니다.
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# 로그인 매니저의 user_loader 데코레이터를 사용해 사용자를 로드하는 함수를 정의합니다.
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# '/' 경로에 접속하면 홈페이지를 보여주는 뷰 함수입니다. 로그인이 필요합니다.
@app.route('/')
@login_required
def home():
    return render_template('/html/home.html', name=current_user.username)

# '/login' 경로에 대한 뷰 함수입니다. POST 요청이 들어오면 사용자 로그인을 처리합니다.
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            # 사용자 이름을 다른 서버에 보내는 코드 부분입니다.
            url = "http://10.191.144.201:5001/receive-username"  
            data = {"username": username}
            response = requests.post(url, data=data)
            if response.status_code == 200:
                print("Username successfully sent to other server.")
            else:
                print("Failed to send username to other server.")
            
            return redirect(url_for('home'))
        else:
            return redirect(url_for('login'))

    return render_template('/html/login.html')

# '/signup' 경로에 대한 뷰 함수입니다. POST 요청이 들어오면 사용자 등록을 처리합니다.
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    username_exists = False

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user:
            username_exists = True
        else:
            new_user = User(username=username, password=generate_password_hash(password, method='pbkdf2:sha256'))
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))

    return render_template('/html/signup.html', username_exists=username_exists)

# '/logout' 경로에 대한 뷰 함수입니다. 로그아웃을 처리합니다.
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# '/service' 경로에 대한 뷰 함수입니다. 로그인이 필요합니다.
@app.route('/service')
@login_required
def service():
     return render_template("/html/service.html")

# 메인 실행 부분입니다. 앱 컨텍스트에서 데이터베이스를 생성하고 앱을 실행합니다.
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
