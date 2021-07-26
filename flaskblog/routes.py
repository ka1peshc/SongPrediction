import os
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, abort, jsonify
from flaskblog import app, db, bcrypt, mail
from flaskblog.forms import (RegistrationForm, LoginForm, RequestResetForm, ResetPasswordForm, UpdateAccountForm, MusicRecom)
from flaskblog.models import User
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt


@app.route("/")
@app.route("/home")
def home():
    page = request.args.get('page', 1, type=int)
    return render_template('home.html')

@app.route("/admin")
@login_required
def admin():
    users = User.query.all()
    return render_template('admin.html', title='Admin Page', users=users)

#@app.route("/admin/<int:user_id>/delete", methods=['POST'])
@app.route("/admin/<int:user_id>", methods=['GET','POST'])
@login_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    print("-------In delete_user-------------",user_id)
    db.session.delete(user)
    db.session.commit()
    flash('User deleted successfully ','success')
    return redirect(url_for('admin'))

@app.route("/get_recommendations",methods=['GET','POST'])
@login_required
def get_recommendations():
    form = MusicRecom()
    if form.validate_on_submit():
        songtitle = form.songname.data
        flash('Great success','success')
        return redirect(url_for('displaySong',song_name=songtitle))
    return render_template('music.html',title='Music recommendation',form=form)

@app.route("/displaySong/<song_name>",methods=['GET','POST'])
@login_required
def displaySong(song_name):
    #Read csv
    df = pd.read_csv("D:/MusicRecom/data.csv")
    df_genre = pd.read_csv("D:/MusicRecom/data_by_genres.csv")
    df_year = pd.read_csv("D:/MusicRecom/data_by_year.csv")
    df_artist = pd.read_csv("D:/MusicRecom/data_by_artist.csv")

    #create pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.cluster import KMeans

    #Get the numerical data
    X = df_genre.select_dtypes(np.number)

    cluster_pipeline = Pipeline([("scaler", StandardScaler()), \
                                 ("kmeans", KMeans(n_clusters=20))])
    cluster_pipeline.fit(X)
    df_genre["general_genre"] = cluster_pipeline.predict(X)
    X = df.select_dtypes(np.number)

    ##number_cols = list(X.columns)
    cluster_pipeline.fit(X)
    cluster_labels = cluster_pipeline.predict(X)
    df['general_genre'] = cluster_labels

    from sklearn.decomposition import PCA
    pca_pipeline = Pipeline([('scaler', StandardScaler()), \
                             ('PCA', PCA(n_components=3))])
    genre_PCA = pca_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=["P1", "P2", "P3"], data=genre_PCA)
    projection['cluster'] = df['general_genre']

    distance_cols = list(X.columns)
    df[distance_cols] = df[distance_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    df.drop_duplicates(subset="name", keep="first", inplace=True)


    song_data = df[df["name"]==song_name]
    song_data = song_data.select_dtypes(np.number)
    genre = int(song_data["general_genre"])
    song_data.drop("general_genre", axis=1, inplace=True)
    df_1 = df[df["name"]!=song_name]
    df_1 = df_1[df_1["general_genre"]==genre]
    #We create a df_2 so we can track the song name in df_1
    df_2=df_1[distance_cols]
    
    #Now we get the distance
    point_a = np.array(song_data)
    distance=[]
    for i in range(df_2.shape[0]):
        point_b = np.array(df_2.iloc[i])
        song_distance = np.linalg.norm(point_a-point_b)
        distance.append(song_distance)
    df_1["distance"]=distance
    df_1.sort_values(by="distance",ascending=True, inplace=True)
    df_1.reset_index(inplace=True)
    
    #Now we print out the song recommendation
    rec_song = df_1.loc[0:10]
    #print("Based on your preference, we recommend ")
    recommend_songs = []
    for i in range(10):
        recommend_songs.append(rec_song.loc[i]["name"] + " by "+rec_song.loc[i]["artists"])
        #print(rec_song.loc[i]["name"] + " by "+rec_song.loc[i]["artists"])
    print("-----------recommend song-------------")
    print(recommend_songs)
    return render_template('songlist.html',title='songlist recommend',recommend_songs=recommend_songs)


@app.route("/about")
def about():
    
    return render_template('about.html', title='About')

@app.route('/search', methods=['GET', 'POST'])
def search():
    term = request.form['q']
    print ('term: ', term)
    
    json_data = json.loads(open('D:/RuparelMSc/MScProject/flaskblog/static/resulte.json').read())
    print (json_data)
    #print (json_data[0])
    
    filtered_dict = [v for v in json_data if term in v] 
    #print(filtered_dict)
    
    resp = jsonify(filtered_dict)
    resp.status_code = 200
    return resp

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            if(form.email.data == 'admin123@admin.com'):
                return redirect(url_for('admin'))
                #return render_template('admin.html', title="admin page", users=alluser)
            else:
                return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account',
                           image_file=image_file, form=form)

def send_reset_email(user):
    print(os.environ.get('EMAIL_USER'))
    print(user.email)
    token = user.get_reset_token()
    msg = Message('Password Reset Request',
                  sender='noreply@demo.com',
                  recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}
If you did not make this request then simply ignore this email and no changes will be made.
'''
    mail.send(msg)


@app.route("/reset_password", methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        send_reset_email(user)
        flash('An email has been sent with instructions to reset your password.', 'info')
        return redirect(url_for('login'))
    return render_template('reset_request.html', title='Reset Password', form=form)


@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been updated! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('reset_token.html', title='Reset Password', form=form)