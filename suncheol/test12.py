import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import db

## Fetch the service account key JSON file contents
#cred = credentials.Certificate("mykey.json")

## Initialize the app with a service account, granting admin privileges
#app = firebase_admin.initialize_app(cred, {
#    'storageBucket': 'https://cap-vino.firebaseio.com/',
#}, name='storage')
#
#bucket = storage.bucket(app=app)
#blob = bucket.blob("<your_blob_path>")
#
#print(blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET'))


cred = credentials.Certificate('mykey.json')
default_app = firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://cap-vino.firebaseio.com/'})
ref = db.reference('/')
users_ref = ref.child('request')

#users_ref.set({'87da01bd8047adb0' : {'output' : '승환아~'}})

#users_ref.update({'output' : '승환아~1'})
        
users_ref.update({'ID' : '승환아~2',
                  'URL':'https://firebasestorage.googleapis.com/v0/b/cap-vino.appspot.com/o/images%2F87da01bd8047adb0.jpg?alt=media&token=36461d10-f88d-4337-97e7-57227739a344',
                  'Wine_Name':'yo wine'})
#users_ref.update({'87da01bd8047adb0' : {'output' : '승환아~2'}})

aa_ref = users_ref.child('87da01bd8047adb0/url') #url값 reference

print(aa_ref.get())#url값 출력

#print(ref.get())

#users_ref.update({'user' : {'output' : 'null', 'url':'null'}})
