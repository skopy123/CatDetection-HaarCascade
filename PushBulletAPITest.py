import os
from dotenv import load_dotenv
#print hello word
print("Hello World by PB api test")
load_dotenv()
print(os.getenv("PUSHBULLET_API_KEY"))
import os
from pushbullet import API

# set the API Class and link it to the api object


# now you can use the API Wrapper

# sending a note, arguments: title, message
pushBulletApi = API()
pushBulletApi.set_token(os.getenv("PUSHBULLET_API_KEY"))
pushBulletApi.send_note("Cat alarm", "test")

