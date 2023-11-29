#print hello word
print("Hello World")

import os
from pushbullet import API

# set the API Class and link it to the api object


# now you can use the API Wrapper

# sending a note, arguments: title, message
pushBulletApi = API()
pushBulletApi.set_token("o.W1XnsL76ftQKbpBF9USPeJcnGQQ4TMzx")
pushBulletApi.send_note("Cat alarm", "test")

