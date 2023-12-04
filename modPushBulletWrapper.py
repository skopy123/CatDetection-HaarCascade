import os
import time
from pushbullet import API
#print hello word
#print("Hello World by PB api test")

#print(os.getenv("PUSHBULLET_API_KEY"))

# sending a note, arguments: title, message
pushBulletApi = API()
pushBulletApi.set_token(os.getenv("PUSHBULLET_API_KEY"))
#pushBulletApi.send_note("Cat alarm", "test")

def SendNotificationImediate(message):
    pushBulletApi.send_note("Cat alarm", message)

lastNotificationTime = time.time() - 60

def SendNotificationWithRateLimiter(message):
    global lastNotificationTime
    if (time.time() - lastNotificationTime) > 60:
        lastNotificationTime = time.time()
        pushBulletApi.send_note("Cat alarm", message)

#TODO send images with messages

