import requests
import json

def slack_notify(user,message):
	msg = {}
	msg["user"] = user
	msg["message"] = message
	json_data = json.dumps(msg)

	r = requests.post("http://127.0.0.1:2111/notification", data={"notification":json_data})
	return r.text