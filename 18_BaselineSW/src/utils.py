import requests
import json
def post_message(name='bot', message='test', incoming_webhook_url='none'):
    if incoming_webhook_url != 'none':
        requests.post(incoming_webhook_url, data = json.dumps({
            'text':message,
            'username':name,
        }))