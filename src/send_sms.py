from twilio.rest import Client


# Your Account Sid and Auth Token from twilio.com/console
account_sid = 'AC48a2b57630cde3ad7acc662ea91cf5fd'
auth_token = '101da4d773c821ed0c60d7f7dd17cb98'
client = Client(account_sid, auth_token)

message = client.messages \
                .create(
                     body="Join Earth's mightiest heroes. Like Kevin Bacon.",
                     from_='+15052786996',
                     to='+918826748151'
                 )

print(message.sid)