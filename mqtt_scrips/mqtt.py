# MQTT

import paho.mqtt.client as mqtt
import json

topic = "location/coordinates/#"
host = "127.0.0.1"
port = 1883

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe(topic)

def data_processing(payload):
    data = json.loads(payload.decode("utf-8"))
    uid = data['uid']
    x = data['x']
    y = data['y']
    z = data['z']
    timestamp = data['timestamp']
    return uid,x,y,z,timestamp

def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))
    [euid,x,y,z,timestamp] = data_processing(payload)

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host, port, 60)
    client.loop_forever()

main()
