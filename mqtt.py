# MQTT

import paho.mqtt.client as mqtt

topic = "location/coordinates/#"
host = "127.0.0.1"
port = 1883

def on_connect(client, userdata, flags, rc):
    print("Connected [" + client + "] with result code "+str(rc))
    client.subscribe(topic)

def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host, port, 60)
    client.loop_forever()

main()
