import websocket
from messages import *
from game import *


def beautify_json(j):
    return json.dumps(json.loads(j), indent=4)


def main():
    ws = websocket.WebSocket()
    ws.connect("ws://0.0.0.0:8080/gameServer")

    requestAllMapsMessage = Message(GetAllMapsMessageBody()).dumps()
    print(requestAllMapsMessage)
    ws.send(requestAllMapsMessage)
    recieved = ws.recv()
    mapSettings = [GameMap.from_dict(item) for item in json.loads(recieved)]
    print("Received 1", mapSettings, end="\n")

    startMessage = Message(StartMessageBody(MapId=0, StepsToPlay=10)).dumps()
    print(startMessage)
    ws.send(startMessage)
    data = GameState.from_json(ws.recv())
    print("Received 2", data, end="\n")

    doStepMessage = Message(
        StepMessageBody(StateId=0, PredictedStateUsefulness=1.0)
    ).dumps()
    ws.send(doStepMessage)
    recieved = ws.recv()
    data = Reward.from_json(recieved)
    print("Received 3", data, end="\n")
    ws.close()


if __name__ == "__main__":
    main()
