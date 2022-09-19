This folder is to develop client simulator applications including the app that generate events and the app that receives real time streaming computation from Cloud
For simulator module. The simulator module create ride request and demand messages to EvenHub.
To run the simulator:
- Create EH namespace and 2 evenhubs, each with 4 partitions
    1. Request_EH
    2. Supply_EH
- Generate a connection string and set the value into the environment variable, e.g.
set EH_CONN="Endpoint=sb://XX.servicebus.windows.net/;SharedAccessKeyName=new;SharedAccessKey=XX/XX/XX="
- Run the simulation program
```python simul.py

