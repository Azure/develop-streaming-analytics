from dash import Dash, dcc, html, Input, Output,State
import datetime
import json
from azure.eventhub.aio import EventHubProducerClient
from azure.eventhub import EventData
import os
import asyncio
from azure.cosmos import CosmosClient
import os
import time

app = Dash(__name__)
locations = ["loc_"+str(i) for i in range(50)]
types = ["x", "xl", "green", "comfort"]
con_str=os.getenv("EHCONN")
eh_name= "availability"

app.layout = html.Div([
    html.H3("simulate a car availability"),
    html.Div([
        "Car Location: ",
        dcc.Dropdown(locations, locations[0],id='location')
    ]),
    
    html.Div([
        "Car type: ",
        dcc.Dropdown(types, types[0],id='car_type')
    ]),
    html.Div([
        html.Button('generate', 
            id='generate', 
            n_clicks=0,
            style = {'text-align': 'center', 'border': '1px solid #00ab00'}
        ),
    ], 
        style={'padding': 40, 'flex': 1}
    ),
    html.Br(),
    html.Div( html.H2(id='matched_trip')),

])


async def send_availability_batch(producer, car_type, location):
    # Without specifying partition_id or partition_key
    # the events will be distributed to available partitions via round-robin.
    starttime = datetime.datetime.now()
    endtime = starttime+ datetime.timedelta(5)
    id = "c_"+location+"_"+car_type+"_"+ str(starttime.timestamp())
    record = json.dumps({"id":id,"starttime":str(starttime),"endtime":str(endtime), "location":location, "car_type":car_type, "car_id":id})
    event_batch = await producer.create_batch(partition_key=car_type)

    event_batch.add(EventData(record))

    await producer.send_batch(event_batch)
    return id
async def send_request(car_type, location):
    producer = EventHubProducerClient.from_connection_string(conn_str=con_str, eventhub_name=eh_name)
    async with producer:
        id =await  send_availability_batch(producer, car_type, location)
        return id

@app.callback(
    Output(component_id='matched_trip', component_property='children'),
        State('car_type', 'value'), 
        State('location', 'value'), 
        Input('generate', 'n_clicks'),prevent_initial_call=True)
def trip_matching(car_type, location, n_click):
    
    id= asyncio.run(send_request(car_type, location))
    print(id)


    return f'Generate car availibility with id {id} with type {car_type} at location {location}'


if __name__ == '__main__':
    app.run_server(debug=True,port=8051)
