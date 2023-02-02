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
eh_name= "requests"

app.layout = html.Div([
    html.H3("Request a ride"),
    html.Div([
        "From Location: ",
        dcc.Dropdown(locations, locations[0],id='from-location')
    ]),
    html.Br(),
    html.Div([
        "To Location: ",
        dcc.Dropdown(locations, locations[0],id='to-location')
    ]),
    html.Div([
        "Car type: ",
        dcc.Dropdown(types, types[0],id='type')
    ]),
    html.Div([
        html.Button('Request', 
            id='request', 
            n_clicks=0,
            style = {'text-align': 'center', 'border': '1px solid #00ab00'}
        ),
    ], 
        style={'padding': 40, 'flex': 1}
    ),
    html.Br(),
    html.Div(html.H2(id='matched_trip')),

])


async def send_ride_request_batch(producer, type, from_location, to_location):
    # Without specifying partition_id or partition_key
    # the events will be distributed to available partitions via round-robin.
    timestamp= datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f0Z")
    id = "r_"+from_location+"_"+to_location+"_"+type+ timestamp

    record = json.dumps({"id":id,"timestamp":timestamp, "from_loc":from_location, "to_loc":to_location, "request_type":type})
    event_batch = await producer.create_batch(partition_key=type)

    event_batch.add(EventData(record))

    await producer.send_batch(event_batch)
    return id
async def send_request(type, from_location, to_location):
    producer = EventHubProducerClient.from_connection_string(conn_str=con_str, eventhub_name=eh_name)
    async with producer:
        id =await  send_ride_request_batch(producer, type, from_location, to_location)
        return id

@app.callback(
    Output(component_id='matched_trip', component_property='children'),
        State('type', 'value'), 
        State('from-location', 'value'), 
        State('to-location', 'value'), 
        Input('request', 'n_clicks'),prevent_initial_call=True)
def trip_matching(type, from_location, to_location, n_click):
    
    id= asyncio.run(send_request( type, from_location, to_location))
    print(id)

    URL ="https://cosmosdbnative01.documentps.azure.com:443/"
    KEY=os.getenv("COSMOSKEY")
    client = CosmosClient(URL, credential=KEY)
    DATABASE_NAME = 'rideservice'
    database = client.get_database_client(DATABASE_NAME)
    CONTAINER_NAME = 'matched_trips'
    container = database.get_container_client(CONTAINER_NAME)

    # Enumerate the returned items
    import json
    matched_trip =None
    price = None
    i =0
    start_time = time.time()
    while True and i <200:
    
        for item in container.query_items(
                query=f'SELECT * FROM c  WHERE c.request_id="{id}"',
                enable_cross_partition_query=True):
                matched_trip = item['request_id']
                price = item['actual_price']
                break
        time.sleep(0.01)
        i+=1
    if matched_trip:
        duration = time.time()-start_time
        return f'Found matching {type} car in {duration} seconds from {from_location} to {to_location}, price {price}, id {id}'
    else:
        return "No available car found, please try again"


if __name__ == '__main__':
    app.run_server(debug=True,port=8052)
