import asyncio
import datetime
import time
import pandas as pd

from azure.eventhub.aio import EventHubProducerClient
from azure.eventhub import EventData
import random
import json
import argparse
import os
import time
def parse_args():
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", default=100, type=int, help="duration to run the simulator")
    parser.add_argument("--avail_duration", default=5, type=int, help="duration for car to appear as available in a location from start time")

    parser.add_argument("--request_eh", type=str, default="requests")
    parser.add_argument("--supply_eh", type=str, default="availability")
    # parse args
    args = parser.parse_args()

    # return args
    return args


locations = ["loc_"+str(i) for i in range(20)]
riders = ["r_"+str(i) for i in range(100000)]
cars = ["c_"+str(i) for i in range(1000)]
types = ["x", "xl", "green", "comfort"]
con_str="Endpoint=sb://kafkaeventhub01.servicebus.windows.net/;SharedAccessKeyName=new;SharedAccessKey=rCRM1EPvMROfVvYmm/T9Yolu5cEc065nIPxz/IpFsZ8="
# con_str = os.getenv('EH_CONN')
async def send_ride_request_batch(producer):
    # Without specifying partition_id or partition_key
    # the events will be distributed to available partitions via round-robin.
    for request_type in random.choices(types,k=len(types)):
        event_batch = await producer.create_batch(partition_key=request_type)
        for from_location in random.choices(locations,k=len(locations))  :            
            for to_location in random.choices(locations,k=len(locations)):
                timestamp= datetime.datetime.now()
                id = "r_"+from_location+"_"+to_location+"_"+request_type+ str(timestamp.timestamp())
                record = json.dumps({"id":id,"timestamp":str(timestamp), "from_loc":from_location, "to_loc":to_location, "request_type":request_type})
                event_batch.add(EventData(record))
                time.sleep(0.00001)
        await producer.send_batch(event_batch)
async def send_car_availability_batch(producer,avail_duration):
    # Without specifying partition_id or partition_key
    # the events will be distributed to available partitions via round-robin.
    for car_type in random.choices(types,k=len(types)):
        event_batch = await producer.create_batch(partition_key=car_type)
        for location in random.choices(locations,k=len(locations)):
            for _ in range(len(locations))  :            
                starttime = datetime.datetime.now()
                endtime = starttime+ datetime.timedelta(avail_duration)
                car = random.choice(cars)
                id = "c_"+location+"_"+car_type+"_"+ str(starttime.timestamp())
                record = json.dumps({"id":id,"starttime":str(starttime),"endtime":str(endtime), "location":location, "car_type":car_type, "car_id":car})
            event_batch.add(EventData(record))
            time.sleep(0.00001)
        await producer.send_batch(event_batch)

async def send_ride_requests(eh_name):
    producer = EventHubProducerClient.from_connection_string(conn_str=con_str, eventhub_name=eh_name)
    async with producer:
        await send_ride_request_batch(producer)
async def send_car_availability(avail_duration, eh_name):
    producer = EventHubProducerClient.from_connection_string(conn_str=con_str, eventhub_name=eh_name)
    async with producer:
        await send_car_availability_batch(producer,avail_duration)
async def main(avail_duration,request_eh,supply_eh ):
    task1 = asyncio.create_task(send_ride_requests(request_eh))

    task2 = asyncio.create_task(send_car_availability(avail_duration,supply_eh))

    print(f"started at {time.strftime('%X')}")

    # Wait until both tasks are completed (should take
    # around 2 seconds.)
    await task1
    await task2

    print(f"finished at {time.strftime('%X')}")

# # run script
if __name__ == "__main__":
    args = parse_args()
    duration = args.duration
    avail_duration = args.avail_duration
    request_eh = args.request_eh
    supply_eh = args.supply_eh
    execution_time=0
    batch =0
    while execution_time <duration:
        batch += 1
        start_time = time.time()
        asyncio.run(main(avail_duration,request_eh,supply_eh ))

        batch_time = time.time() - start_time
        execution_time += batch_time
        print("Batch {0}, sent {1} messages in {2} seconds.".format(batch, len(locations)*len(locations)*len(types), batch_time))
    # run main function
