import asyncio
import datetime
import time
import pandas as pd
import pytz
from azure.eventhub.aio import EventHubProducerClient
from azure.eventhub import EventData
import random
import json
import argparse
import os
import time
import numpy as np
def parse_args():
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", default=1000, type=int, help="duration to run the simulator")
    parser.add_argument("--num_ep", default=2, type=int, help="number of devices")
    parser.add_argument("--num_attr", default=10, type=int, help="number of attributes per device")
    parser.add_argument("--num_msr", default=10, type=int, help="number of measurements per device")

    parser.add_argument("--attribute_eh", type=str, default="endpoint_attributes")
    parser.add_argument("--measurement_eh", type=str, default="endpoint_measurements")

    # parse args
    args = parser.parse_args()

    # return args
    return args



# con_str = os.getenv('EH_CONN')
async def send_attribute_batch(producer,endpoints, attributes,attr_cat_vals,attr_num_vals):
    # Without specifying partition_id or partition_key
    # the events will be distributed to available partitions via round-robin.
    for ep_id in random.choices(endpoints,k=len(endpoints)):
        event_batch = await producer.create_batch(partition_key=ep_id)
        for attr in random.choices(attributes,k=len(attributes))  : 
            if int(attr[5:])%2 ==0:
                attr_val = random.choice(attr_cat_vals)
            else:
                attr_val = random.choice(attr_num_vals)

                timestamp= datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                record = json.dumps({"eventtime":timestamp, "ep_id":ep_id,"attr":attr, "attr_val":str(attr_val)})
                event_batch.add(EventData(record))
                time.sleep(0.01)
        await producer.send_batch(event_batch)
async def send_measurement_batch(producer,endpoints,measurement, msr_vals):
    # Without specifying partition_id or partition_key
    # the events will be distributed to available partitions via round-robin.
    for ep_id in random.choices(endpoints,k=len(endpoints)):
        event_batch = await producer.create_batch(partition_key=ep_id)
        for msr in random.choices(measurement,k=len(measurement))  : 
            msr_val = random.choice(msr_vals)
            timestamp= datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

            record = json.dumps({"ep_id":ep_id,"eventtime":timestamp,"msr":msr, "msr_val":msr_val})
            event_batch.add(EventData(record))
            time.sleep(0.000001)
        await producer.send_batch(event_batch)

async def send_attributes(eh_name,endpoints, attributes,attr_cat_vals,attr_num_vals):
    producer = EventHubProducerClient.from_connection_string(conn_str=con_str, eventhub_name=eh_name)
    async with producer:
        await send_attribute_batch(producer,endpoints, attributes,attr_cat_vals,attr_num_vals)
async def send_measurements(eh_name,endpoints,measurements, msr_vals):
    producer = EventHubProducerClient.from_connection_string(conn_str=con_str, eventhub_name=eh_name)
    async with producer:
        await send_measurement_batch(producer,endpoints,measurements, msr_vals)
async def main(attribute_eh,measurement_eh,endpoints,measurements, msr_vals,attr_cat_vals,attr_num_vals ):
    task = random.choices([1,2], [0.3, 0.7],k=1)
    if task[0]==1:
        task1 = asyncio.create_task(send_attributes(attribute_eh,endpoints, attributes,attr_cat_vals,attr_num_vals))
        await task1

    else:
        task2 = asyncio.create_task(send_measurements(measurement_eh,endpoints,measurements, msr_vals))
        await task2

    print(f"started at {time.strftime('%X')}")

    # Wait until both tasks are completed (should take
    # around 2 seconds.)
    

    print(f"finished at {time.strftime('%X')}")
    return task[0]

# # run script
if __name__ == "__main__":
    args = parse_args()
    duration = args.duration
    attribute_eh = args.attribute_eh
    measurement_eh = args.measurement_eh
    execution_time=0
    batch =0
    endpoints = ["ep_"+str(i) for i in range(args.num_ep)]
    attributes = ["attr_"+str(i) for i in range(args.num_attr)]
    attr_cat_vals = ["cat_"+str(i) for i in range(20)]
    attr_num_vals = np.linspace(0,100,100)

    measurements = ["msr_"+str(i) for i in range(args.num_msr)]
    msr_vals = np.linspace(0,100,1000)
    con_str=os.getenv("EHCONN")
    while execution_time <duration:
        batch += 1
        start_time = time.time()
        task = asyncio.run(main(attribute_eh,measurement_eh,endpoints,measurements, msr_vals,attr_cat_vals,attr_num_vals  ))

        batch_time = time.time() - start_time
        execution_time += batch_time
        if task ==2:
            print("Batch {0}, sent {1} measure values in {2} seconds.".format(batch, len(endpoints)*len(measurements), batch_time))
        else:
            print("Batch {0}, sent {1} attribute values in {2} seconds.".format(batch, len(endpoints)*len(attributes), batch_time))

    # run main function
