 function getAttributeValues(new_event_times, new_event_values, new_attributes,  baseline_attribute_values, baseline_time, msr_eventtime) {
  // sort new_event_times, new_event_values and new_attributes in chronological order
 baseline_attribute_values = baseline_attribute_values.split(' ');

  // if msr_eventtime is before baseline_time, return baseline_attribute_values
  if ((new Date(msr_eventtime) < new Date(baseline_time)) || new_event_values == null) {
    //console.log("Return baseline")
    return baseline_attribute_values;
  }
    // Create an object to store the baseline attributes and their values

  const base_line_attribute_names = ["attr_1","attr_2","attr_3","attr_4","attr_5","attr_6","attr_7","attr_8","attr_9", "attr_10"];
  const sortedEvents = new_event_times
    .map((time, index) => ({ time, value: new_event_values[index], attribute: new_attributes[index] }))
    .sort((a, b) => new Date(a.time) - new Date(b.time));



  // get the closest timestamp before msr_eventtime
  let closestTimestamp =null;
  for (let i = sortedEvents.length-1; i>=0  ; i--) {
    if (new Date(sortedEvents[i].time) < new Date(msr_eventtime)) {
      //console.log("find closest earlier event")
      closestTimestamp = sortedEvents[i].time;
      break;
    }

  }
 if ((closestTimestamp == null) || (closestTimestamp === baseline_time)){
    //console.log("all new attr events are later than measurement event")
    return baseline_attribute_values;

 }


  // create a map from attribute names to values for baseline_attribute_values
  const baselineValuesMap = new Map();

  base_line_attribute_names.forEach((attribute, index) => {
    baselineValuesMap.set(attribute, baseline_attribute_values[index]);
  });


  // travel from baseline_time to closestTimestamp, updating values for new events
  let currentTime = baseline_time;
  console.log(baseline_time)

  for (let i = 0; i < sortedEvents.length; i++) {

    if (new Date(sortedEvents[i].time) > new Date(currentTime)) {

      baselineValuesMap.set(sortedEvents[i].attribute, sortedEvents[i].value);
      currentTime = sortedEvents[i].time;
    }
    if (sortedEvents[i].time === closestTimestamp) {
      break;
    }
  }

  // return the attribute values as a list
  return Array.from(baselineValuesMap.values());
}