function buildHistory(newEventTime, newEventValue, newAttribute, baselineEventValue, baselineEventTime) {
    // Create an object to store the baseline attributes and their values
    const baseLineAttributeName = ["attr_1","attr_2","attr_3","attr_4","attr_5","attr_6","attr_7","attr_8","attr_9", "attr_10"];
    const baselineAttributes = Object.fromEntries(baseLineAttributeName.map((k, i) => [k, baselineEventValue[i]]));
  
    // Create an array to store the history of records
    const history = [];
  
    // Convert the baseline event time to a Date object
    const baselineTime = new Date(baselineEventTime);
  
    // Add the baseline attributes and time to the history array
    history.push({ timestamp: baselineTime, attributes: baselineAttributes });
  
    // Sort the new events by their timestamps
    const newEvents = newEventTime
      .map((t, i) => [new Date(t), newEventValue[i], newAttribute[i]])
      .sort((a, b) => a[0] - b[0]);
  
    // Travel backward to build the history of events before the baseline
    for (let i = newEvents.length - 1; i >= 0; i--) {
      const [eventTime, eventValue, eventAttribute] = newEvents[i];
      if (eventTime < baselineTime) {
        // If the event timestamp is before the baseline, add it to the history
        const attributes = { ...baselineAttributes };
        attributes[eventAttribute] = eventValue;
        history.push({ timestamp: eventTime, attributes });
      } else {
        // Stop when the timestamps are no longer before the baseline
        break;
      }
    }
  
    // Travel forward to build the history of events after the baseline
    for (let i = 0; i < newEvents.length; i++) {
      const [eventTime, eventValue, eventAttribute] = newEvents[i];
      if (eventTime > baselineTime) {
        // If the event timestamp is after the baseline, add it to the history
        const attributes = { ...baselineAttributes };
        attributes[eventAttribute] = eventValue;
        history.push({ timestamp: eventTime, attributes });
      } else {
        // Stop when the timestamps are no longer after the baseline
        break;
      }
    }
  
    // Sort the history by timestamp
    history.sort((a, b) => a.timestamp - b.timestamp);
  
    return history;
  }
  