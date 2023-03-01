import pandas as pd

spikes_per_cluster = 3
values_per_spike = 30
min_spike_size = values_per_spike/3


def count_spikes(data_frame: pd.DataFrame):
    spike_counter = 0
    spike_data = []
    is_in_spike = False

    for idx in data_frame.index:
        # edge case removing
        if idx < 2 or idx > len(data_frame) - 3:
            spike_data.append(0)
            continue

        # discarding everything with a diff to low
        if (
                abs(data_frame["ADC_diff"][idx]) >= 3
                or abs(data_frame["ADC_diff"][idx - 1]) >= 3
                or abs(data_frame["ADC_diff"][idx - 2]) >= 3
                or abs(data_frame["ADC_diff"][idx + 1]) >= 3
                or abs(data_frame["ADC_diff"][idx + 2]) >= 3
        ):
            # start of new spike -> increase Spike counter
            if not is_in_spike:
                spike_counter += 1
                is_in_spike = True
            spike_data.append(spike_counter)

        else:
            spike_data.append(0)
            is_in_spike = False

    return spike_data


def remove_short_spikes(spike_data):
    # the spike numbers of too short spikes get in here
    rem_spike = []
    # keep track of the spike we are working with
    act_val = 0
    # count the length of each spike
    act_count = 1

    for val in spike_data:
        if val == 0:
            continue
        if val == act_val:
            act_count += 1
        else:
            if act_count < min_spike_size:
                rem_spike.append(act_val)
            act_val = val
            act_count = 1
    # keep track of the amount of removed lines because the other spikes must be set back
    rem_counter = 0
    for idx, val in enumerate(spike_data):
        if val == 0:
            continue
        if val in rem_spike:
            if val != act_val:
                rem_counter += 1
                act_val = val
            spike_data[idx] = 0
        else:
            spike_data[idx] -= rem_counter
    return spike_data


def spikes_to_data_frame(old_data_frame, spike_data):
    """
    Every row in the Dataset gets {SpikesPerCluster} clusters.
    So every row becomes {SpikesPerCluster * timesPerSpike} elements long.
    If we have a spike (oldAspSpike != 0) we put the data in the next position of the row.
    Every second spike, we start a new row.
    If we have gone through all positions in the row but the spike is not over, we skip the rest.
    If the spike is over and there is still room in the row, fill it upp with 0s.
    """
    # setting up the dict
    new_data_frame_as_dict = {}
    for i in range(0, spikes_per_cluster * values_per_spike):
        new_data_frame_as_dict.update({f"ADC_diff{i}": []})
    #    newDataFrame.update({f'ADC_diff{i}': [], f'time{i}': []})

    # keep track of the position in the row
    data_counter = -1
    restart_counter_on1_index = False
    counter = 0

    for idx in old_data_frame.index:
        if spike_data[idx] == 0:
            continue
        if restart_counter_on1_index and spike_data[idx] % spikes_per_cluster == 1:
            counter += 1
            for idx2 in range(data_counter + 1, spikes_per_cluster * values_per_spike):
                new_data_frame_as_dict[f"ADC_diff{idx2}"].append(0)
            #    newDataFrame[f'time{idx2}'].append(0)

            data_counter = 0
            restart_counter_on1_index = False
        else:
            data_counter += 1
            if spike_data[idx] % spikes_per_cluster == 0:
                restart_counter_on1_index = True
            if data_counter >= spikes_per_cluster * values_per_spike:
                continue
        new_data_frame_as_dict[f"ADC_diff{data_counter}"].append(
            old_data_frame["ADC_diff"][idx]
        )
    #    newDataFrame[f'time{dataCounter}'].append(oldDataFrame['time'][idx])

    # fill rest of row with 0
    for idx in range(data_counter + 1, spikes_per_cluster * values_per_spike):
        new_data_frame_as_dict[f"ADC_diff{idx}"].append(0)
    #    newDataFrame[f'time{idx}'].append(0)
    return pd.DataFrame(new_data_frame_as_dict)


def fill_up_short_rows(data_frame):
    """
    In every row we count the amount of trailing zeros.
    Otherwise, we copy some values in between to make up for the missing values.
    To compensate that the values are higher due to quick speed we multiply everything with a multiplier.
    The copying is done by first putting everything in a separate list (needed because we insert new values in between)
    and overriding the dataframe with the list.
    """
    for idx in data_frame.index:
        zero_counter = 0
        idx_list = []
        for idx2 in reversed(range(0, spikes_per_cluster * values_per_spike)):
            if data_frame[f"ADC_diff{idx2}"][idx] != 0:
                break
            zero_counter += 1

        if zero_counter == 0:
            continue

        if zero_counter > spikes_per_cluster * values_per_spike - spikes_per_cluster * min_spike_size:
            print("ERROR something wrong")
            continue

        copy_each_n_elements = zero_counter / (spikes_per_cluster * values_per_spike - zero_counter)
        total = 0

        multiplier = (spikes_per_cluster * values_per_spike - zero_counter) / (
                spikes_per_cluster * values_per_spike
        )

        for idx2 in range(0, spikes_per_cluster * values_per_spike - zero_counter):
            total += copy_each_n_elements
            idx_list.append(round(data_frame[f"ADC_diff{idx2}"][idx] * multiplier))
            while total > 1:
                idx_list.append(round((data_frame[f"ADC_diff{idx2}"][idx] + data_frame[f"ADC_diff{idx2 + 1}"][idx])/2
                                      * multiplier))
                total -= 1

        for idx2 in range(0, min(len(idx_list), spikes_per_cluster * values_per_spike)):
            data_frame[f"ADC_diff{idx2}"][idx] = idx_list[idx2]


def normed_dataframe_to_list(data_frame
                             +
):
    ret_list = []
    for i in data_frame.index:
        for j in range(0, spikes_per_cluster * values_per_spike):
            ret_list.append(data_frame[f"ADC_diff{j}"][i])
    return ret_list


def normalizing_data(data_frame):
    spike_data = count_spikes(data_frame)
    spike_data = remove_short_spikes(spike_data)
    res_data_frame = spikes_to_data_frame(data_frame, spike_data)
    return fill_up_short_rows(res_data_frame)


