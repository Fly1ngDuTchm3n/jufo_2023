import pandas as pd
import numpy as np

spikes_per_cluster = 3
values_per_spike = 30
min_spike_size = values_per_spike / 3


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


def spikes_to_np_mat(old_data_frame, spike_data):
    """
    Every row in the Dataset gets {SpikesPerCluster} clusters.
    So every row becomes {SpikesPerCluster * timesPerSpike} elements long.
    If we have a spike (oldAspSpike != 0) we put the data in the next position of the row.
    Every second spike, we start a new row.
    If we have gone through all positions in the row but the spike is not over, we skip the rest.
    If the spike is over and there is still room in the row, fill it upp with 0s.
    """
    spike_counter = len(spike_data)
    highest_val = 0
    while highest_val == 0:
        spike_counter -= 1
        highest_val = max(highest_val, spike_data[spike_counter])

    np_mat = np.zeros(
        shape=(
            int(np.ceil(highest_val / spikes_per_cluster)),
            spikes_per_cluster * values_per_spike,
        ),
        dtype=np.int16,
    )

    # keep track of the position in the row
    data_counter = -1
    restart_counter_on1_index = False
    counter = 0

    for idx in old_data_frame.index:
        if spike_data[idx] == 0:
            continue
        if restart_counter_on1_index and spike_data[idx] % spikes_per_cluster == 1:
            counter += 1

            data_counter = 0
            restart_counter_on1_index = False
        else:
            data_counter += 1
            if spike_data[idx] % spikes_per_cluster == 0:
                restart_counter_on1_index = True
            if data_counter >= spikes_per_cluster * values_per_spike:
                continue
        np_mat[counter][data_counter] = old_data_frame["ADC_diff"][idx]
    return np_mat


def fill_up_short_rows(np_mat):
    """
    In every row we count the amount of trailing zeros.
    Otherwise, we copy some values in between to make up for the missing values.
    To compensate that the values are higher due to quick speed we multiply everything with a multiplier.
    The copying is done by first putting everything in a separate list (needed because we insert new values in between)
    and overriding the dataframe with the list.
    """
    for idx in range(len(np_mat)):
        zero_counter = 0
        idx_list = []
        for idx2 in reversed(range(spikes_per_cluster * values_per_spike)):
            if np_mat[idx][idx2] != 0:
                break
            zero_counter += 1

        if zero_counter == 0:
            continue

        if (
            zero_counter
            > spikes_per_cluster * values_per_spike
            - spikes_per_cluster * min_spike_size
        ):
            print("ERROR something wrong")
            continue

        copy_each_n_elements = zero_counter / (
            spikes_per_cluster * values_per_spike - zero_counter
        )
        total = 0

        multiplier = (spikes_per_cluster * values_per_spike - zero_counter) / (
            spikes_per_cluster * values_per_spike
        )

        for idx2 in range(spikes_per_cluster * values_per_spike - zero_counter):
            total += copy_each_n_elements
            idx_list.append(round(np_mat[idx][idx2] * multiplier))
            while total > 1:
                idx_list.append(
                    round((np_mat[idx][idx2] + np_mat[idx][idx2 + 1]) / 2 * multiplier)
                )
                total -= 1

        for idx2 in range(min(len(idx_list), spikes_per_cluster * values_per_spike)):
            np_mat[idx][idx2] = idx_list[idx2]


def normalizing_data(data_frame):
    spike_data = count_spikes(data_frame)
    spike_data = remove_short_spikes(spike_data)
    res_data_frame = spikes_to_np_mat(data_frame, spike_data)
    fill_up_short_rows(res_data_frame)
    return res_data_frame


def hand_picked_algorithm(data_mat, isAsp):
    wrong_counter = 0
    for arr in data_mat:
        if check_for_asp(arr) != isAsp:
            wrong_counter += 1
    return wrong_counter


def check_for_asp(arr):
    """
    :param arr: normed array
    :return: idea is that asp has one clear spike while gra has multiple
    """
    count = 0
    pause_after_massive_spike = 0
    for i in range(len(arr) - 1):
        if pause_after_massive_spike > 0:
            pause_after_massive_spike -= 1
            continue
        if arr[i + 1] - arr[i] < -5:
            count += 1
        if arr[i + 1] - arr[i] < -20:
            pause_after_massive_spike = 20

    return count < spikes_per_cluster * 2
