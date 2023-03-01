from spikeclustering import normalizing_data, normed_dataframe_to_list, spikes_per_cluster, values_per_spike
from my_ai import build_ai
from src.fourier import fft_check_for_asp
import pandas as pd


oldAsp = pd.read_csv("datasets/ASPHALT_06.02.CSV")
oldAsp.drop(["ADC_filtered", "current", "charging_time", "time"], axis=1)
oldGra = pd.read_csv("datasets/GRAVEL_06.02.CSV")
oldGra.drop(["ADC_filtered", "current", "charging_time", "time"], axis=1)

aspDs = normalizing_data(oldAsp)
graDs = normalizing_data(oldGra)


four_asp = normed_dataframe_to_list(aspDs)
four_gra = normed_dataframe_to_list(graDs)


aspLab = pd.Series(["Asp"] * len(aspDs))
graLab = pd.Series(["Gra"] * len(graDs))

totalSet = [aspDs, graDs]
totalSet = pd.concat(totalSet, ignore_index=True)
totalLab = [aspLab, graLab]
totalLab = pd.concat(totalLab, ignore_index=True)


build_ai(totalSet, totalLab)
