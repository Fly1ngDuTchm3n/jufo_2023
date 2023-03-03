from spikeclustering import normalizing_data, hand_picked_algorithm
from my_ai import build_ai
from fourier import replace_with_fft, hand_picked_fft_algorithm
import pandas as pd
import numpy as np

oldAsp = pd.read_csv("datasets/ASPHALT_06.02.CSV")
oldAsp.drop(["ADC_filtered", "current", "charging_time", "time"], axis=1)
oldGra = pd.read_csv("datasets/GRAVEL_06.02.CSV")
oldGra.drop(["ADC_filtered", "current", "charging_time", "time"], axis=1)

aspDs = normalizing_data(oldAsp)
graDs = normalizing_data(oldGra)

four_mat_asp = replace_with_fft(aspDs)
four_mat_gra = replace_with_fft(graDs)

print(
    f"accuracy four total = {1 - (hand_picked_fft_algorithm(four_mat_asp, isAsp=True) + hand_picked_fft_algorithm(four_mat_gra, isAsp=False))/(len(four_mat_asp) + len(four_mat_gra))}"
)
print(
    f"accuracy handy total = {1 - (hand_picked_algorithm(aspDs, isAsp=True) + hand_picked_algorithm(graDs, isAsp=False))/(len(four_mat_asp) + len(four_mat_gra))}"
)

aspLab = np.full((len(aspDs)), "Asphalt")
graLab = np.full((len(graDs)), "Kies")

totalSet = np.concatenate((aspDs, graDs))
totalSet2 = np.concatenate((four_mat_asp, four_mat_gra))
totalLab = np.concatenate((aspLab, graLab))


print(
    f"Genauigkeit maschinelles lernen auf normierten Daten: {build_ai(totalSet, totalLab)}"
)
print(f"Genauigkeit fft mit maschinellem lernen: {build_ai(totalSet2, totalLab)}")
