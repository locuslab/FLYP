import requests
import os
import multiprocessing as mp
from io import BytesIO
import numpy as np
import PIL
from PIL import Image
import pickle
import sys
import pandas as pd
import src.templates as templates
import datetime
import pytz

template = getattr(templates, 'fmow_template')

out = open(f"./datasets/csv/fmow.csv", "w")

out.write("title\tfilepath\n")

categories = [
    "airport", "airport_hangar", "airport_terminal", "amusement_park",
    "aquaculture", "archaeological_site", "barn", "border_checkpoint",
    "burial_site", "car_dealership", "construction_site", "crop_field", "dam",
    "debris_or_rubble", "educational_institution", "electric_substation",
    "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
    "gas_station", "golf_course", "ground_transportation_station", "helipad",
    "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
    "lighthouse", "military_facility", "multi-unit_residential",
    "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
    "parking_lot_or_garage", "place_of_worship", "police_station", "port",
    "prison", "race_track", "railway_bridge", "recreational_facility",
    "road_bridge", "runway", "shipyard", "shopping_mall",
    "single-unit_residential", "smokestack", "solar_farm", "space_facility",
    "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
    "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
    "wind_farm", "zoo"
]

metadata = pd.read_csv('./datasets/data/fmow_v1.1/rgb_metadata.csv')
####Filtering out the Training ID samples from the meta data (Code borrowed from WILDS Github)
split_array = np.zeros(len(metadata))
year = 2016
year_dt = datetime.datetime(year, 1, 1, tzinfo=pytz.UTC)
test_ood_mask = np.asarray(pd.to_datetime(metadata['timestamp']) >= year_dt)
# use 3 years of the training set as validation
year_minus_3_dt = datetime.datetime(year - 3, 1, 1, tzinfo=pytz.UTC)
val_ood_mask = np.asarray(
    pd.to_datetime(metadata['timestamp']) >= year_minus_3_dt) & ~test_ood_mask
ood_mask = test_ood_mask | val_ood_mask
idxs = np.arange(len(metadata))
split_mask = np.asarray(metadata['split'] == 'train')
idxs = idxs[~ood_mask & split_mask]
split_array[idxs] = 1
seq_mask = np.asarray(metadata['split'] == 'seq')
split_array = split_array[~seq_mask]
train_idx = np.where(split_array)[0]

root = './datasets/data/fmow_v1.1/images/'
count = 0
for idx in train_idx:
    img_file = f'rgb_img_{idx}.png'
    fp = os.path.join(root, img_file)
    class_name = metadata['category'][idx]
    y = categories.index(class_name)
    for t in template:
        count += 1
        caption = t(class_name)
        out.write("%s\t%s\n" % (caption, fp))