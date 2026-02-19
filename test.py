import pandas as pd

sec_fp = r"D:\Hurricane Paper\second_2022_9_28.csv.gz"
df = pd.read_csv(sec_fp)

seg_id = "ebcd8c9df037ddb503041c82c980a641"

cols = [
    "gps_timestamp",
    "gps_speed",
    "accel_mphps"
]

d = df[df["segment_id"] == seg_id][cols].copy()
d["gps_timestamp"] = pd.to_datetime(d["gps_timestamp"])
d = d.sort_values("gps_timestamp").reset_index(drop=True)

print(d.head(20))
