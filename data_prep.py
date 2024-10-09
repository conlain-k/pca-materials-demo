# Sweeps through the Micro2D dataset to build a mini dataset for regression and classification

from h5py import File
import numpy as np

micro_f = File("MICRO2D_homogenized.h5")

dataset_names = sorted(list(micro_f.keys()))

print(dataset_names)

num_per_ds = 100

num_total = num_per_ds * len(dataset_names)

output_f = File("materials_data.h5", "w")
output_f.create_dataset(
    "micros",
    (num_total, 256, 256),
    chunks=(16, 256, 256),
    compression="gzip",
    compression_opts=9,
    shuffle=True,
    dtype=int,
)
output_f.create_dataset("responses", (num_total, 2), dtype=float)
# integer encoding of clases
output_f.create_dataset("classes", (num_total, 1), dtype=int)

# sweep over datasets and collect
for ind, key in enumerate(dataset_names):
    print(f"writing {key}")
    # get write bounds
    start = ind * num_per_ds
    stop = start + num_per_ds
    # get first set of micros
    output_f["micros"][start:stop] = micro_f[key][key][:num_per_ds]
    # get high-contrast thermal props
    output_f["responses"][start:stop] = micro_f[key]["homogenized_thermal"][
        :num_per_ds, -1, :
    ]

    # also write classes
    output_f["classes"][start:stop] = ind

for k in output_f.keys():
    print(k, output_f[k].shape)
