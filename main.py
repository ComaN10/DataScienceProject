import pandas as pd
import numpy as np

diabetes_dataset = pd.read_csv(filepath_or_buffer="fetal_health.csv")

print(diabetes_dataset.describe().T)
print(diabetes_dataset.info())

for colum in diabetes_dataset.columns:

    print(diabetes_dataset[colum].quantile([0.25, 0.5, 0.75]))
    values = np.array(diabetes_dataset[colum])
    print(colum)
    #print(values[0:10])
    print(np.mean(values))
    print(np.std(values))
    print(np.min(values))
    print(np.max(values))
    print(np.median(values))

    print(colum,np.mean(values),np.std(values),np.min(values),np.max(values),np.median(values))

    print("\n\n\n")


