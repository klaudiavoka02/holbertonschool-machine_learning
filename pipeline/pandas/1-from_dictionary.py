import pandas as pd

dictionary = {
    "First": ["0.0",
              "0.5",
              "1.0",
              "1.5",
              ],
    "Second": [
        "one",
        "two",
        "three",
        "four",
    ],
}
index_name = ["A", "B", "C", "D"]
df = pd.DataFrame(dictionary, index=index_name)

print(df)