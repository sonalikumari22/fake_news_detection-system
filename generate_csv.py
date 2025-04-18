import pandas as pd

data = {
    'text': [
        "The economy is improving and stock markets are on the rise.",
        "Aliens have landed in London and taken over the government.",
        "Vaccines have proven to reduce the risk of severe COVID-19.",
        "Bill Gates created the virus to microchip people."
    ],
    'label': [
        "REAL",
        "FAKE",
        "REAL",
        "FAKE"
    ]
}

df = pd.DataFrame(data)
df.to_csv("news.csv", index=False)
print("✅ news.csv file created!")




